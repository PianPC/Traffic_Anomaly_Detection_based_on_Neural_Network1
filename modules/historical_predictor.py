# 文件路径：project_root/modules/historical_predictor.py
"""
python modules/historical_predictor.py --model_type DNN --data_path E:\workplace\Code\VSCodeProject\Traffic_Anomaly_Detection_based_on_Neural_Network\category\binary_classification.csv --mode predict
历史数据预测与评估模块
功能：
1. 加载历史数据集（CSV）
2. 批量预测并保存结果
3. 生成评估报告
"""

# historical_predictor.py
import argparse
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import realtime_monitor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("HistoricalPredictor")

class HistoricalEvaluator:
    def __init__(self, model_type):
        self.model_type = model_type
        self._validate_model_type()
        self.label_mapping = self._load_label_mapping()
        self.label_names = self._prepare_label_names()

    def _validate_model_type(self):
        """验证模型类型有效性"""
        valid_types = realtime_monitor.CONFIG.keys()
        if self.model_type not in valid_types:
            raise ValueError(f"无效模型类型: {self.model_type}，可选: {list(valid_types)}")

    def _load_label_mapping(self):
        """从JSON文件加载标签映射"""
        mapping_path = Path(__file__).parent.parent / "models" / "label_mapping.json"
        try:
            with open(mapping_path, 'r') as f:
                raw_mapping = json.load(f)
            
            # 转换并清洗映射关系
            cleaned_mapping = {}
            for str_id, label in raw_mapping.items():
                cleaned_label = self._clean_label_name(label)
                cleaned_mapping[cleaned_label] = int(str_id)
            
            # 添加未知类别处理
            if "UNKNOWN" not in cleaned_mapping.values():
                max_id = max(cleaned_mapping.values()) 
                cleaned_mapping["UNKNOWN"] = max_id + 1
            
            return cleaned_mapping
            
        except FileNotFoundError:
            raise RuntimeError(f"标签映射文件未找到: {mapping_path}")
        except json.JSONDecodeError:
            raise RuntimeError("标签映射文件格式错误，必须为合法JSON")
        except Exception as e:
            raise RuntimeError(f"加载标签映射失败: {str(e)}")

    def _clean_label_name(self, label):
        """统一标签格式"""
        return str(label).upper().replace(" ", "").strip()

    def _prepare_label_names(self):
        """准备分类报告用的标签名称"""
        inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return [inverse_mapping[i] for i in sorted(inverse_mapping.keys()) if i != self.label_mapping["UNKNOWN"]]

    def load_data(self, csv_path, has_labels=True):
        """加载并预处理数据"""
        logger.info(f"开始加载数据: {csv_path}")
        
        # 读取数据
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"原始数据维度: {df.shape}")
        except Exception as e:
            raise IOError(f"文件读取失败: {str(e)}")

        # 数据清洗流程
        df = self._clean_raw_data(df, has_labels)
        return self._process_features(df, has_labels)

    def _clean_raw_data(self, df, has_labels):
        """执行数据清洗"""
        # 删除全空行
        df = df.dropna(how='all')
        
        # 处理特征列
        feature_cols = realtime_monitor.CONFIG[self.model_type]['feature_columns']
        for col in feature_cols:
            # 移除非数字字符
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # 转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除无效数据
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=feature_cols)
        logger.info(f"清洗后数据维度: {df.shape}")
        
        # 标签处理
        if has_labels:
            df = self._process_labels(df)
        return df

    def _process_labels(self, df):
        """处理标签数据"""
        df['Label'] = df['Label'].apply(self._clean_label_name)
        
        # 映射标签ID
        df['Label'] = df['Label'].map(
            lambda x: self.label_mapping.get(x, self.label_mapping["UNKNOWN"])
        )
        
        # 检测未知标签
        unknown_labels = df[df['Label'] == self.label_mapping["UNKNOWN"]]
        if not unknown_labels.empty:
            invalid = unknown_labels['Label'].unique().tolist()
            raise ValueError(
                f"发现{len(unknown_labels)}条未知标签数据，示例: {invalid[:3]}..."
                f"\n请更新label_mapping.json文件"
            )
        return df

    def _process_features(self, df, has_labels):
        """处理特征矩阵"""
        feature_cols = realtime_monitor.CONFIG[self.model_type]['feature_columns']
        X = df[feature_cols].astype(np.float32)
        
        if has_labels:
            y = df['Label'].values.astype(np.int8)
            logger.info(f"类别分布:\n{df['Label'].value_counts().to_string()}")
            return X, y
        return X, None

    def predict(self, X):
        """执行批量预测"""
        logger.info(f"开始预测，样本数: {len(X)}")
        try:
            preds = realtime_monitor.model.predict(X)
            # 处理不同模型输出
            if self.model_type == "DNN" and preds.ndim == 1:
                return (preds > 0.5).astype(int)
            return np.argmax(preds, axis=1)
        except AttributeError:
            raise RuntimeError("模型未正确初始化")
        except Exception as e:
            raise RuntimeError(f"预测失败: {str(e)}")

    def evaluate(self, X, y_true):
        """生成评估报告"""
        y_pred = self.predict(X)
        return classification_report(
            y_true,
            y_pred,
            target_names=self.label_names,
            digits=4,
            zero_division=0
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="流量异常检测历史数据分析")
    parser.add_argument("--model_type", required=True, choices=["DNN", "LSTM"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mode", required=True, choices=["predict", "evaluate"])
    
    try:
        args = parser.parse_args()
        realtime_monitor.init_system(args.model_type)
        evaluator = HistoricalEvaluator(args.model_type)
        
        if args.mode == "predict":
            X, _ = evaluator.load_data(args.data_path, has_labels=False)
            predictions = evaluator.predict(X)
            pd.DataFrame(predictions).to_csv("predictions.csv", index=False)
            logger.info("预测结果已保存至 predictions.csv")
        else:
            X, y = evaluator.load_data(args.data_path, has_labels=True)
            report = evaluator.evaluate(X, y)
            print("\n分类评估报告:")
            print(report)
            
    except Exception as e:
        logger.error(f"运行终止: {str(e)}")
        exit(1)