# 文件路径：project_root/modules/historical_predictor.py
"""
python modules/historical_predictor.py --model_type DNN --data_path E:\workplace\Code\VSCodeProject\Traffic_Anomaly_Detection_based_on_Neural_Network\category\binary_classification.csv --mode predict
历史数据预测与评估模块
功能：
1. 加载历史数据集（CSV）
2. 批量预测并保存结果
3. 生成评估报告
"""

import argparse
import pandas as pd
from sklearn.metrics import classification_report
from realtime_monitor import init_system, CONFIG, model  # 复用已有资源

class HistoricalEvaluator:
    def __init__(self, model_type):
        self.model_type = model_type
        self.scaler = None  # 复用实时系统的标准化器
    
    def load_data(self, csv_path, has_labels=True):
        df = pd.read_csv(csv_path)
        
        # 改为从 CONFIG 获取特征列
        required_cols = CONFIG[self.model_type]['feature_columns']
        if has_labels:
            required_cols += ['Label']
        
        # 验证列名
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"数据集缺少列: {missing}")
        
        # 提取数据
        X = df[CONFIG[self.model_type]['feature_columns']]
        y = df['Label'] if has_labels else None
        return X, y
    
    def predict(self, X):
        """批量预测接口"""
        return model.predict(X)
    
    def evaluate(self, X, y_true):
        """生成评估报告"""
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred)
        return report

if __name__ == "__main__":
    # 命令行接口
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", required=True, choices=["DNN", "LSTM"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mode", choices=["predict", "evaluate"], required=True)
    args = parser.parse_args()
    
    # 初始化系统（复用实时监测的模型加载）
    init_system(args.model_type)
    
    # 执行任务
    evaluator = HistoricalEvaluator(args.model_type)
    if args.mode == "predict":
        X, _ = evaluator.load_data(args.data_path, has_labels=False)
        predictions = evaluator.predict(X)
        pd.DataFrame(predictions).to_csv("predictions.csv")
    else:
        X, y = evaluator.load_data(args.data_path, has_labels=True)
        print(evaluator.evaluate(X, y))