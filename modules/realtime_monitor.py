# 运行命令
# python modules/realtime_monitor.py --model_type DNN

#!/usr/bin/env python
# coding=UTF-8
"""
实时流量异常检测系统（支持DNN/LSTM双模型）
版本：2.1
更新日志：
- 增强线程安全性
- 添加输入验证
- 优化批量处理
- 强化异常处理
"""

# %% 导入库
import os
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 新增：禁用oneDNN警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # 新增：减少TensorFlow日志

import argparse
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import time
import logging
from collections import defaultdict
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import psutil

# %% 全局配置
predict_queue = []  # 确保在全局作用域初始化
# 线程池配置
from concurrent.futures import ThreadPoolExecutor
predict_executor = ThreadPoolExecutor(max_workers=4)

#region 配置部分
# ----------------- 日志配置 -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("realtime_monitor.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrafficMonitor")

# ----------------- 路径配置 -----------------
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
MODELS_DIR = BASE_DIR / "models"         # 模型存储目录

# ----------------- 模型参数配置 -----------------
FEATURE_COLUMNS = [  # 必须与训练数据严格一致
    'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
    'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
    'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
    'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
    'Average_Packet_Size'
]

# %% 新增全局状态类
class SystemState:
    """管理全局状态"""
    def __init__(self):
        self.label_map = self._load_label_map()
        
    def _load_label_map(self):
        """加载标签映射文件"""
        label_path = MODELS_DIR / "label_mapping.json"
        try:
            with open(label_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"标签映射加载失败: {str(e)}")
            exit(1)

# 初始化全局状态
system_state = SystemState()

CONFIG = {
    "DNN": {
        "feature_columns": FEATURE_COLUMNS, 
        "model_path": MODELS_DIR / "traffic_model.keras",
        "requires_window": False,
        "max_batch_size": 128,           # 新增：批量处理大小
        "input_dim": len(FEATURE_COLUMNS),
        "normal_class": "BENIGN"  # 替代原来的threshold
    },
    "LSTM": {
        "feature_columns": FEATURE_COLUMNS,
        "model_path": MODELS_DIR / "lstm_traffic_model.keras",
        "window_size": 1000,
        "step": 6,
        "max_sequence_length": 166,      # 新增：1000//6=166
        "requires_window": True,
        "input_dim": len(FEATURE_COLUMNS),
        "min_window": 30,                 # 新增：最小有效窗口
        "normal_class": "BENIGN"
    }
}

# ----------------- 流表配置 -----------------
FLOW_TIMEOUT = 120                       # 流超时时间（秒）
MAX_PACKET_SIZE = 9000                   # 单个包最大字节数
#endregion

# %% 全局变量
#region 全局状态
flow_table = defaultdict(lambda: {
    "start_time": None,
    "last_seen": None,
    "fwd_packets": [],
    "bwd_packets": [],
    "timestamps": [],
    "psh_flags": 0,
    "init_win_bytes_fwd": None,
    "subflow_fwd_bytes": 0,
    "feature_window": []
})

flow_table_lock = threading.Lock()       # 流表访问锁
model = None                              # 当前模型实例
current_config = None                     # 当前配置参数
predict_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Predictor")
#endregion

# %% 核心功能
#region 初始化模块
def init_system(model_type: str):
    """
    初始化系统资源
    Args:
        model_type: 模型类型 DNN/LSTM
    """
    global model, current_config
    
    logger.info(f"正在初始化 {model_type} 系统...")
    
    # 参数校验
    if (cfg := CONFIG.get(model_type)) is None:
        logger.error(f"无效的模型类型: {model_type}")
        exit(1)
    current_config = cfg
    
    # 模型加载
    try:
        logger.info(f"加载模型: {cfg['model_path'].resolve()}")
        load_start = time.time()
        model = tf.keras.models.load_model(
            cfg["model_path"],
            compile=False  # 预测不需要优化器
        )
        logger.info(f"模型加载成功 ({time.time()-load_start:.2f}s)")
        
        # 新增：模型输入验证
        validate_model_input()
        
    except Exception as e:
        logger.exception("模型加载失败")
        exit(1)

    return model

def validate_model_input():
    """验证模型输入是否符合预期"""
    try:
        # 生成测试输入
        if current_config["requires_window"]:
            test_input = np.zeros(
                (1, current_config["max_sequence_length"], current_config["input_dim"]),
                dtype=np.float32
            )
        else:
            test_input = np.zeros(
                (1, current_config["input_dim"]),
                dtype=np.float32
            )
        
        # 执行预测
        _ = model.predict(test_input, verbose=0)
        logger.debug("模型输入验证通过")
    except Exception as e:
        logger.error("模型输入验证失败: " + str(e))
        exit(1)
#endregion

#region 数据包处理
def packet_handler(pkt):
    """Scapy数据包处理回调"""
    try:
        # 基础过滤
        if not (IP in pkt and (TCP in pkt or UDP in pkt)):
            return

        # 五元组提取
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        proto = pkt[IP].proto
        layer = TCP if TCP in pkt else UDP
        src_port, dst_port = layer.sport, layer.dport
        
        # 生成流键（规范化双向流）
        flow_key = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))) + [proto])

        # 线程安全更新流表
        with flow_table_lock:
            update_flow_stats(flow_key, pkt)

        # 特征提取与预测触发
        features = extract_features(flow_key)
        trigger_prediction(flow_key, features)

    except Exception as e:
        logger.error(f"包处理异常: {str(e)}")

def update_flow_stats(flow_key, pkt):
    """更新流统计信息"""
    flow = flow_table[flow_key]
    
    # 初始化流记录
    if flow["start_time"] is None:
        flow.update({
            "start_time": pkt.time,
            "init_win_bytes_fwd": pkt[TCP].window if TCP in pkt else 0
        })

    # 包方向判断
    is_forward = pkt[IP].src == flow_key[0][0]
    packet_size = min(len(pkt), MAX_PACKET_SIZE)  # 限制最大尺寸

    # 更新统计字段
    if is_forward:
        flow["fwd_packets"].append(packet_size)
        flow["subflow_fwd_bytes"] += packet_size
    else:
        flow["bwd_packets"].append(packet_size)

    # 更新时间序列
    flow["timestamps"].append(pkt.time)
    flow["last_seen"] = pkt.time
    
    # 更新PSH标记
    if TCP in pkt and pkt[TCP].flags & 0x08:
        flow["psh_flags"] += 1
#endregion

#region 特征工程
def extract_features(flow_key) -> list:
    """从流记录中提取特征向量"""
    try:
        with flow_table_lock:
            flow = flow_table[flow_key]
            
            # 数值稳定性处理
            timestamps = flow["timestamps"] or [time.time()]
            flow_duration = max(timestamps[-1] - timestamps[0], 1e-6)
            
            # 包统计
            fwd_packets = flow["fwd_packets"][-1000:]  # 限制历史长度
            bwd_packets = flow["bwd_packets"][-1000:]
            
            # 特征计算
            return [
                min(bwd_packets) if bwd_packets else 0.0,  # Bwd_Min
                flow["subflow_fwd_bytes"],
                sum(fwd_packets),
                np.mean(fwd_packets).astype(np.float32) if fwd_packets else 0.0,
                np.std(bwd_packets).astype(np.float32) if len(bwd_packets)>=2 else 0.0,
                flow_duration,
                np.std(np.diff(timestamps)).astype(np.float32) if len(timestamps)>=2 else 0.0,
                flow["init_win_bytes_fwd"] or 0,
                len(bwd_packets)/flow_duration if flow_duration > 0 else 0.0,
                flow["psh_flags"],
                (sum(fwd_packets)+sum(bwd_packets))/(len(fwd_packets)+len(bwd_packets)+1e-6)
            ]
    except Exception as e:
        logger.error(f"特征提取失败: {str(e)}")
        return [0.0] * len(FEATURE_COLUMNS)
#endregion

#region 预测模块
def trigger_prediction(flow_key, features):
    """根据模型类型触发预测"""
    try:
        # DNN即时预测
        if not current_config["requires_window"]:
            if len(features) != current_config["input_dim"]:
                logger.warning(f"特征维度错误: {len(features)}")
                return
            
            # 加入批量队列
            global predict_queue
            predict_queue.append(features)
            
            # 批量处理
            if len(predict_queue) >= current_config["max_batch_size"]:
                batch = np.array(predict_queue, dtype=np.float32)
                predict_queue.clear()
                predict_executor.submit(run_prediction, batch, [flow_key]*len(batch))
        
        # LSTM窗口预测
        else:
            with flow_table_lock:
                flow = flow_table[flow_key]
                flow["feature_window"].append(features)
                window = flow["feature_window"]
            
            # 窗口采样
            if len(window) >= current_config["window_size"]:
                sampled = window[-current_config["window_size"]::current_config["step"]]
                if len(sampled) >= current_config["min_window"]:
                    # 转换为模型输入格式
                    input_seq = np.array(sampled[-current_config["max_sequence_length"]:], dtype=np.float32)
                    predict_executor.submit(run_prediction, input_seq[np.newaxis, ...], flow_key)

    except Exception as e:
        logger.error(f"预测触发失败: {str(e)}")

def run_prediction(input_data, flow_key):
    """执行实际预测"""
    try:
        # 输入验证
        if np.isnan(input_data).any():
            logger.warning("输入包含NaN值")
            return
        
        # 维度调整
        if current_config["requires_window"]:
            if input_data.shape[1:] != (current_config["max_sequence_length"], current_config["input_dim"]):
                logger.error(f"LSTM输入维度错误: {input_data.shape}")
                return
        else:
            if input_data.shape[1] != current_config["input_dim"]:
                logger.error(f"DNN输入维度错误: {input_data.shape}")
                return
        
        # 执行预测
        start_time = time.time()
        preds = model.predict(input_data, verbose=0)
        logger.debug(f"预测耗时: {time.time()-start_time:.4f}s")
        
        # 结果处理
        process_predictions(preds, flow_key)

    except Exception as e:
        logger.error(f"预测执行失败: {str(e)}")

def process_predictions(preds, flow_key):
    """统一处理多分类预测结果"""
    label_map = system_state.label_map
    results = []
    try:
        # 获取配置信息
        is_multiclass = len(system_state.label_map) > 2
        
        # DNN模型处理（适配多分类）
        if not current_config["requires_window"]:
            if is_multiclass:
                # 多分类使用softmax
                probabilities = tf.nn.softmax(preds).numpy()
            else:
                # 二分类使用sigmoid
                probabilities = tf.sigmoid(preds).numpy()
        # LSTM模型处理
        else:
            probabilities = tf.nn.softmax(preds).numpy()

        # 统一结果处理
        for i, prob_vec in enumerate(probabilities):
            # 获取预测类别
            class_id = np.argmax(prob_vec)
            confidence = prob_vec[class_id]
            
            # 从标签映射获取类别信息
            class_info = system_state.label_map.get(str(class_id), "UNKNOWN")
            
            # 解析是否为异常（BENIGN类为正常）
            is_anomaly = (class_info != "BENIGN")
            status = "异常" if is_anomaly else "正常"
            
            # 构建结果字典
            result = {
                "flow_key": flow_key[i] if isinstance(flow_key, list) else flow_key,
                "model_type": "DNN" if not current_config["requires_window"] else "LSTM",
                "class_id": int(class_id),
                "class_name": class_info,
                "confidence": float(confidence),
                "is_anomaly": is_anomaly,
                "timestamp": time.time()
            }
            results.append(result)
            
            # 生成日志信息
            log_msg = (
                f"{status}流量 [{result['model_type']}] "
                f"类别: {class_info}({class_id}) "
                f"置信度: {confidence:.2%} "
                f"流: {result['flow_key']}"
            )
            
            # 差异化日志输出
            if is_anomaly:
                logger.warning(log_msg)
                # 此处可添加警报触发逻辑
            else:
                logger.info(log_msg)

        return results

    except Exception as e:
        logger.error(f"结果处理失败: {str(e)}")
        return []

#region 维护模块
def cleanup_flows():
    """定期清理过期流"""
    while True:
        time.sleep(FLOW_TIMEOUT // 2)  # 每60秒清理一次
        
        try:
            current_time = time.time()
            cleanup_count = 0
            
            with flow_table_lock:
                for flow_key in list(flow_table.keys()):
                    flow = flow_table[flow_key]
                    
                    # 判断超时
                    if flow["last_seen"] and (current_time - flow["last_seen"] > FLOW_TIMEOUT):
                        # LSTM窗口处理
                        if current_config["requires_window"] and flow["feature_window"]:
                            window = flow["feature_window"]
                            if len(window) >= current_config["min_window"]:
                                sampled = window[-current_config["window_size"]::current_config["step"]]
                                if len(sampled) >= current_config["min_window"]:
                                    input_seq = np.array(sampled, dtype=np.float32)
                                    run_prediction(input_seq[np.newaxis, ...], flow_key)
                        
                        del flow_table[flow_key]
                        cleanup_count += 1
            
            logger.info(f"流清理完成，删除{cleanup_count}个流，当前活跃流数: {len(flow_table)}")
            
        except Exception as e:
            logger.error(f"流清理失败: {str(e)}")
#endregion

# %% 主程序
if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser(description="实时流量异常检测系统")
    parser.add_argument("--model_type", choices=["DNN", "LSTM"], required=True)
    args = parser.parse_args()

    # 资源初始化
    start_mem = psutil.Process().memory_info().rss  # 获取内存占用（字节）
    
    init_system(args.model_type)
    
    try:
        # 启动维护线程
        threading.Thread(target=cleanup_flows, daemon=True, name="Cleaner").start()
        
        # 开始抓包
        logger.info(f"启动{args.model_type}监测，初始内存占用: {start_mem // 1024}KB")
        sniff(
            prn=packet_handler,
            filter="tcp or udp",
            store=False,
            stop_filter=lambda _: False,
            count=0  # 无限抓包
        )

    except KeyboardInterrupt:
        logger.info("用户终止操作")
    finally:
        # 资源清理
        predict_executor.shutdown()
        end_mem = psutil.Process().memory_info().rss
        logger.info(f"程序退出，峰值内存占用: {(end_mem - start_mem) // 1024}KB")