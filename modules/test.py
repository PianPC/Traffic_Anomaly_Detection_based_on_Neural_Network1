#!/usr/bin/env python
# coding=UTF-8
"""
实时流量异常检测系统（自适应多分类版本）
支持动态加载不同结构的DNN/LSTM模型，自动适配类别标签体系
"""

# %% [1] 环境配置
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import json
import time
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import psutil
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP

# %% [2] 全局配置
class GlobalConfig:
    """统一配置中心"""
    # 路径配置
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    
    # 网络配置
    NETWORK_INTERFACE = "eth0"
    MAX_PACKET_SIZE = 9000
    
    # 流表配置
    FLOW_TIMEOUT = 120  # 秒
    MAX_FLOW_ENTRIES = 10000
    
    # 线程配置
    MAX_WORKERS = 4
    
    # 特征列（必须与训练数据一致）
    FEATURE_COLUMNS = [
        'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size'
    ]

# %% [3] 日志配置
def setup_logging():
    """配置日志格式和处理器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("realtime_monitor.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TrafficMonitor")

logger = setup_logging()

# %% [4] 全局状态
class SystemState:
    """管理全局状态和共享资源"""
    def __init__(self):
        # 流表数据结构
        self.flow_table = defaultdict(self._new_flow_entry)
        self.flow_lock = Lock()
        
        # 模型相关状态
        self.model = None
        self.label_map = {}
        self.model_type = None
        self.model_config = {}
        
        # 预测队列
        self.predict_queue = []
        self.executor = ThreadPoolExecutor(max_workers=GlobalConfig.MAX_WORKERS)
    
    def _new_flow_entry(self):
        """初始化新的流表条目"""
        return {
            "start_time": None,
            "last_seen": None,
            "fwd_packets": [],
            "bwd_packets": [],
            "timestamps": [],
            "psh_flags": 0,
            "init_win_bytes_fwd": None,
            "subflow_fwd_bytes": 0,
            "feature_window": []
        }

system_state = SystemState()

# %% [5] 模型管理
class ModelLoader:
    """模型加载和验证器"""
    def __init__(self, model_type: str):
        self.model_type = model_type.upper()
        self.config = self._get_model_config()
        
    def _get_model_config(self):
        """获取模型配置参数"""
        config_map = {
            "DNN": {
                "input_dim": len(GlobalConfig.FEATURE_COLUMNS),
                "model_path": GlobalConfig.MODELS_DIR / "traffic_model.keras",
                "requires_window": False,
                "threshold": 0.65
            },
            "LSTM": {
                "input_dim": len(GlobalConfig.FEATURE_COLUMNS),
                "model_path": GlobalConfig.MODELS_DIR / "lstm_traffic_model.keras",
                "requires_window": True,
                "window_size": 1000,
                "step": 6,
                "min_window": 30
            }
        }
        if self.model_type not in config_map:
            raise ValueError(f"无效模型类型: {self.model_type}")
        return config_map[self.model_type]
    
    def load_model(self):
        """加载模型和标签映射"""
        # 加载模型
        logger.info(f"加载{self.model_type}模型: {self.config['model_path']}")
        model = tf.keras.models.load_model(
            self.config["model_path"],
            compile=False  # 预测不需要优化器
        )
        
        # 加载标签映射
        label_path = self.config["model_path"].parent / "label_mapping.json"
        with open(label_path, 'r') as f:
            label_map = json.load(f)
        
        # 验证标签映射完整性
        if not all(str(i) in label_map for i in range(len(label_map))):
            raise ValueError("标签映射不连续或缺失")
        
        return model, label_map

# %% [6] 流量处理
class PacketProcessor:
    """数据包处理核心逻辑"""
    @staticmethod
    def parse_packet(pkt):
        """解析数据包基本信息"""
        if not (IP in pkt and (TCP in pkt or UDP in pkt)):
            return None
        
        # 提取五元组
        src_ip = pkt[IP].src
        dst_ip = pkt[IP].dst
        proto = pkt[IP].proto
        layer = TCP if TCP in pkt else UDP
        src_port = layer.sport
        dst_port = layer.dport
        
        # 生成规范化的流键
        return tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))) + [proto])
    
    @staticmethod
    def update_flow_stats(flow_key, pkt):
        """更新流统计信息"""
        with system_state.flow_lock:
            flow = system_state.flow_table[flow_key]
            
            # 初始化流记录
            if flow["start_time"] is None:
                flow.update({
                    "start_time": pkt.time,
                    "init_win_bytes_fwd": pkt[TCP].window if TCP in pkt else 0
                })
            
            # 更新包统计
            is_forward = pkt[IP].src == flow_key[0][0]
            packet_size = min(len(pkt), GlobalConfig.MAX_PACKET_SIZE)
            
            if is_forward:
                flow["fwd_packets"].append(packet_size)
                flow["subflow_fwd_bytes"] += packet_size
            else:
                flow["bwd_packets"].append(packet_size)
            
            # 更新时间序列
            flow["timestamps"].append(pkt.time)
            flow["last_seen"] = pkt.time
            
            # 更新PSH标志
            if TCP in pkt and pkt[TCP].flags & 0x08:
                flow["psh_flags"] += 1

# %% [7] 特征工程
class FeatureExtractor:
    """实时特征计算引擎"""
    @staticmethod
    def extract(flow_key):
        """从流记录中提取特征向量"""
        with system_state.flow_lock:
            flow = system_state.flow_table[flow_key]
            
            # 获取时间序列数据
            timestamps = flow["timestamps"] or [time.time()]
            flow_duration = max(timestamps[-1] - timestamps[0], 1e-6)
            
            # 限制历史数据长度
            fwd_packets = flow["fwd_packets"][-1000:]
            bwd_packets = flow["bwd_packets"][-1000:]
            
            # 计算统计特征
            features = [
                min(bwd_packets) if bwd_packets else 0.0,  # Bwd_Packet_Length_Min
                flow["subflow_fwd_bytes"],  # Subflow_Fwd_Bytes
                sum(fwd_packets),  # Total_Length_of_Fwd_Packets
                np.mean(fwd_packets).astype(np.float32) if fwd_packets else 0.0,  # Fwd_Packet_Length_Mean
                np.std(bwd_packets).astype(np.float32) if len(bwd_packets)>=2 else 0.0,  # Bwd_Packet_Length_Std
                flow_duration,  # Flow_Duration
                np.std(np.diff(timestamps)).astype(np.float32) if len(timestamps)>=2 else 0.0,  # Flow_IAT_Std
                flow["init_win_bytes_fwd"] or 0,  # Init_Win_bytes_forward
                len(bwd_packets)/flow_duration if flow_duration > 0 else 0.0,  # Bwd_Packets/s
                flow["psh_flags"],  # PSH_Flag_Count
                (sum(fwd_packets)+sum(bwd_packets))/(len(fwd_packets)+len(bwd_packets)+1e-6)  # Average_Packet_Size
            ]
            
            # 窗口数据维护（LSTM专用）
            if system_state.model_config.get("requires_window", False):
                flow["feature_window"].append(features)
                flow["feature_window"] = flow["feature_window"][-system_state.model_config["window_size"]:]
            
            return features

# %% [8] 预测引擎
class PredictionEngine:
    """预测任务调度和执行"""
    @staticmethod
    def trigger(flow_key, features):
        """触发预测任务"""
        config = system_state.model_config
        
        # DNN即时预测
        if not config["requires_window"]:
            if len(features) != config["input_dim"]:
                logger.warning(f"特征维度错误: {len(features)}")
                return
            
            system_state.predict_queue.append((flow_key, features))
            
            # 批量处理
            if len(system_state.predict_queue) >= 128:
                batch = [f for _, f in system_state.predict_queue]
                keys = [k for k, _ in system_state.predict_queue]
                system_state.predict_queue.clear()
                system_state.executor.submit(
                    PredictionEngine.batch_predict, keys, np.array(batch)
                )
        
        # LSTM窗口预测
        else:
            with system_state.flow_lock:
                window = system_state.flow_table[flow_key]["feature_window"]
            
            if len(window) >= config["min_window"]:
                sampled = window[-config["window_size"]::config["step"]]
                if len(sampled) >= config["min_window"]:
                    input_seq = np.array(sampled[-config["window_size"]:], dtype=np.float32)
                    system_state.executor.submit(
                        PredictionEngine.single_predict, flow_key, input_seq[np.newaxis, ...]
                    )
    
    @staticmethod
    def batch_predict(flow_keys, features):
        """批量预测（DNN）"""
        try:
            # 执行预测
            predictions = system_state.model.predict(features, verbose=0)
            
            # 处理结果
            for key, pred in zip(flow_keys, predictions):
                class_id = int(pred > system_state.model_config["threshold"])
                confidence = float(pred[0] if class_id == 1 else 1 - pred[0])
                PredictionEngine.process_result(key, class_id, confidence)
        
        except Exception as e:
            logger.error(f"批量预测失败: {str(e)}")
    
    @staticmethod
    def single_predict(flow_key, input_seq):
        """单样本预测（LSTM）"""
        try:
            # 执行预测
            predictions = system_state.model.predict(input_seq, verbose=0)
            
            # 处理结果
            class_id = np.argmax(predictions)
            confidence = predictions[0][class_id]
            PredictionEngine.process_result(flow_key, int(class_id), float(confidence))
        
        except Exception as e:
            logger.error(f"单样本预测失败: {str(e)}")
    
    @staticmethod
    def process_result(flow_key, class_id, confidence):
        """统一处理预测结果"""
        label = system_state.label_map.get(str(class_id), "UNKNOWN")
        is_anomaly = (label != "BENIGN")
        
        # 日志输出
        log_msg = f"流量 {flow_key} - 类别: {label}({class_id}) 置信度: {confidence:.2%}"
        if is_anomaly:
            logger.warning(f"[异常] {log_msg}")
        else:
            logger.info(f"[正常] {log_msg}")

# %% [9] 系统维护
class SystemMaintenance:
    """后台维护任务"""
    @staticmethod
    def cleanup_flows():
        """定期清理过期流"""
        while True:
            time.sleep(GlobalConfig.FLOW_TIMEOUT // 2)
            
            try:
                current_time = time.time()
                removed = 0
                
                with system_state.flow_lock:
                    for key in list(system_state.flow_table.keys()):
                        flow = system_state.flow_table[key]
                        if (current_time - flow["last_seen"]) > GlobalConfig.FLOW_TIMEOUT:
                            del system_state.flow_table[key]
                            removed += 1
                
                logger.info(f"清理过期流完成，移除{removed}条，当前流表大小: {len(system_state.flow_table)}")
            
            except Exception as e:
                logger.error(f"流清理失败: {str(e)}")

# %% [10] 主程序
def main(model_type: str):
    """程序入口"""
    try:
        # 初始化模型
        loader = ModelLoader(model_type)
        system_state.model, system_state.label_map = loader.load_model()
        system_state.model_config = loader.config
        system_state.model_type = model_type
        
        # 启动维护线程
        threading.Thread(
            target=SystemMaintenance.cleanup_flows,
            daemon=True,
            name="FlowCleaner"
        ).start()
        
        # 开始抓包
        logger.info(f"启动{model_type}监测，接口: {GlobalConfig.NETWORK_INTERFACE}")
        sniff(
            iface=GlobalConfig.NETWORK_INTERFACE,
            prn=lambda pkt: process_packet(pkt),
            store=False,
            filter="tcp or udp",
            stop_filter=lambda _: False
        )
    
    except KeyboardInterrupt:
        logger.info("用户终止操作")
    finally:
        # 清理资源
        system_state.executor.shutdown()
        logger.info("系统安全关闭")

def process_packet(pkt):
    """包处理入口函数"""
    try:
        # 解析数据包
        flow_key = PacketProcessor.parse_packet(pkt)
        if not flow_key:
            return
        
        # 更新流表
        PacketProcessor.update_flow_stats(flow_key, pkt)
        
        # 提取特征并触发预测
        features = FeatureExtractor.extract(flow_key)
        PredictionEngine.trigger(flow_key, features)
    
    except Exception as e:
        logger.error(f"包处理异常: {str(e)}")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="实时流量异常检测系统")
    parser.add_argument("--model_type", choices=["DNN", "LSTM"], required=True)
    args = parser.parse_args()
    
    # 启动主程序
    main(args.model_type)