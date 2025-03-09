# 实时流量监测脚本
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import time
import os
from collections import defaultdict

# ----------------- 配置部分 -----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__),  '..', 'models', 'traffic_model.keras')
FLOW_TIMEOUT = 120  # 流超时时间（秒）
FEATURE_COLUMNS = [
    'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes', 'Total_Length_of_Fwd_Packets',
    'Fwd_Packet_Length_Mean', 'Bwd_Packet_Length_Std', 'Flow_Duration',
    'Flow_IAT_Std', 'Init_Win_bytes_forward', 'Bwd_Packets/s',
    'PSH_Flag_Count', 'Average_Packet_Size'
]

# ----------------- 全局流表 -----------------
flow_table = defaultdict(lambda: {
    'start_time': None,
    'last_seen': None,
    'fwd_packets': [],
    'bwd_packets': [],
    'timestamps': [],
    'psh_flags': 0,
    'init_win_bytes_fwd': None,
    'subflow_fwd_bytes': 0
})

# ----------------- 核心逻辑 -----------------
def packet_handler(pkt):
    """处理每个数据包，更新流统计"""
    if not (IP in pkt and (TCP in pkt or UDP in pkt)):
        return

    # 提取五元组（双向流统一处理）
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
    proto = pkt[IP].proto
    src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
    dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
    flow_key = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))) + [proto])

    # 初始化流记录
    flow = flow_table[flow_key]
    if flow['start_time'] is None:
        flow.update({
            'start_time': pkt.time,
            'init_win_bytes_fwd': pkt[TCP].window if TCP in pkt else 0
        })

    # 更新流统计
    packet_size = len(pkt)
    is_forward = (pkt[IP].src == src_ip)  # 判断包方向
    if is_forward:
        flow['fwd_packets'].append(packet_size)
        flow['subflow_fwd_bytes'] += packet_size
    else:
        flow['bwd_packets'].append(packet_size)

    # 其他特征更新
    flow['timestamps'].append(pkt.time)
    if TCP in pkt and pkt[TCP].flags & 0x08:
        flow['psh_flags'] += 1
    flow['last_seen'] = pkt.time

def extract_features(flow):
    """从流记录中提取模型所需特征"""
    # 后向包统计
    bwd_packets = flow['bwd_packets'] or [0]
    bwd_min = min(bwd_packets)
    bwd_std = np.std(bwd_packets).astype('float32')

    # 前向包统计
    fwd_packets = flow['fwd_packets'] or [0]
    fwd_total = sum(fwd_packets)
    fwd_mean = np.mean(fwd_packets).astype('float32')

    # 时间统计
    timestamps = flow['timestamps'] or [0]
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) >=2 else 0.0
    iat_std = np.std(np.diff(timestamps)).astype('float32') if len(timestamps)>=2 else 0.0

    # 其他特征
    bwd_packets_per_sec = len(bwd_packets)/flow_duration if flow_duration >0 else 0.0
    avg_packet_size = (sum(fwd_packets) + sum(bwd_packets)) / (len(fwd_packets)+len(bwd_packets)) if (fwd_packets or bwd_packets) else 0.0

    # 严格按FEATURE_COLUMNS顺序返回
    return [
        bwd_min,                     # Bwd_Packet_Length_Min
        flow['subflow_fwd_bytes'],    # Subflow_Fwd_Bytes
        fwd_total,                   # Total_Length_of_Fwd_Packets
        fwd_mean,                    # Fwd_Packet_Length_Mean
        bwd_std,                     # Bwd_Packet_Length_Std
        flow_duration,               # Flow_Duration
        iat_std,                     # Flow_IAT_Std
        flow['init_win_bytes_fwd'],   # Init_Win_bytes_forward
        bwd_packets_per_sec,         # Bwd_Packets/s
        flow['psh_flags'],           # PSH_Flag_Count
        avg_packet_size              # Average_Packet_Size
    ]

def cleanup_flows():
    """定期清理超时流并触发预测"""
    current_time = time.time()
    for flow_key in list(flow_table.keys()):
        flow = flow_table[flow_key]
        if flow['last_seen'] and (current_time - flow['last_seen'] > FLOW_TIMEOUT):
            # 提取特征并预测
            features = extract_features(flow)
            predict_anomaly(features)
            del flow_table[flow_key]

def predict_anomaly(features):
    """使用加载的模型进行预测"""
    input_data = np.array([features], dtype='float32')
    prediction = model.predict(input_data)
    class_id = np.argmax(prediction, axis=1)[0]
    print(f"流量预测: 类别 {class_id} - {'正常' if class_id ==0 else '可疑'}")

# ----------------- 主程序 -----------------
if __name__ == "__main__":
    # 加载预训练模型
    model = tf.keras.models.load_model(MODEL_PATH)

    # 启动抓包（全端口）
    sniff(prn=packet_handler, filter="tcp or udp", store=0)
    """
    prn，​回调函数，每捕获一个包立即调用，packet_handler 接收 pkt 参数处理
    filter，​BPF 过滤表达式，指定捕获的流量类型
    store，​是否在内存中保存原始包
    """

    # 启动流清理线程（需另开线程）
    while True:
        cleanup_flows()
        time.sleep(10)