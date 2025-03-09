# 实时监测脚本框架
import tensorflow as tf
from scapy.all import sniff
import pyshark
import numpy as np
import subprocess
import os

def packet_handler(pkt):
    """处理每个捕获的数据包"""
    if not (IP in pkt and (TCP in pkt or UDP in pkt)): 
        """
        IP in pkt：检查数据包是否包含IP层
        TCP in pkt or UDP in pkt：检查数据包是否包含TCP或UDP层（传输层）。
        只处理 ​IP + TCP 或 ​IP + UDP 的组合流量（如HTTP、DNS等）
        过滤以下数据包：
            非IP流量（如ARP、ICMP-over-L2）。
            纯IP层无传输层的数据（如OSPF路由协议）
        """
        return  # 仅处理IP层流量

    # 提取五元组
    """
    pkt结构示例
    pkt = Ether(src="00:11:22:33:44:55")/IP(src="192.168.1.1", dst="8.8.8.8")/UDP(sport=1234, dport=53)/DNS(...)
    """
    print(pkt)
    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    proto = pkt[IP].proto
    src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
    dst_port = pcpket[TCP].dport if TCP in pkt else pkt[UDP].dport

    # 生成流标识键（双向流统一处理）
    flow_key = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))) + [proto])    # 无论数据包是 A→B 还是 B→A，排序后键值相同，统一视为同一流。

    # 更新流表信息
    update_flow_stats(flow_key, pkt)

def update_flow_stats(flow_key, pkt):
    """更新流统计特征"""
    if flow_key not in flow_table:
        # 初始化流记录（完整特征集需与训练数据一致）
        flow_table[flow_key] = {
            'start_time': time.time(),
            'last_seen': time.time(),
            'fwd_packets': [],
            'bwd_packets': [],
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            'psh_flags': 0,
            # 其他特征字段...
        }
    
    # 更新包方向及统计信息
    is_forward = (pkt[IP].src == flow_key[0][0])
    packet_size = len(pkt)
    if is_forward:
        flow_table[flow_key]['fwd_packets'].append(packet_size)
        flow_table[flow_key]['fwd_bytes'] += packet_size
    else:
        flow_table[flow_key]['bwd_packets'].append(packet_size)
        flow_table[flow_key]['bwd_bytes'] += packet_size
    
    # 更新PSH标志（TCP特定）
    if TCP in pkt and pkt[TCP].flags & 0x08:  # PSH标志位
        flow_table[flow_key]['psh_flags'] += 1

    # 更新时间戳
    flow_table[flow_key]['last_seen'] = time.time()

def extract_features(flow):
    """
    从流对象中提取模型所需的特征（必须与训练时的顺序一致）
    参数说明：
      flow : 实时流量流对象，应包含以下属性：
        - bwd_packet_lengths  : 后向包长度列表
        - fwd_packet_lengths  : 前向包长度列表
        - subflow_fwd_bytes   : 子流前向总字节数
        - timestamps          : 包到达时间戳列表
        - init_win_bytes_fwd  : 前向初始窗口字节数
        - psh_flag_count      : PSH标志出现次数
    """
    # 后向包长度统计
    bwd_lengths = flow.bwd_packet_lengths or [0.0]
    bwd_min = min(bwd_lengths)
    bwd_std = np.std(bwd_lengths).astype('float32')

    # 前向包长度统计（从通信的发起方，如客户端，发送到接收方，如服务器，的数据包）
    fwd_lengths = flow.fwd_packet_lengths or [0.0]
    fwd_total = sum(fwd_lengths)
    fwd_mean = np.mean(fwd_lengths).astype('float32')

    # 流时间统计
    time_diffs = np.diff(sorted(flow.timestamps)) if len(flow.timestamps) >=2 else [0.0]
    flow_duration = (flow.timestamps[-1] - flow.timestamps[0]) if flow.timestamps else 0.0
    iat_std = np.std(time_diffs).astype('float32')

    # 其他特征计算
    bwd_packets_per_sec = (len(bwd_lengths) / flow_duration) if flow_duration > 0 else 0.0
    avg_packet_size = ((sum(fwd_lengths) + sum(bwd_lengths)) / 
                      (len(fwd_lengths) + len(bwd_lengths))).astype('float32') if (fwd_lengths or bwd_lengths) else 0.0

    # 按feature_columns顺序返回特征值
    return [
        bwd_min,                     # Bwd_Packet_Length_Min
        flow.subflow_fwd_bytes,      # Subflow_Fwd_Bytes
        fwd_total,                   # Total_Length_of_Fwd_Packets
        fwd_mean,                    # Fwd_Packet_Length_Mean
        bwd_std,                     # Bwd_Packet_Length_Std
        flow_duration,               # Flow_Duration
        iat_std,                     # Flow_IAT_Std
        flow.init_win_bytes_fwd,     # Init_Win_bytes_forward
        bwd_packets_per_sec,         # Bwd_Packets/s
        flow.psh_flag_count,         # PSH_Flag_Count
        avg_packet_size              # Average_Packet_Size
    ]

# 实时流量处理函数
def packet_handler(pkt):
    try:
        # Step 1: 特征提取（需与训练时一致）
        features = extract_features(pkt)  # 需要实现特征提取函数
        
        # Step 2: 数据预处理
        processed_features = preprocess(features)  # 标准化/归一化
        
        # Step 3: 模型推理
        dnn_pred = dnn_model.predict(np.array([processed_features]))
        lstm_pred = lstm_model.predict(np.array([processed_features]))
        
        # Step 4: 决策融合
        final_pred = (dnn_pred + lstm_pred) / 2
        
        # Step 5: 告警触发
        if final_pred > 0.5:
            send_alert(pkt, final_pred)
            
    except Exception as e:
        print(f"处理异常: {str(e)}")

# 启动抓包
sniff(prn=packet_handler, filter="tcp or udp", store=0)

# 流（Flow）​ 是由多个数据包组成的通信上下文，其统计特征（如总包数、持续时间、包长方差等）需要基于多个连续数据包的聚合计算。直接在每个数据包上调用extract_features无法获得完整的流级特征，导致模型输入不完整。
# 定义流表字典，用于存储活跃流
flow_table = {}

#----------------------------
# 获取当前文件（realtime.py）的绝对路径，并计算模型路径
current_dir = os.path.dirname(__file__)
dnn_model_path = os.path.join(current_dir, '..', 'models', 'traffic_model.keras')
lstm_model_path = os.path.join(current_dir, '..', 'models', 'lstm_traffic_model.keras')

# 加载预训练模型
dnn_model = tf.keras.models.load_model(dnn_model_path)
lstm_model = tf.keras.models.load_model(lstm_model_path)

# 假设已捕获到一个流对象flow
features =  (flow)  # 提取特征

# 转换为模型输入格式（注意数据类型和维度）
input_data = np.array([features], dtype='float32')

# 执行预测
prediction = model.predict(input_data)
class_id = np.argmax(prediction, axis=1)[0]

# 输出结果（根据你的标签编码）
print(f"预测结果: 类别 {class_id} - {'正常流量' if class_id == 0 else '攻击流量'}")
