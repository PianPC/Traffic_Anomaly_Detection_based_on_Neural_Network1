# # 使用DNN模型
# python realtime_monitor.py --model_type DNN

# # 使用LSTM模型
# python realtime_monitor.py --model_type LSTM
# 实时流量监测脚本（支持DNN/LSTM双模型）
import argparse
import tensorflow as tf
from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import time
import os
from collections import defaultdict
import threading
from pathlib import Path

# ----------------- 配置部分 -----------------
# 基础路径配置
BASE_DIR = Path(__file__).parent.parent  # 项目根目录
MODELS_DIR = BASE_DIR / 'models'         # 模型存储目录

# 模型参数配置字典
# 包含DNN和LSTM两种模型的配置参数
CONFIG = {
    'DNN': {  # 深度神经网络配置
        'model_path': MODELS_DIR / 'traffic_model.keras',  # 模型文件路径
        'requires_window': False  # 不需要时间窗口数据
    },
    'LSTM': {  # 长短期记忆网络配置
        'model_path': MODELS_DIR / 'lstm_traffic_model.keras',
        'window_size': 1000,   # 时间窗口大小（与训练时PAST_HISTRY一致）
        'step': 6,            # 滑动步长（与训练时STEP一致）
        'requires_window': True  # 需要时间窗口数据
    }
}

# 特征列定义（必须与训练时的特征顺序完全一致）
FEATURE_COLUMNS = [
    'Bwd_Packet_Length_Min',     # 后向包最小长度
    'Subflow_Fwd_Bytes',         # 前向子流字节总数
    'Total_Length_of_Fwd_Packets', # 前向包总长度
    'Fwd_Packet_Length_Mean',    # 前向包平均长度
    'Bwd_Packet_Length_Std',     # 后向包长度标准差
    'Flow_Duration',             # 流持续时间
    'Flow_IAT_Std',              # 流到达时间标准差
    'Init_Win_bytes_forward',    # 前向初始窗口大小
    'Bwd_Packets/s',             # 每秒后向包数量
    'PSH_Flag_Count',            # PSH标记计数
    'Average_Packet_Size'        # 平均包大小
]

# ----------------- 全局变量 -----------------
# 流表数据结构说明：
# 使用defaultdict自动创建新流记录，键为五元组（src_ip, src_port, dst_ip, dst_port, proto）
# 每个流记录包含以下字段：
flow_table = defaultdict(lambda: {
    'start_time': None,       # 流开始时间
    'last_seen': None,        # 最后观察到的时间
    'fwd_packets': [],        # 前向包大小列表（用于统计）
    'bwd_packets': [],        # 后向包大小列表
    'timestamps': [],         # 包到达时间戳序列
    'psh_flags': 0,           # PSH标记计数器
    'init_win_bytes_fwd': None,  # 前向初始窗口大小（仅记录第一个包）
    'subflow_fwd_bytes': 0,   # 前向子流字节累计
    'feature_window': []      # LSTM专用特征缓存（存储历史特征序列）
})

# 全局模型相关变量
model = None        # 当前加载的TF模型
current_config = None  # 当前模型配置
prediction_lock = threading.Lock()  # 预测线程锁（保证预测顺序）

# ----------------- 核心逻辑函数 -----------------
def init_resources(model_type):
    """初始化模型和预处理工具
    参数：
        model_type : 模型类型 'DNN' 或 'LSTM'
    """
    global model, current_config
    
    # 设置当前配置
    current_config = CONFIG[model_type]
    print(f"正在加载 {model_type} 模型...")
    
    try:
        # 加载TensorFlow模型
        model = tf.keras.models.load_model(current_config['model_path'])
    except Exception as e:
        print(f"资源加载失败: {str(e)}")
        exit(1)


def packet_handler(pkt):
    # print(f"[DEBUG] 捕获到数据包: {pkt.summary()}")
    """数据包处理函数（Scapy回调）
    功能：
        1. 解析网络包的五元组信息
        2. 更新流统计信息
        3. （LSTM）缓存特征窗口数据
    参数：
        pkt : Scapy解析的网络包对象
    """
    # 过滤非IP/TCP-UDP包
    if not (IP in pkt and (TCP in pkt or UDP in pkt)):
        return

    # === 五元组提取 ===
    src_ip, dst_ip = pkt[IP].src, pkt[IP].dst  # IP地址
    proto = pkt[IP].proto                       # 协议类型（6=TCP, 17=UDP）
    # 端口提取（根据协议类型选择TCP/UDP层）
    src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
    dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
    # 生成双向流统一键（排序IP对和端口对）
    flow_key = tuple(sorted(((src_ip, src_port), (dst_ip, dst_port))) + [proto])

    # === 流记录初始化 ===
    flow = flow_table[flow_key]
    if flow['start_time'] is None:
        flow.update({
            'start_time': pkt.time,  # 记录第一个包的时间
            # 提取TCP初始窗口大小（仅第一个包有效）
            'init_win_bytes_fwd': pkt[TCP].window if TCP in pkt else 0
        })

    # === 更新流统计 ===
    packet_size = len(pkt)  # 当前包总长度
    # 判断包方向（前向：src_ip是流的发起方）
    is_forward = (pkt[IP].src == src_ip)
    if is_forward:
        flow['fwd_packets'].append(packet_size)
        flow['subflow_fwd_bytes'] += packet_size
    else:
        flow['bwd_packets'].append(packet_size)

    # 更新时间序列和PSH标记
    flow['timestamps'].append(pkt.time)
    if TCP in pkt and pkt[TCP].flags & 0x08:  # 检测PSH标记
        flow['psh_flags'] += 1
    flow['last_seen'] = pkt.time  # 更新最后活跃时间

    # === 新增DNN实时预测触发 ===
    if not current_config['requires_window']:  # DNN模型
        features = extract_features(flow)
        threading.Thread(target=async_predict, args=(features,)).start()

    # 提取特征向量
    features = extract_features(flow)

    # === 根据模型类型立即触发预测 ===
    if current_config['requires_window']:  # LSTM处理
        flow['feature_window'].append(features)
        # 当窗口达到大小时触发预测
        if len(flow['feature_window']) >= current_config['window_size']:
            window_data = flow['feature_window'][-current_config['window_size']::current_config['step']]
            if len(window_data) == current_config['window_size'] // current_config['step']:
                threading.Thread(target=async_predict, args=(window_data,)).start()
                flow['feature_window'].clear()
    else:  # DNN处理
        threading.Thread(target=async_predict, args=([features],)).start()

def extract_features(flow):
    """从流记录中提取特征向量
    返回：
        按FEATURE_COLUMNS顺序排列的特征列表
    """
    # 后向包统计
    bwd_packets = flow['bwd_packets'] or [0]  # 处理空列表情况
    bwd_min = min(bwd_packets) if bwd_packets else 0.0
    bwd_std = np.std(bwd_packets).astype('float32') if len(bwd_packets)>=2 else 0.0

    # 前向包统计
    fwd_packets = flow['fwd_packets'] or [0]
    fwd_total = sum(fwd_packets)
    fwd_mean = np.mean(fwd_packets).astype('float32') if fwd_packets else 0.0

    # 时间统计
    timestamps = flow['timestamps'] or [0]
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps)>=2 else 1e-6
    iat_std = np.std(np.diff(timestamps)).astype('float32') if len(timestamps)>=2 else 0.0

    # 其他特征计算
    bwd_packets_per_sec = len(bwd_packets)/flow_duration if flow_duration >0 else 0.0
    avg_packet_size = (sum(fwd_packets)+sum(bwd_packets))/(len(fwd_packets)+len(bwd_packets)) if (fwd_packets or bwd_packets) else 0.0

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

def async_predict(features):
    try:
        # 统一转换为numpy数组
        features = np.array(features, dtype=np.float32)
        
        # ===== 新增维度校验逻辑 =====
        if features.size == 0:
            raise ValueError("空特征输入")
            
        expected_size = 11 * (current_config['window_size'] 
                            if current_config['requires_window'] else 1)
        
        if features.size != expected_size:
            print(f"[ERROR] 特征数量异常: 期望{expected_size} 实际{features.size}")
            return
        # ========================
        
        # 正确维度重塑
        if current_config['requires_window']:
            # LSTM输入需为 (1, window_size, 11)
            input_data = features.reshape(1, 
                                        current_config['window_size'], 
                                        len(FEATURE_COLUMNS))
        else:
            # DNN输入需为 (1, 11)
            input_data = features.reshape(1, -1)
        
        # 添加调试日志
        print(f"[DEBUG] 最终输入维度: {input_data.shape}")
        
        # 执行预测
        prediction = model.predict(input_data, verbose=0)
        # ...后续处理...
        
    except Exception as e:
        print(f"预测流程异常: {str(e)}")

def cleanup_flows():
    """流清理函数（后台线程）
    功能：
        1. 定期检查并移除超时流
        2. 对即将移除的流触发最终预测（LSTM）
    """
    while True:
        # 清理间隔 = 窗口大小/4（默认30秒）
        time.sleep(current_config.get('window_size', 120) // 4)
        
        current_time = time.time()
        # 遍历流表副本（避免修改字典时迭代）
        for flow_key in list(flow_table.keys()):
            flow = flow_table[flow_key]
            
            # 判断流是否超时（120秒无活动）
            if flow['last_seen'] and (current_time - flow['last_seen'] > 120):
                # === LSTM最终预测 ===
                if current_config['requires_window'] and flow['feature_window']:
                    # 提取剩余窗口数据
                    window = flow['feature_window'][-current_config['window_size']::current_config['step']]
                    if len(window) >= (current_config['window_size'] // current_config['step']):
                        async_predict(window[:current_config['window_size']//current_config['step']])
                
                # 删除流记录
                del flow_table[flow_key]

# ----------------- 主程序 -----------------
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="实时流量异常检测系统")
    parser.add_argument('--model_type', choices=['DNN', 'LSTM'], required=True,
                       help="选择检测模型类型：DNN 或 LSTM")
    args = parser.parse_args()

    # 初始化模型和资源
    init_resources(args.model_type)

    # 启动后台清理线程（daemon=True随主线程退出）
    threading.Thread(target=cleanup_flows, daemon=True).start()

    # 打印启动信息
    print(f"启动 {args.model_type} 流量监测...")
    
    # 启动抓包（使用Scapy）
    sniff(
        prn=packet_handler,    # 每个包的回调处理
        filter="tcp or udp",   # 过滤TCP/UDP流量
        store=False,           # 不存储原始包（节省内存）
        stop_filter=lambda x: False  # 持续运行直到手动终止
    )