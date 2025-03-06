#!/usr/bin/env python
# coding=UTF-8  
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 30000
CSV_FILE_PATH = 'E:\\workplace\\Code\\VSCodeProject\\Traffic_Anomaly_Detection_based_on_Neural_Network\\binary_classification.csv'
df = pd.read_csv(CSV_FILE_PATH)

# 数据预处理
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].cat.codes
for col in df.columns:
    if df[col].dtype != 'float64':
        df[col] = df[col].astype(float)

# 特征选择
features_considered = ['Bwd_Packet_Length_Min',         # ​反向数据包长度最小值；DDoS控制命令、心跳包会频繁发送小包，若反向流量中出现大量极小包，可能是隐蔽信道通信或心跳信号的标志。【检测隐蔽信道】
                       'Subflow_Fwd_Bytes',             # ​子流前向字节数；分析一个TCP流中某个时间窗口（子流）内客户端到服务器的数据传输量。端口扫描可能在短时间内产生大量小流量子流
                       'Total_Length_of_Fwd_Packets',   # 前向数据包总长度；统计客户端到服务器的总数据量，异常高值可能指向大文件传输或数据渗透攻击。【流量规模】
                        'Fwd_Packet_Length_Mean',        # ​前向数据包长度均值；DNS请求包通常较小（~100字节），而恶意负载（如SQL注入）可能嵌入在异常大包中，加密流量（如TLS）的数据包长度分布更均匀，而明文协议（如HTTP）可能波动更大
                      'Bwd_Packet_Length_Std',          # 反向数据包长度标准差；量化服务器到客户端数据包长度的波动性，异常响应检测：正常Web响应长度相对稳定，而C&C服务器的响应可能随机化长度以规避检测。加密流量分析：加密流量（如VPN）的反向包长度标准差通常较低（因填充机制）。
                      'Flow_Duration',                  # 流持续时间
                      'Flow_IAT_Std',                   # 流到达时间间隔标准差；数据包到达时间的波动性，僵尸网络（Botnet）的C&C通信通常具有规律性时序（低标准差），而人类交互（如网页浏览）时序更随机（高标准差），隐蔽信道可能通过固定时间间隔传输数据（低标准差）【时序】
                      'Init_Win_bytes_forward',         # 前向初始窗口字节数；记录TCP三次握手阶段客户端声明的初始窗口大小，攻击者可能操纵初始窗口大小（如设为0或极大值）以绕过防火墙规则或发起资源耗尽攻击。
                      'Bwd_Packets/s',                  # 反向数据包速率（数据包/秒）【检测洪泛】
                      'PSH_Flag_Count',                 # ​PSH标志计数；PSH标志强制接收端立即处理数据，高频使用可能关联攻击（如HTTP Flood）或实时应用（如VoIP）【协议细节】
                      'Average_Packet_Size']            # 平均数据包大小；流量类型识别：小包（~64字节）：心跳包、控制命令（如ICMP Flood）。大包（~1500字节）：文件传输、视频流。加密流量检测：加密流量（如TLS）的平均包大小通常接近MTU（最大传输单元），而明文协议更分散

features = df[features_considered]
data_result = df['Target']

# 标准化
dataset = features.values
feature_mean = dataset.mean(axis=0)
feature_std = dataset.std(axis=0)
dataset = (dataset - feature_mean) / feature_std
dataset = pd.DataFrame(dataset, columns=features_considered)
dataset.insert(0, 'Target', data_result)
dataset = dataset.values

# 时间序列数据生成函数
"""
将时间序列数据转换为监督学习（Supervised Learning）格式，生成适合时间序列预测模型。从原始时间序列中滑动一个固定长度的窗口，将窗口内的历史数据作为输入特征（data），窗口后的未来数据作为标签（labels），从而生成多个样本。
"""
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size                                    # 跳过前 history_size 个时间步
    end_index = end_index if end_index else len(dataset) - target_size          # 默认设为 len(dataset) - target_size：防止生成标签时越界
    for i in range(start_index, end_index):                                     
        indices = range(i - history_size, i, step)                              # 取 [i-history_size, i) 区间内每隔 step 步的数据（indices）
        data.append(dataset[indices])
        if single_step:                                                         # 用过去7天的数据预测第8天，history_size=7, target_size=1
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)

# 用过去10000条的数据预测第10001~10100
past_history = 10000
future_target = 100
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP, single_step=True)

# 数据管道
BATCH_SIZE = 256
BUFFER_SIZE = 10000
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()       # 小批量梯度下降解决了内存限制、计算效率和训练稳定性的问题
val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# 模型定义（关键修正在于使用完整的tf.keras.layers路径，不用以前from tensorflow.keras import layers导入模块）
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]),  # 使用 tf.keras.layers
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译与训练
model.compile(optimizer='adam', # 优化算法，用于调整模型参数以最小化损失函数。'adam' 是通用选择，适合大多数任务，自适应学习率，收敛快且无需手动调整学习率
              loss='binary_crossentropy', # 损失函数，衡量模型预测值与真实标签的差异。二分类任务使用 binary_crossentropy。
              metrics=['accuracy']) # 评估指标，用于监控训练和验证过程中的模型表现（不参与参数更新）。'accuracy' 直接反映分类正确率。

log_dir = "graph/log_fit/7" # 日志保存路径
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)    # 记录训练过程中的损失、指标、权重分布等数据，供TensorBoard可视化分析。
model.fit(train_data_single, epochs=5, steps_per_epoch=100, validation_data=val_data_single, validation_steps=50, callbacks=[tensorboard_callback])