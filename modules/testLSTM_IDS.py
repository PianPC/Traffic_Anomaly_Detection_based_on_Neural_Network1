#!/usr/bin/env python
# coding=UTF-8
"""
基于LSTM的流量异常检测系统（集成标准化层版本）
"""

# %% [1] 环境配置
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# 配置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 全局配置
recent_PATH = os.path.dirname(__file__)
CSV_RELATIVE_PATH = os.path.join(recent_PATH,  '..', 'category', 'binary_classification.csv')
TRAIN_SPLIT = 30000          # 训练集分割点
PAST_HISTORY = 1000          # 历史窗口
FUTURE_TARGET = 10           # 预测窗口
STEP = 6                     # 采样间隔
BATCH_SIZE = 256             

# %% [2] 数据加载管道（保持原始数据）
def load_data(file_path):
    """加载数据并保留原始数值"""
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].astype('category').cat.codes  # 标签编码
    return df

# %% [3] 特征工程
def select_features(df):
    """返回特征和标签的DataFrame"""
    features = [
        'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size'
    ]
    labels = ['Target']
    return df[features], df[labels]

# %% [4] 时序数据生成器（处理原始数据）
def create_time_series(features, labels, start_idx, end_idx, hist_size, pred_size, step):
    """生成未标准化的时序数据"""
    data = []
    labels_out = []
    
    end_idx = end_idx or (len(features) - pred_size)
    start_idx = max(start_idx, hist_size)  # 防止越界
    
    for i in range(start_idx, end_idx):
        indices = range(i-hist_size, i, step)
        data.append(features.iloc[indices].values)
        labels_out.append(labels.iloc[i+pred_size].values)
        
    return np.array(data), np.array(labels_out)

# %% [5] 主流程
if __name__ == "__main__":
    # 加载原始数据
    raw_df = load_data(CSV_RELATIVE_PATH)
    features, labels = select_features(raw_df)

    # 分割数据集（保持时序）
    train_features, train_labels = features.iloc[:TRAIN_SPLIT], labels.iloc[:TRAIN_SPLIT]
    val_features, val_labels = features.iloc[TRAIN_SPLIT:], labels.iloc[TRAIN_SPLIT:]

    # 生成时序数据（使用原始值）
    x_train, y_train = create_time_series(
        train_features, train_labels,
        start_idx=0, end_idx=None,
        hist_size=PAST_HISTORY,
        pred_size=FUTURE_TARGET,
        step=STEP
    )
    
    x_val, y_val = create_time_series(
        val_features, val_labels,
        start_idx=PAST_HISTORY,
        end_idx=None,
        hist_size=PAST_HISTORY,
        pred_size=FUTURE_TARGET,
        step=STEP
    )

    # 构建含标准化层的模型
    # ================= 关键修改部分 =================
    # 创建标准化层并手动设置参数
    normalization_layer = tf.keras.layers.Normalization(axis=-1)
    normalization_layer.adapt(train_features.values)  # 仅用训练集计算参数
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        normalization_layer,  # 集成标准化层
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # ==============================================
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    # 数据管道
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(1)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=val_dataset,
        verbose=1
    )

    # 保存完整模型（包含标准化层参数）
    model.save(os.path.join(recent_PATH,  '..', 'models', 'lstm_traffic_model.keras'))
    print("模型已保存，包含内置标准化层")