#!/usr/bin/env python
# coding=UTF-8
"""
基于LSTM的流量异常检测系统（时序数据版本）
数据集：binary_classification.csv（需放在同一目录）
"""

# %% [1] 环境配置
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 配置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化信息

# 全局配置
CSV_RELATIVE_PATH = './binary_classification.csv'  # 相对路径
TRAIN_SPLIT = 30000          # 训练集分割点
PAST_HISTORY = 1000          # 历史窗口（调整为1000以测试）
FUTURE_TARGET = 10           # 预测窗口
STEP = 6                     # 采样间隔
BATCH_SIZE = 256             

# %% [2] 数据加载管道
def load_data(file_path):
    """加载并预处理数据"""
    df = pd.read_csv(file_path)
    
    # 标签编码（修复版）
    df['Label'] = df['Label'].astype('category').cat.codes
    
    # 类型统一转换
    float_cols = df.select_dtypes(exclude=['float']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    return df

# %% [3] 特征工程
def select_features(df):
    """特征选择与数据准备"""
    features = [
        'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size', 'Target'  # 包含目标列
    ]
    return df[features]

# %% [4] 数据标准化
def normalize_data(train_df, val_df):
    """基于训练集的标准化"""
    train_mean = train_df.mean(axis=0)
    train_std = train_df.std(axis=0)
    
    norm_train = (train_df - train_mean) / train_std
    norm_val = (val_df - train_mean) / train_std
    
    return norm_train, norm_val

# %% [5] 时序数据生成器
def create_time_series(dataset, target, start_idx, end_idx, hist_size, pred_size, step):
    """
    生成时序数据样本
    参数：
    hist_size : 历史时间步长
    pred_size : 预测步长
    step      : 滑动窗口步长
    """
    data = []
    labels = []
    
    # 自动计算结束索引
    end_idx = end_idx or (len(dataset) - pred_size)
    start_idx += hist_size  # 跳过初始无历史数据段
    
    for i in range(start_idx, end_idx):
        indices = range(i-hist_size, i, step)
        data.append(dataset[indices])
        labels.append(target[i+pred_size])  # 单步预测
        
    return np.array(data), np.array(labels)

# %% [6] 主流程
if __name__ == "__main__":
    # 数据加载
    raw_df = load_data(CSV_RELATIVE_PATH)
    feature_df = select_features(raw_df)
    
    # 时序数据划分（保留时间顺序）
    train_df = feature_df.iloc[:TRAIN_SPLIT]
    val_df = feature_df.iloc[TRAIN_SPLIT:]
    
    # 标准化处理
    norm_train, norm_val = normalize_data(
        train_df.drop('Target', axis=1), 
        val_df.drop('Target', axis=1)
    )
    
    # 合并标签
    train_data = np.hstack([train_df[['Target']].values, norm_train.values])
    val_data = np.hstack([val_df[['Target']].values, norm_val.values])
    
    # 生成时序数据集（修复索引越界）
    x_train, y_train = create_time_series(
        train_data, train_data[:, 0], 
        start_idx=0, end_idx=None,
        hist_size=PAST_HISTORY, 
        pred_size=FUTURE_TARGET,
        step=STEP
    )
    
    x_val, y_val = create_time_series(
        val_data, val_data[:, 0],
        start_idx=PAST_HISTORY,  # 确保足够历史数据
        end_idx=None,
        hist_size=PAST_HISTORY,
        pred_size=FUTURE_TARGET,
        step=STEP
    )
    
    # 数据管道
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_loader = train_loader.shuffle(10000).batch(BATCH_SIZE).prefetch(1)
    
    val_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_loader = val_loader.batch(BATCH_SIZE)
    
    # LSTM模型
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=x_train.shape[-2:]),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # 训练
    history = model.fit(
        train_loader,
        epochs=10,
        validation_data=val_loader,
        verbose=1
    )
    
    # 保存模型
    model.save('./lstm_traffic_model.keras')
    print("模型训练完成并已保存")