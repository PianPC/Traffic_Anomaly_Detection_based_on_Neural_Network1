#!/usr/bin/env python
# coding=UTF-8
"""
基于深度神经网络的流量异常检测（分类任务）
使用Keras API实现，包含数据预处理、模型训练与评估全流程
"""

# %% [1] 环境配置
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# 配置TensorFlow日志级别和Matplotlib后端
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')

# %% [2] 数据准备
# 数据集路径配置
recent_PATH = os.path.dirname(__file__)
CSV_RELATIVE_PATH = os.path.join(recent_PATH,  '..', 'category', 'binary_classification.csv')

def load_and_preprocess_data():
    """数据加载与预处理流程"""
    # 加载原始数据
    df = pd.read_csv(CSV_FILE_PATH)
    
    # 标签编码（注意：当前使用Label列作为分类目标）
    df['Label'] = pd.Categorical(df['Label']).codes
    
    # 类型转换（解决TensorFlow类型警告）
    for col in df.columns:
        if df[col].dtype != 'float32':
            df[col] = df[col].astype('float32')
    
    # 特征工程配置
    features_considered = [
        'Target', 'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size'
    ]
    feature_last = [col for col in features_considered if col != 'Target']
    
    return df[features_considered], feature_last

# 执行数据准备
start_time = datetime.datetime.now()
feature_data, feature_columns = load_and_preprocess_data()

# %% [3] 数据集划分
# 注意：当前使用Target列作为标签，请根据实际情况确认
train, test = train_test_split(feature_data, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# %% [4] 数据预处理管道
def create_normalization_layer(training_features):
    """创建数据标准化层"""
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(training_features.values)
    return normalizer

# 初始化标准化层
train_features = train[feature_columns]
normalizer = create_normalization_layer(train_features)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    """将DataFrame转换为TensorFlow Dataset对象"""
    dataframe = dataframe.copy()
    labels = dataframe.pop('Target')  # 提取标签列
    
    # 注意：此处使用Target作为标签，请确认与Label列的关系
    ds = tf.data.Dataset.from_tensor_slices(
        (dataframe[feature_columns].values, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    return ds.batch(batch_size)

# 创建数据管道
BATCH_SIZE = 50
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test, shuffle=False, batch_size=BATCH_SIZE)

# %% [5] 模型架构
def build_dnn_model():
    """构建深度神经网络模型"""
    return tf.keras.Sequential([
        normalizer,  # 输入标准化层
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        # 注意：输出层设置为4个单元，适用于四分类任务
        # 如果是二分类任务，建议改为1个单元+sigmoid激活
        layers.Dense(4, activation='softmax')
    ])

model = build_dnn_model()

# %% [6] 模型训练
def compile_and_train(model):
    """模型编译与训练流程"""
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=1
    )

# 执行训练
history = compile_and_train(model)

# %% [7] 模型评估与保存
# 测试集评估
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"\n评估结果: 测试集损失={test_loss:.4f}, 准确率={test_accuracy:.4f}")

# 模型保存
SAVE_PATH = "traffic_model.keras"
model.save(os.path.join(recent_PATH,  '..', 'models', SAVE_PATH))
print(f"\n模型已保存至: {os.path.abspath(SAVE_PATH)}")

# 模型加载验证
loaded_model = tf.keras.models.load_model(SAVE_PATH)
loaded_loss, loaded_acc = loaded_model.evaluate(test_ds)
print(f"加载模型验证: 准确率={loaded_acc:.4f}")

# %% [8] 耗时统计
end_time = datetime.datetime.now()
print(f"\n总耗时: {(end_time - start_time).seconds} 秒")