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
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import datetime

start_time = datetime.datetime.now()

# 修复路径并加载数据
CSV_FILE_PATH = 'E:\\workplace\\Code\\VSCodeProject\\Traffic_Anomaly_Detection_based_on_Neural_Network\\binary_classification.csv'
df = pd.read_csv(CSV_FILE_PATH)

# 标签编码
df['Label'] = pd.Categorical(df['Label']).codes

# 强制转换为 float32（避免 TensorFlow 类型警告）
for col in df.columns:
    if df[col].dtype != 'float32':
        df[col] = df[col].astype('float32')

# 选取特征和标签
features_considered = [
    'Target', 'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
    'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
    'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
    'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
    'Average_Packet_Size'
]
feature_last = [col for col in features_considered if col != 'Target']
feature = df[features_considered]

# 划分数据集
train, test = train_test_split(feature, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# 数据标准化（使用 Keras Normalization 层替代手动标准化）
train_features = train[feature_last]
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(train_features.values)

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Target')
    # 直接传递数值数组，无需特征列
    ds = tf.data.Dataset.from_tensor_slices((dataframe[feature_last].values, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    return ds.batch(batch_size)

# 创建数据管道
batch_size = 50
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# 构建模型（使用新 API）
model = tf.keras.Sequential([
    normalizer,  # 标准化层
    layers.Dense(20, activation='selu'),
    layers.Dense(20, activation='selu'),
    layers.Dense(20, activation='selu'),
    layers.Dense(20, activation='selu'),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    verbose=1
)

# 评估
loss, accuracy = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 保存模型（Keras 格式）
save_path = "traffic_model.keras"
model.save(save_path)
print(f"Model saved to: {os.path.abspath(save_path)}")

# 加载验证
loaded_model = tf.keras.models.load_model(save_path)
loaded_loss, loaded_acc = loaded_model.evaluate(test_ds)
print(f"Loaded Model Accuracy: {loaded_acc:.4f}")

# 计时
end_time = datetime.datetime.now()
print(f"Total Time: {(end_time - start_time).seconds} seconds")

