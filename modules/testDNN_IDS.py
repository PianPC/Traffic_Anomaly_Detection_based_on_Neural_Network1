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
CSV_FILE_PATH = os.path.join(recent_PATH,  '..', 'category', 'binary_classification.csv')

def load_and_preprocess_data():
    """数据加载与预处理流程"""
    df = pd.read_csv(CSV_FILE_PATH)
    
    # 获取唯一标签类别数量和编码
    label_categories = pd.Categorical(df['Label'])
    global n_classes  # 声明为全局变量以便模型构建使用
    n_classes = len(label_categories.categories)
    print(n_classes)
    df['Label'] = label_categories.codes  # 生成数字编码标签  
    # 特征工程配置
    features = [
        'Bwd_Packet_Length_Min', 'Subflow_Fwd_Bytes',
        'Total_Length_of_Fwd_Packets', 'Fwd_Packet_Length_Mean',
        'Bwd_Packet_Length_Std', 'Flow_Duration', 'Flow_IAT_Std',
        'Init_Win_bytes_forward', 'Bwd_Packets/s', 'PSH_Flag_Count',
        'Average_Packet_Size'
    ]

    # 仅转换特征列为float32
    df[features] = df[features].astype('float32')

    return df[features], df['Label']

# 执行数据准备
start_time = datetime.datetime.now()
data_feature, data_label = load_and_preprocess_data()

# %% [3] 数据集划分（关键修改）
# 正确分离特征和标签
X_train, X_test, y_train, y_test = train_test_split(
    data_feature, 
    data_label,
    test_size=0.2,
    stratify=data_label,  # 保持类别分布
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# %% [4] 数据预处理管道
def create_normalization_layer(training_features):
    """创建数据标准化层"""
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(training_features.values)
    return normalizer

# 初始化标准化层（使用训练数据）
normalizer = create_normalization_layer(X_train)

def df_to_dataset(features, labels, shuffle=True, batch_size=32):
    """创建TensorFlow数据集管道"""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 创建数据管道
BATCH_SIZE = 64
train_ds = df_to_dataset(X_train, y_train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(X_val, y_val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(X_test, y_test, shuffle=False, batch_size=BATCH_SIZE)

# %% [5] 模型架构
def build_dnn_model():
    """构建深度神经网络模型"""
    return tf.keras.Sequential([        # 以顺序堆叠（逐层线性叠加）​的方式快速搭建神经网络模型
        normalizer,  # 输入标准化层
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        layers.Dense(20, activation='selu'),
        # 注意：输出层设置为4个单元，适用于四分类任务
        # 如果是二分类任务，建议改为1个单元+sigmoid激活
        layers.Dense(n_classes, activation='softmax')
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
    
    # 添加早停回调，防止过拟合并优化训练效率
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',     # 监控验证集损失
        patience=5,             # 容忍连续5个epoch无改善
        restore_best_weights=True  # 恢复最佳权重
    )

    model.fit(
    train_ds,                # 训练数据集
    validation_data=val_ds,  # 验证数据集
    epochs=50,               # 最大训练轮次
    callbacks=[early_stop],  # 回调函数（此处为早停）
    verbose=1                # 日志输出模式
    )

# 执行训练（保持不变）
history = compile_and_train(model)

# %% [7] 模型评估与保存（更新模型保存逻辑）
# 保存最佳模型（而非最后一个epoch的模型）
best_model = model  # 因为使用了restore_best_weights=True
test_loss, test_accuracy = best_model.evaluate(test_ds)

# 模型保存路径优化
MODEL_SAVE_DIR = os.path.join(recent_PATH, '..', 'models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model_save_path = os.path.join(MODEL_SAVE_DIR, "traffic_model.keras")

best_model.save(model_save_path)
print(f"\n模型已保存至: {os.path.abspath(model_save_path)}")

# 模型加载验证
loaded_model = tf.keras.models.load_model(model_save_path)
loaded_loss, loaded_acc = loaded_model.evaluate(test_ds)
print(f"加载模型验证: 准确率={loaded_acc:.4f}")

# %% [8] 耗时统计
end_time = datetime.datetime.now()
print(f"\n总耗时: {(end_time - start_time).seconds} 秒")