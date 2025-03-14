#!/usr/bin/env python
# coding=UTF-8
"""
基于LSTM的流量异常检测系统（集成标准化层版本）
"""

# %% [1] 环境配置
import os
import json
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
def load_and_preprocess_data(CSV_FILE_PATH):
    """数据加载与预处理流程"""
    df = pd.read_csv(CSV_FILE_PATH)
    
    # 获取唯一标签类别数量和编码
    label_categories = pd.Categorical(df['Label'])      # 自动提取所有唯一标签类别，Categories (4, object): ['BENIGN', 'DDoS', 'DoS Hulk', 'PortScan']
    global n_classes, class_names
    class_names = label_categories.categories.tolist()  # 获取类别名称列表
    n_classes = len(label_categories.categories)
    df['Label'] = label_categories.codes                # 生成数字编码标签  
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


# %% [4] 时序数据生成器（处理原始数据）
def create_time_series(features, labels, start_idx, end_idx, hist_size, pred_size, step):
    """
    生成未标准化的时序数据，将时间序列数据转换为 ​监督学习格式，生成历史特征窗口（data）和对应的未来标签（labels_out）
    假设原始数据为 [t0, t1, t2, t3, t4]，hist_size=2, pred_size=1：
    样本1：特征窗口 [t0, t1] → 标签 t2
    样本2：特征窗口 [t1, t2] → 标签 t3
    hist_size：窗口越大，模型可捕捉更长期依赖，但计算成本更高。
​    step：步长越大，窗口数据越稀疏（节省计算资源，但可能丢失细节）。
    """
    data = []
    labels_out = []
    
    end_idx = end_idx or (len(features) - pred_size)
    start_idx = max(start_idx, hist_size)  # start_idx 和 end_idx 确保不越界，避免用未来数据预测过去
    
    for i in range(start_idx, end_idx):
        indices = range(i-hist_size, i, step)
        data.append(features.iloc[indices].values)
        labels_out.append(labels.iloc[i+pred_size].values)
        
    return np.array(data), np.array(labels_out)

# %% [5] 主流程
if __name__ == "__main__":
    # 加载原始数据
    features, labels = load_and_preprocess_data(CSV_RELATIVE_PATH)
    labels = labels.to_frame(name='Label')  # 将 Series 转回 DataFrame
    '''
    （保持原代码兼容性）此处是为了能够实现代码复用，所用数据加载管道函数与DNN里的一样，该函数返回的features是DataFrame，但labels是Series，格式不一样
    或许后续可以改成返回两个DF？但是这样的话DNN又要改好多

    '''
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
        tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2])),      # Input(shape=(时间步长, 特征数))，LSTM 输入要求：三维输入 (样本数, 时间步长, 特征数)
        normalization_layer,  # 集成标准化层
        tf.keras.layers.LSTM(64, return_sequences=False),                       # return_sequences=False：仅返回最后一个时间步的输出。
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')  # 动态设置输出单元数
    ])
    # ==============================================
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # 适用整数标签
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc', multi_label='ovr')]       # multi_label=True 适用于 ​多标签分类​（一个样本可同时属于多个类别）,所以要改成ovr
    )
    
    # 添加早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',       # 监控验证集 AUC
        patience=3,             # 允许 3 轮不提升
        mode='max',             # AUC 越高越好
        restore_best_weights=True  # 恢复最佳模型权重
    )

    # 数据管道
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(1)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # 训练模型
    history = model.fit(
        train_dataset,
        epochs=50,              # 增大总轮次，由早停控制实际训练轮数
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    # 保存完整模型（包含标准化层参数）
    model_save_path = os.path.join(recent_PATH, '..', 'models', 'lstm_traffic_model.keras')
    model.save(model_save_path)
    
    print(f"模型已保存至 {model_save_path}")

    # 保存标签映射（新增关键部分）
    label_mapping = {str(i): name for i, name in enumerate(class_names)}
    label_map_path = os.path.join(os.path.dirname(model_save_path), 'label_mapping.json')
    with open(label_map_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"模型已保存至: {model_save_path}")
    print(f"标签映射已保存至: {label_map_path}")

    # 模型加载验证
    loaded_model = tf.keras.models.load_model(model_save_path)
    print("\n加载模型验证:")
    loaded_model.evaluate(val_dataset, verbose=1)