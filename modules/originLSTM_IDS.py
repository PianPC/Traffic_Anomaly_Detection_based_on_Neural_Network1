#!/usr/bin/env python
# coding=UTF-8  
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 禁用TensorFlow的冗余日志（只显示错误和警告）
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.python import layers
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 30000

CSV_FILE_PATH = '/Users/klaus_imac/Desktop/毕设/数据集/IDS2017/Test/dataset.csv'
df = pd.read_csv(CSV_FILE_PATH)

#修改数据类型
#Object类型转换为离散数值（Label列）
df['Label'] = pd.Categorical(df['Label']) # 将Label列转换为分类数据类型
df['Label'] = df['Label'].cat.codes       # 将分类标签转换为数值编码
columns_counts = df.shape[1]              # 获取列数，shape[1]是columns列，shape[0]row行
for i in range(columns_counts):           # 将所有非float64类型的列强制转换为float类型
  if(df.iloc[:,i].dtypes) != 'float64':
    df.iloc[:, i] = df.iloc[:,i].astype(float)

#选取11个特征和Label
features_considered = ['Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
features = df[features_considered]        # 从DataFrame中提取选定特征
data_result = df['Target']

#标准化
dataset = features.values                                        # 将DataFrame转换回numpy数组以便后续时间窗口处理
feature_mean = dataset.mean(axis=0)
feature_std = dataset.std(axis=0)
dataset = (dataset-feature_mean)/feature_std                     # 对特征数据进行标准化（Z-score标准化）
dataset = pd.DataFrame(dataset,columns=features_considered)      # 将标准化后的numpy数组重新转换为DataFrame，保留原始特征名称
dataset.insert(0,'Target',data_result)                           # 在首列位置插入目标变量
dataset = dataset.values                                         # 频繁切换原因：numpy数组用于数学运算，df用于数据管理？

#返回时间窗,根据给定步长对过去的观察进行采样  history_size为过去信息窗口的大小，target_size为模型需要预测的未来时间
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []                                     # 创建历史数据窗口
  labels = []                                   # 生成对应的未来目标值

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size      # 如果未指定end_index,则设置最后一个训练点

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)    # 从i-history_size到i，每隔step取一个
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])      # 仅仅预测未来的单个点
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


past_history = 10000
future_target = 100
STEP = 6

# 生成训练集，使用前30000个样本作为训练数据
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)            #dataset[:,1]取最后一列的所有值
# 验证集
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

#训练集、验证集
BATCH_SIZE = 256      # 每个训练批次包含256个样本
BUFFER_SIZE = 10000   # 用于数据打乱的缓冲区容量

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))        # 将NumPy数组转换为TensorFlow即(features, label)格式的集对象，
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()   # 数据优化流水线
"""
注意顺序问题！（若无特殊说明，则训练集与验证集均必需）
train_data_single = train_data_single.cache()              # 缓存数据到内存；避免重复预处理（如标准化）
                       .shuffle(BUFFER_SIZE)               # 打乱数据顺序；通过缓冲区随机打乱数据顺序（buffer_size越大随机性越好）；【验证集可选】
                       .batch(BATCH_SIZE)                  # 分批次；用于小批量梯度下降；验证集仅用于评估模型性能，无需随机性。【验证集不需要】
                       .repeat()                           # 无限循环数据
"""

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# 创建模型
model = tf.keras.Sequential([
    layers.LSTM(32,input_shape=x_train_single.shape[-2:]),    # LSTM单元数量（隐层维度）32，input_shape=(时间步数, 特征数)
    layers.Dense(32),                                         # 隐层全连接
    layers.Dense(1, activation='sigmoid')                     # 输出层，输出层使用sigmoid：适合二分类问题（输出0-1概率值）
])

# 当前是二分类任务，sparse_categorical...不适用，若未来可能扩展为多分类任务
# loss = 'sparse_categorical_crossentropy'
# optimizer = tf.keras.optimizers.SGD(0.1)

# 当前配置
model.compile(optimizer='Adam',               # 默认学习率0.001
              loss = 'binary_crossentropy',   # 二分类标准损失
              metrics=['accuracy'])           # 监控准确率

log_dir = "graph/log_fit/7"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train_single, y_train_single, epochs=5, batch_size=256,callbacks=[tensorboard_callback])
"""
model.fit(
    x_train_single, y_train_single,  # 训练数据
    epochs=5,                        # 训练5轮
    batch_size=256,                  # 与Dataset的batch需一致
    callbacks=[tensorboard_callback] # 启用监控
)
"""


