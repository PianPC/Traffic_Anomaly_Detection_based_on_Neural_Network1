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
from tensorflow import feature_column
from tensorflow.python import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime

start_time = datetime.datetime.now()

CSV_FILE_PATH = '~/four_classification.csv'
df = pd.read_csv(CSV_FILE_PATH)

#Object类型转换为离散数值（Label列）
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].cat.codes

#将int类型转换为float类型
columns_counts = df.shape[1]                                                #获取列数
for i in range(columns_counts):
  if(df.iloc[:,i].dtypes) != 'float64':
    df.iloc[:, i] = df.iloc[:,i].astype(float)


"""
#将特征随时间变化用图像展示出来
ts = df['Init_Win_bytes_forward']
ts.plot(title='PortScan:Init_Win_bytes_forward')  # 调用 Pandas Series 的 .plot() 方法，生成折线图。
plt.xlabel('Time-Step')
plt.ylim(0,80000)
plt.xlim(0,1000)
plt.show()
"""


#选取11个特征和Label
features_considered = ['Target','Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']    # 包含标签和特征的所有相关列
feature_last = ['Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']    # 与 features_considered 相比，去除了 'Target'
feature = df[features_considered]
print(len(feature))

"""
与LSTM的区别开始出现
​DNN这里先划分训练集、测试集，再分别标准化；避免数据泄露；无时序依赖，适用于独立样本。
LSTM先对整个数据集标准化，再划分训练集、测试集；导致测试集信息泄露到训练阶段，评估结果可能高估；需处理时间序列的滑动窗口，标准化方式错误加剧时序数据泄露风险。
"""


#将数据集分为训练集、验证集、测试集
train, test = train_test_split(feature,test_size=0.2)

#标准化
def normalize_dataset(dataset, dataset_mean, dataset_std, insert_target):
    dataset = (dataset-dataset_mean)/dataset_std
    final_dataset = pd.DataFrame(dataset, columns=feature_last)
    final_dataset.insert(0, 'Target', insert_target)
    return final_dataset

train.reset_index(drop=True,inplace=True)                                   #重置索引，很关键！
train_target = train['Target']
train_other = train[feature_last]
train_dataset = train_other.values
train_mean = train_dataset.mean(axis=0)
train_std = train_dataset.std(axis=0)
train = normalize_dataset(train_dataset, train_mean, train_std, train_target)

#对测试集进行标准化时使用训练集的均值和标准差
test.reset_index(drop=True,inplace=True)
test_target = test['Target']
test_other = test[feature_last]
test_dataset = test_other.values
test = normalize_dataset(test_dataset, train_mean, train_std, test_target)

train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

#使用tf.data.Dataset读取数据，将 Pandas DataFrame 转换为 TensorFlow 的 tf.data.Dataset 对象，用于高效数据加载。
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  """
  dataframe: 输入的 Pandas DataFrame，包含特征和标签。
  shuffle: 是否打乱数据顺序（默认 True），防止模型因数据顺序产生偏差。
  batch_size: 每个批次的样本数（默认 32），影响内存使用和训练效率。
  """
  dataframe = dataframe.copy()                                          # 创建输入 DataFrame 的副本，避免直接修改原始数据
  labels = dataframe.pop('Target')                                      # 移除 'Target' 列，并将其值赋给 labels
  #如果赋值给变量，返回的是Series
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))    # Dataset 的每个元素为一个元组 (特征字典, 标签)，({'Feature1': 0.5, 'Feature2': 10}, 1)  # 单个样本
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))                         # 打乱数据顺序，避免模型学习到数据排列的潜在模式，buffer_size=len(dataframe): 设置缓冲区大小为数据集的总样本数，确保完全随机化
  ds = ds.batch(batch_size)
  return ds

#选择要使用的列
feature_use = []
for header in feature_last:
  feature_use.append(feature_column.numeric_column(header))                   # 将特征名列表 feature_last 转换为 TensorFlow 的数值特征列（NumericColumn）

feature_layer = tf.keras.layers.DenseFeatures(feature_use,dtype='float64')    # 将特征列列表封装为 Keras 层，用于将原始输入数据转换为模型可处理的张量格式

batch_size = 50                                                               # 50-256，较小的批次可能提高泛化能力，较大的批次加速训练。
train_ds = df_to_dataset(train, batch_size=batch_size)                        # 将训练集 DataFrame 转换为 tf.data.Dataset
# print(list(train_ds.as_numpy_iterator())[0])
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#创建，编译和训练模型
model = tf.keras.Sequential([
  feature_layer,                          # 特征处理层（DenseFeatures）
  layers.Dense(20, activation='selu'),    # 全连接层1，20个神经元，SELU激活
  layers.Dense(20, activation='selu'),    # 全连接层2
  layers.Dense(20, activation='selu'),
  layers.Dense(20, activation='selu'),
  layers.Dense(4, activation='softmax')   # 输出层，4个类别（softmax归一化）
])

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',   # 损失函数：稀疏多分类交叉熵（标签为整数）
              metrics=['accuracy'],
              run_eagerly=True)                         # 启用即时执行模式（调试用，通常设为False以提升性能），禁用计算图优化，便于调试（如打印中间张量值），但降低训练速度。

#其他指标、使用Tensorboard
# tf.keras.metrics.Recall(),
# tf.keras.metrics.FalsePositives(),
# tf.keras.metrics.TrueNegatives()
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_dir = "graph/log_fit/13"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_ds,                   # 从训练数据集（tf.data.Dataset格式）加载数据
          validation_data=val_ds,     # 验证数据集
          epochs=20                   # 训练轮次
          )
# callbacks=[tensorboard_callback]

loss, accuracy = model.evaluate(test_ds)  # 在测试集上评估
print("loss:",loss)
print("Accuracy:", accuracy)
# print("recall:",recall)
# print("FPR:",FP/(FP+TN))

end_time = datetime.datetime.now()
print("spend_time:",(end_time-start_time).seconds)

#保存模型
'''
model.save('Final_Model')
begin_time = datetime.datetime.now()
reconstructed_model = tf.keras.models.load_model('Final_Model')

reconstructed_model.evaluate(test_ds)

final_moment = datetime.datetime.now()
print('保存的模型预测时间：', (final_moment-begin_time).seconds)
'''

# === 加载模型 ===
# loaded_model = tf.keras.models.load_model('my_model')