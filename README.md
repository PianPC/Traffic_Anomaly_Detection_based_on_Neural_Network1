  ### 待实现
  1. 使用scapy、PyShark混合架构方案
    * 实时监测模块 realtime_monitor.py
      自己电脑开启实时监测理应全部是“正常”数据？但是有不少“可疑”数据
  2. flask框架搭建
    * app.py：​Flask 框架的入口文件，主要承担Web服务启动、​路由控制、​请求处理、​模板渲染、​API接口
  
  ### 备忘/待解决
  1. ~~originDNN_IDS.py、originLSTM_IDS.py是原项目 https://github.com/KlausMichael0/Taffic_Anomaly_Detection_based_on_Neural_Network 的源代码，其中originDNN_IDS.py中20行的CSV_FILE_PATH = '~/four_classification.csv'以及originLSTM_IDS.py中18行的CSV_FILE_PATH = '/Users/klaus_imac/Desktop/毕设/数据集/IDS2017/Test/dataset.csv'不知道具体指代哪些数据集，因此为了先跑通代码，testDNN_IDS.py与testLSTM_IDS.py这两个文件的数据集统一使用了根目录下的binary_classification.csv，它们是修改后的可成功运行的代码，怕后续把代码改到面目全非找不回初版，因此新建两个新py文件。~~
  LSTM需要的是时间序列的数据集，DNN需要的是非时间数据集，但实际上，同一份数据可以通过不同的处理方式适配两种模型。
  * LSTM：网络流量中的时序性攻击检测​（如DDoS攻击的流量在时间上会突然暴增、端口扫描会在一段时间内高频出现）。需要将数据组织成时间窗口。
  * DNN：处理独立样本的分类或回归任务，每个样本的特征是独立的，不依赖前后顺序。基于单条流量特征的分类（如根据单个数据包的特征判断是否为SQL注入）。每个样本是一个独立的特征向量。
  2. ~~originDNN_IDS.py中54行理解代码过程中注释中提到的LSTM的处理顺序有泄露数据的风险~~testLSTM已处理泄露数据风险
  3. ~~构建新的适合LSTM与DNN适用的数据集？~~ 原项目文件binary_classification.csv比原项目文件夹category里的csv最后多一个target标签，其余无异
  4. CICIDS2017 http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/ 下载的两个数据集TrafficLabelling、MachineLearningCVE仅有几个标签不一样，拥有更多标签的是TrafficLabelling文件夹中的，后续准备可供LSTM与DNN共同使用的数据集时，先找一个带时间戳的，然后加上target即可。（应该这样就够了？？）
 ![alt text](image.png)
  5. 完善优化了一下DNN与LSTM文件的代码结构以及一些细节问题。
  6. 突然发现原项目作用没完全符合开题报告里的要求，还有一些未满足的部分，比如图神经网络应用、多模型融合、系统集成等。
  7. 仍未构建新数据集。
  ~~8. 文件夹结构框架~~