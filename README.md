  ### 待解决
  1. ~~originDNN_IDS.py、originLSTM_IDS.py是原项目 https://github.com/KlausMichael0/Taffic_Anomaly_Detection_based_on_Neural_Network 的源代码，其中originDNN_IDS.py中20行的CSV_FILE_PATH = '~/four_classification.csv'以及originLSTM_IDS.py中18行的CSV_FILE_PATH = '/Users/klaus_imac/Desktop/毕设/数据集/IDS2017/Test/dataset.csv'不知道具体指代哪些数据集，因此为了先跑通代码，testDNN_IDS.py与testLSTM_IDS.py这两个文件的数据集统一使用了根目录下的binary_classification.csv，它们是修改后的可成功运行的代码，怕后续把代码改到面目全非找不回初版，因此新建两个新py文件。~~
  LSTM需要的是时间序列的数据集，DNN需要的是非时间数据集，但实际上，同一份数据可以通过不同的处理方式适配两种模型。
  * LSTM：网络流量中的时序性攻击检测​（如DDoS攻击的流量在时间上会突然暴增、端口扫描会在一段时间内高频出现）。需要将数据组织成时间窗口。
  * DNN：处理独立样本的分类或回归任务，每个样本的特征是独立的，不依赖前后顺序。基于单条流量特征的分类（如根据单个数据包的特征判断是否为SQL注入）。每个样本是一个独立的特征向量。
  2. ~~originDNN_IDS.py中54行理解代码过程中注释中提到的LSTM的处理顺序有泄露数据的风险~~testLSTM已处理泄露数据风险
  3. 构建新的适合LSTM与DNN适用的数据集？
  