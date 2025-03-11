  ### 待实现
  1. 使用scapy、PyShark混合架构方案
  * 实时监测模块 realtime_monitor.py
      自己电脑开启实时监测理应全部是“正常”数据？但是有不少“可疑”数据
  2. flask框架搭建
  * app.py：​Flask 框架的入口文件，主要承担Web服务启动、​路由控制、​请求处理、​模板渲染、​API接口
  3. 协议识别引擎
  4. 增加pcap回放功能，支持离线数据集测试
  5. 可视化与交互
  
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


  ### deepseek总结要做什么
  根据你的毕设项目研究内容和预期目标，系统需要实现以下核心功能模块及详细功能点：

​一、核心功能模块
​1. 数据采集与解析
​实时流量抓取：支持从网络接口实时捕获加密流量（TCP/UDP/QUIC等协议）
​协议识别：自动识别TLS/SSL、HTTPS等常见加密协议，解析协议头部元数据
​数据包重组：实现会话级流量重组（如TCP流重组），支持PCAP文件导入/导出
​2. 数据预处理
​流量清洗：过滤无效/重复数据包，处理缺失值与异常值
​特征工程：
统计特征提取（包长分布、时间间隔方差等）
时序特征建模（滑动窗口统计、会话持续时间等）
加密协议特征提取（TLS版本、密码套件、证书指纹等）
​数据标准化：对特征进行归一化（Min-Max/Z-Score）和维度压缩
​3. 异常检测引擎
​多模型集成：
监督学习（Random Forest/XGBoost + 特征选择）
深度学习（1D-CNN处理包序列，LSTM建模时序依赖）
混合模型（图神经网络处理流量拓扑关系）
​实时推理：支持微批处理（100ms级延迟）与流式计算
​模型热更新：允许动态加载新模型版本无需停机
​4. 可视化与交互
​实时监控仪表盘：
流量热力图（源-目的IP矩阵）
协议分布环形图
异常事件时间线
​深度分析模式：
会话详情反查（原始载荷十六进制视图）
特征重要性雷达图
模型决策可视化（Grad-CAM突出可疑流量段）
​告警管理：分级告警（高危/中危/低危）与工单系统集成
​二、关键技术指标
​性能要求：

检测准确率 ≥85%（F1-score）
单节点处理能力 ≥1Gbps（千兆网络线速）
端到端延迟 ≤500ms（从抓包到告警）
​兼容性：

支持TLS 1.3/DoT/DoH等新型加密协议
适配AWS/GCP/Azure云原生环境
提供REST API供第三方系统调用
​可扩展性：

模块化架构（插件式特征提取器/检测模型）
分布式部署（Kafka+Flink流处理集群）
​三、非功能性需求
​安全审计：操作日志记录（符合GDPR合规要求）
​资源监控：CPU/内存/网络使用率实时统计
​文档体系：
技术白皮书（系统架构设计说明）
API文档（Swagger UI集成）
用户手册（含典型部署场景示例）
​四、开发技术栈建议
模块	推荐技术方案
​流量采集	libpcap（C）/ Scapy（Python） + DPDK加速
​特征计算	Apache Spark Structured Streaming + Pandas UDF
​模型服务	PyTorch Lightning + ONNX Runtime + Triton Inference Server
​可视化	Elastic Stack（Kibana） + Grafana + 自定义React前端
​部署运维	Docker Compose（开发环境）/ Kubernetes（生产环境） + Prometheus监控体系
​五、功能验证方案
​测试数据集：

基准测试：CIC-IDS2017/USTC-TFC2016
自制数据集：通过Metasploit生成模拟攻击流量
​对比实验：

Baseline方法：CICFlowMeter特征+随机森林
消融实验：验证混合模型相对单一模型的提升效果
​压力测试：

使用tcpreplay重放10Gbps背景流量
测量误报率（FPR）随负载变化曲线


### deepseek分析完成进度
一、研究内容完成度分析
​1. 需求分析（50%完成）​
✅ ​已实现部分

基本流量特性提取：支持TCP/UDP协议解析
流表管理机制：实现五元组流记录（含超时清理）
实时性支持：通过scapy实时抓包与批量处理
❌ ​待完善部分

加密协议深度解析：缺少TLS/SSL证书链、SNI提取
用户角色适配：未实现不同用户权限/需求配置
部署场景适配：未区分云环境/IoT等特殊处理
​2. 算法设计（70%完成）​
✅ ​已实现部分

基础特征工程：11个统计特征提取（Flow_Duration等）
双模型架构：支持DNN/LSTM动态切换
流式处理：窗口采样（LSTM）、批量推理优化
❌ ​待完善部分

特征标准化：缺少RobustScaler等预处理
模型融合机制：未实现多模型投票/Stacking
增量学习：不支持在线模型更新
​3. 系统实现（80%完成）​
✅ ​已实现部分

核心模块完整：
mermaid
graph LR
A[抓包] --> B[流表管理]
B --> C[特征提取]
C --> D[模型预测]
D --> E[结果输出]
线程安全设计：流表锁、线程池管理
资源监控：内存占用统计
❌ ​待完善部分

协议识别引擎：未实现TLS指纹检测
分布式支持：单机架构，无Kafka集成
特征自动提取：缺乏CNN自动特征学习
​4. 性能评估（40%完成）​
✅ ​已实现部分

基础指标计算：准确率、召回率日志输出
实时延迟监控：预测耗时记录
❌ ​待完善部分

公开数据集测试：未集成CIC-IDS2017验证
对抗测试：缺乏对抗样本鲁棒性评估
资源效率报告：无GPU利用率统计