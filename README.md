# Wechat_Bot
Chatbot for Wechat 参考自 https://github.com/chiphuyen/tf-stanford-tutorials.git

## 食用方法
1.  操作系统是ubuntu（没在windows上测试过），依次安装python2.7，virtualenv（可选，帮助设置独立的python环境）， tensorflow 0.12.1，web.py 
2.  准备训练文件，必须是中文，UTF-8编码，第一行为问，第二行为答，第二行为问，第三行为答，依次类推。
3.  下载程序解压至英文路径，打开config.py，设置训练文件的DATA_PATH
4.  在chatbot文件夹下打开终端，执行``` ~$ python data.py ```开始准备词汇表及上下文。
5.  随后训练``` ~$ python chatbot.py --mode 'train' ```
6.  开始对话执行``` ~$ python chatbot.py --mode 'chat' ``` 这个功能有待更新。
7.  微信服务器设置
---
## 原理
* 根据对方的句子生成我方应答的第一个字（或词），根据对方的句子及我方应答的第一个字（或词）生成我方应答的下一个字（或词），依次类推，直至生成一句完整的句子。

* 其中生成字（或词）的模型
* 可以是N-Gram（基于统计的概率模型，计算量异常巨大）
* 可以是RNN（基本神经网络模型，会出现梯度爆炸）
* 可以是LSTM或GRU（神经网络模型，每个cell中含几个gates控制数据的"遗忘"或“记忆”，解决了训练时的梯度爆炸问题，时间序列建模中常用的算法）。
* 具体原理参考下面的paper。
---
## Tricks

* 词向量，每个词用向量的形式表示，也作为trainable参数一起训练（具体实现及原因参考word2vec和embedding的paper）。

* attention model，应答中的每个字（或词）出现的概率都与对方句子中的词有”不同程度的关系”，用attention trainable参数描述”不同程度的关系“进行训练。

* bucket，训练时用的随机批梯度下降保证训练效率，这样每一批（batch）要保证用同样的长度的问答句方可进行训练，所以创造了bucket，少的就padding几个0填充。

* sampled softmax， 网络最后一层用softmax输出概率时太消耗内存，用抽样的方法训练。
---
## Papers

* Neural Probabilistic Language Model http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf

* Word2vec http://arxiv.org/pdf/1301.3781.pdf

* Recurrent Neural Network https://arxiv.org/pdf/1308.0850.pdf

* LSTM http://arxiv.org/abs/1409.2329

* Attention model http://arxiv.org/abs/1412.7449

* Sampled softmax https://arxiv.org/pdf/1412.2007v2.pdf
