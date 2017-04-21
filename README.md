# Wechat_Bot
Chatbot for Wechat 参考于

## 原理
* 根据对方的句子生成我方应答的第一个字（或词），根据对方的句子及我方应答的第一个字（或词）生成我方应答的下一个字（或词），依次类推，直至生成一句完整的句子。

* 其中生成字（或词）的模型可以是N-Gram（基于统计的概率模型，计算量异常巨大），可以是RNN（基本神经网络模型，会出现梯度爆炸），可以是LSTM或GRU（神经网络模型，每个cell中含几个gates控制数据的"遗忘"或“记忆”，解决了训练时的梯度爆炸问题，时间序列建模中常用的算法）。具体原理参考下面的paper。
---
## 用到的Tricks

* 词向量，每个词用向量的形式表示，也作为trainable参数一起训练（具体实现及原因参考word2vec和embedding的paper）。

* attention model，应答中的每个字（或词）出现的概率都与对方句子中的词有”不同程度的关系”，用attention trainable参数描述”不同程度的关系“进行训练。

* bucket，训练时用的随机批梯度下降保证训练效率，这样每一批（batch）要保证用同样的长度的问答句方可进行训练，所以创造了bucket，少的就padding几个0填充。

* sampled softmax， 网络最后一层用softmax输出概率时太消耗内存，用抽样的方法训练。

N-Gram

word2vec

Recurrent Neural Network.

Long short term memory / GRU

tricks: embedding ; attention ; bucket ; sampled softmax
