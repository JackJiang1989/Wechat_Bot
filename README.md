# Wechat_Bot
Chatbot for Wechat 参考自 https://github.com/chiphuyen/tf-stanford-tutorials.git

## 食用方法
1.  申请微信公众号，申请服务器（推荐亚马逊AWS免费使用一年），随后公众号与服务器绑定，参考https://mp.weixin.qq.com/wiki?action=doc&id=mp1472017492_58YV5&t=0.9634440120054247#1.1
2.  操作系统就选ubuntu吧（没在windows上测试过），依次安装python2.7，virtualenv（可选，帮助设置独立的python环境）， tensorflow 0.12.1，web.py，tensorflow安装参考https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/g3doc/get_started/os_setup.md 
3.  准备训练用的语料，必须是中文，UTF-8编码，第一行为问，第二行为答，然后第二行为问，第三行为答，依次类推。语聊可从下面链接下载，但需要按上述要求处理https://github.com/rustch3n/dgk_lost_conv
4.  下载程序解压至英文路径，打开config.py，设置处理后的语料的DATA_PATH
5.  在chatbot文件夹下打开终端，执行``` ~$ python data.py ```程序开始按语料准备词汇表及上下文。
6.  随后训练``` ~$ python chatbot.py --mode 'train' ```我在本地用老电脑跑了2天。
7.  服务器打开http 80端口，开始服务``` ~$ python main.py 80 ```这里估计要用 ``` ~$ su root ```开80口。

---
## 原理
* 根据对方的句子生成我方应答的第一个字，根据对方的句子及我方应答的第一个字生成我方应答的下一个字，依次类推，直至生成一句完整的句子。

* 其中生成下一个字的模型，用的是现在最流行的LSTM或GRU（神经网络模型，每个cell中含几个gates控制数据的"遗忘"或“记忆”，解决了RNN训练时的梯度消失/爆炸问题，时间序列建模中常用的算法）。LSTM或GRU是由RNN演化而来的，本质是用神经网络拟合概率模型。概率模型可参考用于自然语言处理最经典的基于统计的概率模型n-gram。
---
## Tricks

* 词向量，每个词用向量的形式表示，也作为trainable参数一起训练，程序中用tensorflow的embedding函数实现。也可参考word2vec的paper。

* attention model，应答中的下一个字出现的概率都与对方句子中的词有”不同程度的关系”，用可训练的attention参数描述”不同程度的关系“。参考下面paper

* bucket，训练时用的随机批梯度下降保证训练效率，这样每一批（batch）要保证用同样的长度的问答句方可进行训练，所以创造了bucket，少的就padding几个0填充。tensorflow源码seq2seq中model_with_buckets函数有具体实现方法。

* sampled softmax， 网络最后一层用softmax输出全词汇表概率时太消耗内存，用抽样的方法输出部分概率进行训练。参考下面paper
---
## Papers

* Neural Probabilistic Language Model http://machinelearning.wustl.edu/mlpapers/paper_files/BengioDVJ03.pdf

* Word2vec http://arxiv.org/pdf/1301.3781.pdf

* RNN / LSTM https://arxiv.org/pdf/1308.0850.pdf

* BasicLSTMCell in Tensorlfow https://arxiv.org/pdf/1409.2329.pdf

* Attention model https://arxiv.org/pdf/1409.0473v7.pdf

* Sampled softmax https://arxiv.org/pdf/1412.2007v2.pdf

---
## 未来工作
1. 老人，小孩，男人，女人，不同的chatbot，这需要带标签的样本训练
2. 个性化，用一个人的样本训练或者把代表一个人个性的对话样本反复多次喂给程序。
3. 记忆力
4. 生成回答时不用贪心算法，尝试寻找全局概率最大
5. 可在对话中训练机器人
