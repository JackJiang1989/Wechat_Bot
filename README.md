# Wechat_Bot
Chatbot for Wechat 参考自stanford的一门deep learning课程练习 https://github.com/chiphuyen/tf-stanford-tutorials.git
微信公众号：Dogfish2017

## 食用方法
1.  申请微信公众号，申请服务器（推荐亚马逊AWS免费使用一年），随后公众号与服务器绑定，参考https://mp.weixin.qq.com/wiki?action=doc&id=mp1472017492_58YV5&t=0.9634440120054247#1.1
2.  操作系统就选ubuntu吧（没在windows上测试过），依次安装python2.7，virtualenv（可选，帮助设置独立的python环境）， tensorflow 0.12.1，web.py，tensorflow安装参考https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/g3doc/get_started/os_setup.md 
3.  准备训练用的语料，必须是中文，UTF-8编码，第一行为问，第二行为答，然后第二行为问，第三行为答，依次类推。语聊可从下面链接下载，但需要按上述要求处理https://github.com/rustch3n/dgk_lost_conv
4.  下载程序解压至英文路径，打开config.py，设置处理后的语料的DATA_PATH
5.  在chatbot文件夹下打开终端，执行``` ~$ python data.py ```程序开始按语料准备词汇表及上下文。
6.  随后训练``` ~$ python chatbot.py --mode 'train' ```我在本地用老电脑跑了2天。
7.  服务器打开http 80端口，开始服务``` ~$ nohup python main.py 80 &```这里估计要用 ``` ~$ su root ```权限开80端口。

---
## 原理
* 根据对方问句生成我方应答的第一个字，根据对方问句及我方应答的第一个字生成我方应答的下一个字，依次类推，直至生成一句完整的句子。

* 其中生成下一个字的算法，用的是现在最流行的LSTM/GRU（神经网络模型，其中每个cell里含几个gates分别控制数据的"遗忘"或“记忆”，解决了RNN训练时的梯度消失/爆炸问题，时间序列建模中常用的算法）。LSTM/GRU是由RNN演化而来的，本质是用神经网络拟合概率模型，可参考基于统计的概率模型n-gram。
---
## Tricks

* 词向量，每个词用向量的形式表示，也作为trainable参数一起训练，程序中用tensorflow的embedding函数实现。也可参考word2vec的paper。

* attention model，应答中的下一个字出现的概率与问句中的每个字都有”不同程度的关系”，用可训练的attention参数描述”不同程度的关系“。参考下面paper

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

* Seq2Seq https://arxiv.org/pdf/1703.03906.pdf

## Interesting web page:

* Standford deep learning basic: http://deeplearning.stanford.edu/tutorial/

* Standford NLP: http://web.stanford.edu/class/cs224n/

* Google Seq2seq code: https://github.com/google/seq2seq

* LSTM: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

* Blogs: https://karpathy.github.io/

* Blogs: https://colah.github.io/

---
## 开发方向（大坑1号）
1. 个性化 - 基于特定样本训练bot，使bot拥有这个特定样本的个性。
2. 记忆力 - 这个比较难，可以尝试通过增加encoding长度及关键字提取实现。
3. 贪心改半全局 - 用beam search使相同问句输出不同对话内容。
4. 对话中训练 - 可在对话中训练bot，这个比较有意思，下一步准备尝试实现。
5. 图像识别 - CNN卷积神经网络跑跑alexnet...

---
## 语料收集（大坑2号）
* 借鉴 http://www.image-net.org/ 上线一个中文对话语料网站，任何人都可以自由使用里面的数据。
