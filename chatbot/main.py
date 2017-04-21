# -*- coding: utf-8 -*-
# filename: main.py

import hashlib
import reply
import receive
import web
#from handle import Handle
import os
#import sys

import tensorflow as tf

from model import ChatBotModel
from chatbot import _check_restore_parameters
from predict import get_predicted_sentence
import config
import data



urls = (
    '/wx', 'Handle',
)



class Handle(object):
    def POST(self):
        try:

            webData = web.data()
            print "Handle Post webdata is ", webData   #后台打日志
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg) and recMsg.MsgType == 'text':
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                content = get_predicted_sentence(recMsg.Content, enc_vocab, inv_dec_vocab, model, sess)
                content = content.encode('utf-8')
#                content = recMsg.Content
                replyMsg = reply.TextMsg(toUser, fromUser, content)
                return replyMsg.send()
            else:
                print "暂且不处理"
                return "success"
        except Exception, Argment:
            return Argment

    def GET(self):
        return get_predicted_sentence(u'你好', enc_vocab, inv_dec_vocab, model, sess)
#       return '好'


model = ChatBotModel(True, batch_size=1)
model.build_graph()
_, enc_vocab = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.enc'))
inv_dec_vocab, _ = data.load_vocab(os.path.join(config.PROCESSED_PATH, 'vocab.dec'))
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
_check_restore_parameters(sess, saver)

if __name__ == '__main__':
#    sess = tf.Session()
#    model = create_model(sess, forward_only=True)
#    model.batch_size = 1
#    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
#    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    app = web.application(urls, globals())
    app.run()
                                                                                                 66,1          Bot

