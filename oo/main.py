# -*- coding: utf-8 -*-
# filename: main.py

import hashlib
import reply
import receive
import web
#from handle import Handle
import os
import sys

import tensorflow as tf

from tf_seq2seq_chatbot.configs.config import FLAGS
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib.seq2seq_model_utils import create_model, get_predicted_sentence

urls = (
    '/wx', 'Handle',
)



class Handle(object):
    def POST(self):
        try:

            webData = web.data()
            print "Handle Post webdata is ", webData   #�~P~N�~O��~I~S�~W���~W
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg) and recMsg.MsgType == 'text':
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                content = get_predicted_sentence(recMsg.Content, vocab, rev_vocab, model, sess)
                content = content.encode('utf-8')
#               content = recMsg.Content
                replyMsg = reply.TextMsg(toUser, fromUser, content)
                return replyMsg.send()
            else:
                print "�~Z~B��~T��~M��~D�~P~F"
                return "success"
        except Exception, Argment:
            return Argment

    def GET(self):
        return get_predicted_sentence('��| ', vocab, rev_vocab, model, sess).encode('utf-8')
#       return '好'

sess = tf.Session()
model = create_model(sess, forward_only=True)
model.batch_size = 1
vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)



if __name__ == '__main__':
#    sess = tf.Session()
#    model = create_model(sess, forward_only=True)
#    model.batch_size = 1
#    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
#    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    app = web.application(urls, globals())
    app.run()
                                                                                                 66,1          Bot

