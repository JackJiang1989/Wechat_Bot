# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import data
from model import ChatBotModel
from chatbot import _check_restore_parameters, _find_right_bucket , run_step, _construct_response

def get_predicted_sentence(input_sentence, enc_vocab, inv_dec_vocab, model, sess):
    """ in test mode, we don't to create the backward path
    """
    line = input_sentence
    token_ids = data.sentence2id(enc_vocab, line)
    bucket_id = _find_right_bucket(len(token_ids))
        # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, decoder_masks = data.get_batch([(token_ids, [])], bucket_id, batch_size=1)
    decoder_inputs[0][0]=2
        # Get output logits for the sentence.
    _, _, output_logits = run_step(sess, model, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
    response = _construct_response(output_logits, inv_dec_vocab)
    return response

