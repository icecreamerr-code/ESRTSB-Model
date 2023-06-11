# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Transformer network
'''
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tools.modules import ff, positional_encoding, multihead_attention


class Transformer(object):
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self,hidden_size, num_blocks=1,num_heads=2,dropout_rate=0.2,maxlen=512):

        self.d_model=hidden_size
        self.d_ff=4*hidden_size
        self.num_heads=num_heads
        self.maxlen=maxlen
        self.dropout_rate=dropout_rate
        self.num_blocks=num_blocks
    def encode(self, enc,src_masks=None, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.compat.v1.variable_scope("encoder", reuse=tf.compat.v1.AUTO_REUSE):
            tf.compat.v1.disable_eager_execution()
            # src_masks
            # embedding
            # (N, T1, d_model)
            # enc *= self.d_model**0.5 # scale
            #
            # enc += positional_encoding(enc, self.maxlen,E=self.d_model,T=self.maxlen)
            enc = tf.nn.dropout(enc, self.dropout_rate)
            # src_masks = tf.compat.v1.ones(enc_shape[:-1])

            # print('src_masks',src_masks.get_shape().as_list())

            ## Blocks
            for i in range(self.num_blocks):
                with tf.compat.v1.variable_scope("num_blocks_{}".format(i), reuse=tf.compat.v1.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.num_heads,
                                              d_model=self.d_model,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.d_model])
        memory = enc
        return memory



