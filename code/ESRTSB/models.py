from __future__ import absolute_import
from layers import *
from tensorflow.python.ops.rnn_cell import GRUCell
import sys
import tensorflow as tf
sys.path.append('..')

from tensorflow.python.keras.models import Model,Layer
from tools.transformer import Transformer
from train import FLAGS


class MyModel(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        self.layers = []
        self.activations = []
        self.inputs = None

    def build(self):
        raise NotImplementedError
    def forward(self,**kwargs):
        raise NotImplementedError

class GCN(MyModel):
    def __init__(self, inputs, adj, f_dropout, adj_dropout, num_support_nonzero,user_num,item_num,eb_dim,**kwargs):
        super(GCN, self).__init__(**kwargs)
        self.inputs = inputs
        self.input_dim = FLAGS.embedding_dim
        self.adj = adj
        self.f_dropout = f_dropout
        self.adj_dropout = adj_dropout
        self.num_support_nonzero = num_support_nonzero
        self.user_num = user_num
        self.item_num = item_num
        self.eb_dim = eb_dim
        self.trans = Transformer(eb_dim)
        tf.compat.v1.disable_eager_execution()
        self.user_maxpooling_weight = tf.Variable(tf.random.normal([eb_dim, eb_dim], mean=0, stddev=0.1))
        self.item_maxpooling_weight = tf.Variable(tf.random.normal([eb_dim, eb_dim], mean=0, stddev=0.1))

    def build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            support=self.adj,
                                            f_dropout=self.f_dropout,
                                            adj_dropout=self.adj_dropout,
                                            num_support_nonzero=self.num_support_nonzero,
                                            user_num=self.user_num,
                                            item_num=self.item_num
                                            ))

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.input_dim,
                                            support=self.adj,
                                            f_dropout=self.f_dropout,
                                            adj_dropout=self.adj_dropout,
                                            num_support_nonzero=self.num_support_nonzero,
                                            user_num=self.user_num,
                                            item_num=self.item_num
                                            ))


        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        for i in range(len(self.activations)):
            self.activations[i]=tf.expand_dims(self.activations[i],1)
        outputs = tf.concat(self.activations,axis=1)
        user_output,item_output = tf.split(outputs,[self.user_num,self.item_num],0)

        ########mutil_head_attention
        user_outputs=self.trans.encode(user_output)
        item_outputs=self.trans.encode(item_output)
        user_outputs=tf.reduce_max(user_outputs,axis=1)
        item_outputs=tf.reduce_max(item_outputs,axis=1)
        user_outputs=tf.matmul(user_outputs, self.user_maxpooling_weight)
        item_outputs=tf.matmul(item_outputs, self.item_maxpooling_weight)

        return tf.concat([user_outputs,item_outputs],axis=0)


class W_contrastive(Layer):
    def __init__(self,d):
        super().__init__()

        self.W = tf.Variable(tf.compat.v1.initializers.glorot_uniform([d,d]))

    def _call(self,x):
        return tf.matmul(x,self.W)
def dot(x, y, sparse=False):
    '''

    :param x: [E,2]
    :param y: [N,F]
    :param sparse:
    :return:
    '''
    """Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * tf.expand_dims(sp.values(),axis=1)
    result = tf.zeros((sp.shape[0],emb.shape[1])).cuda(tf.device(device))
    result.index_add_(0, rows, col_segs)
    return result



