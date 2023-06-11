from inits import *
import tensorflow as tf
xavier_initializer=tf.compat.v1.initializers.glorot_normal

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}
paramId = 0
biasDefault = False
params = {}
regParams = {}
ita = 0.2
leaky = 0.1
def getParam(name):
	return params[name]
def getParamId():
	global paramId
	paramId += 1
	return paramId
def defineRandomNameParam(shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	name = 'defaultParamName%d'%getParamId()
	return defineParam(name, shape, dtype, reg, initializer, trainable)

def defineParam(name, shape, dtype=tf.float32, reg=False, initializer='xavier', trainable=True):
	global params
	global regParams
	assert name not in params, 'name %s already exists' % name
	if initializer == 'xavier':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, shape=shape,
			initializer=xavier_initializer(dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'trunc_normal':
		ret = tf.compat.v1.get_variable(name=name, initializer=tf.random.truncated_normal(shape=[int(shape[0]), shape[1]], mean=0.0, stddev=0.03, dtype=dtype))
	elif initializer == 'zeros':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=tf.zeros(shape=shape, dtype=tf.float32),
			trainable=trainable)
	elif initializer == 'ones':
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype, initializer=tf.ones(shape=shape, dtype=tf.float32), trainable=trainable)
	elif not isinstance(initializer, str):
		ret = tf.compat.v1.get_variable(name=name, dtype=dtype,
			initializer=initializer, trainable=trainable)
	else:
		print('ERROR: Unrecognized initializer')
		exit()
	params[name] = ret
	if reg:
		regParams[name] = ret
	return ret
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


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
        res = tf.compat.v1.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, support,f_dropout=0, adj_dropout=0, num_support_nonzero=None,user_num=None,item_num=None,**kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.f_dropout = f_dropout
        self.adj_dropout = adj_dropout
        self.num_support_nonzero = num_support_nonzero
        self.support = support
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.user_num=user_num
        self.item_num=item_num

    def _call(self, inputs):

        x = inputs
        # dropout
        x = tf.nn.dropout(x, self.f_dropout)
        pre_sup = dot(self.support,x,sparse=True)
    
        return pre_sup


class AttentionAggregator(Layer):

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.1, bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''

        tf.compat.v1.disable_eager_execution()
        self.att_agg_weight=tf.Variable(tf.random.normal([neigh_input_dim, output_dim], mean=0, stddev=0.1))

        self.input_dim = input_dim
        self.output_dim = output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, self.dropout)

        self_vecs=tf.transpose(self_vecs,perm=[1,0,2])
        query = self_vecs
        neigh_self_vecs = tf.concat([neigh_vecs, query], axis=1)
        score = tf.matmul(query, neigh_self_vecs, transpose_b=True)
        score = tf.nn.softmax(score, axis=-1)

        # alignment(ESRTSB) shape is [batch_size, 1, depth]
        context = tf.matmul(score, neigh_self_vecs)

        context = tf.squeeze(context, axis=1)

        output = tf.matmul(context, self.att_agg_weight)


        return self.act(output)











