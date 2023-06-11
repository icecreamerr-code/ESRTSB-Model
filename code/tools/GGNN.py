import tensorflow as tf
import math


class GGNN():
    def __init__(self,hidden_size=100, out_size=100, batch_size=300, n_node=None,
                 lr=None, l2=None, step=1, decay=None, lr_dc=0.1, nonhybrid=False):
        super(GGNN,self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.nonhybrid = nonhybrid

        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.compat.v1.get_variable('nasr_w1', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_w2 = tf.compat.v1.get_variable('nasr_w2', [self.out_size, self.out_size], dtype=tf.float32,
                                       initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_v = tf.compat.v1.get_variable('nasrv', [1, self.out_size], dtype=tf.float32,
                                      initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.nasr_b = tf.compat.v1.get_variable('nasr_b', [self.out_size], dtype=tf.float32, initializer=tf.compat.v1.zeros_initializer())

        self.n_node = n_node
        self.L2 = l2
        self.step = step
        self.nonhybrid = nonhybrid
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size

        self.cell=cell = tf.compat.v1.nn.rnn_cell.GRUCell(self.out_size)
        self.W_in = tf.compat.v1.get_variable('W_in', shape=[self.out_size, self.out_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_in = tf.compat.v1.get_variable('b_in', [self.out_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_out = tf.compat.v1.get_variable('W_out', [self.out_size, self.out_size], dtype=tf.float32,
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_out = tf.compat.v1.get_variable('b_out', [self.out_size], dtype=tf.float32,
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))

        self.W_ih=tf.compat.v1.get_variable('W_ih', shape=[self.gate_size, self.input_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_ih = tf.compat.v1.get_variable('b_ih', [self.gate_size], dtype=tf.float32,
                                     initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.W_hh = tf.compat.v1.get_variable('W_hh', shape=[self.gate_size, self.hidden_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))
        self.b_hh = tf.compat.v1.get_variable('b_hh', [self.gate_size], dtype=tf.float32,
                                    initializer=tf.compat.v1.random_uniform_initializer(-self.stdv, self.stdv))

        # with tf.variable_scope('ggnn_model', reuse=None):
        #     self.loss_train, _ = self.forward(self.ggnn())
        # with tf.variable_scope('ggnn_model', reuse=True):
        #     self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)
        #



    def ggnn(self,fin_state,adj):
        adj_in,adj_out=tf.split(adj,2,2)
        S=fin_state.get_shape().as_list()
        batch_size=S[0]
        with tf.compat.v1.variable_scope('gru'):
            for i in range(self.step):
                fin_state = tf.reshape(fin_state, [batch_size, -1, self.out_size])
                fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                    self.W_in) + self.b_in, [batch_size, -1, self.out_size])
                print(fin_state_in.get_shape().as_list())
                fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                     self.W_out) + self.b_out, [batch_size, -1, self.out_size])
                av = tf.concat([tf.matmul(adj_in, fin_state_in),
                                tf.matmul(adj_out, fin_state_out)], axis=-1)
                print('dafsdgg',av.get_shape())
                gi = tf.reshape(tf.matmul(av, self.W_ih,transpose_b=True) + self.b_ih, [batch_size, -1, self.gate_size])
                gh = tf.reshape(tf.matmul(fin_state,self.W_hh,transpose_b=True) + self.b_hh, [batch_size, -1, self.gate_size])
                # print(gi.get_shape())
                i_r, i_i, i_n = tf.split(gi,3, 2)
                h_r, h_i, h_n = tf.split(gh,3, 2)
                resetgate=tf.sigmoid(i_r+h_r)
                inputgate=tf.sigmoid(i_i+h_i)
                newgate = tf.tanh(i_n + resetgate * h_n)
                fin_state = newgate + inputgate * (fin_state - newgate)
                # state_output, fin_state = \
                #     tf.nn.dynamic_rnn(self.cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1),
                #                       initial_state=tf.reshape(fin_state, [-1, self.out_size]))
        print('fin_state',fin_state.get_shape())
        return tf.reshape(fin_state, [batch_size, -1, self.out_size])
