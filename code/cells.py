import tensorflow as tf
#from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
# https://github.com/tensorflow/tensorflow/blob/5dab22191d70d2dcd247d2d7b11628981c0a6f12/tensorflow/python/keras/layers/recurrent.py#L1068
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

class GraphCell(tf.keras.layers.Layer):
    '''
    Graph invariance RNN cell
    '''

    def __init__(self, units,
                 links,
                 v,
                 w,
                 message_units=4,
                 dropout=0.,
                 recurrent_dropout=0.,
                 masked=False,
                 coupled=True, **kwargs):
        super(GraphCell, self).__init__( **kwargs)
        # [l,e]
        self.state_size = tf.TensorShape([links, units])
        self.output_size = tf.TensorShape([links, units])
        #self.state_size = tf.TensorShape([None, units])
        self.units = units
        self.links = links
        self._batch_size = None
        self.v = v
        self.w = w
        self.masked=masked
        self.coupled=coupled
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        with tf.name_scope("update"):
            self.gru = tf.keras.layers.GRUCell(self.units, name='update')
        self.message_units = message_units
        with tf.name_scope("message"):
            self.message = tf.keras.layers.Dense(self.message_units,name='message',activation=tf.nn.selu) #if coupled else None

    # See https://keras.io/guides/serialization_and_saving/
    def get_config(self):
        config = super(GraphCell, self).get_config()
        config.update({"units": self.units,
                "links": self.links,
                "v": self.v.numpy(),
                "w": self.w.numpy(),
                "coupled": self.coupled,
                "masked": self.masked,
                "dropout": self.dropout,
                "recurrent_dropout": self.recurrent_dropout,
                "message_units": self.message_units})
        return config

    def build(self, input_shapes):
        # [batch,link,feature]
        if input_shapes[1]:
            assert input_shapes[1] == self.links
        self.feature_dim = input_shapes[2]

        if self.coupled or True:
            with tf.name_scope("update"):
                self.gru.build(tf.TensorShape([None, self.feature_dim + self.message_units]))
            if self.masked: # Set the input size for the masked model
                with tf.name_scope("message"):
                    self.message.build(tf.TensorShape([None, self.feature_dim + self.units]))
            else:
                with tf.name_scope("message"):
                    self.message.build(tf.TensorShape([None, self.units]))
        else:
            self.gru.build(tf.TensorShape([None, self.feature_dim]))


    @tf.function
    def call(self, inputs, states, training=None):
        # inputs should be in [batch,links,1]
        # states should be in [batch,links,self.feature_dim]

        _batch_size = tf.shape(inputs)[0]
        new_states = tf.reshape(states[0], [_batch_size * self.links, self.units])
      
        if self.coupled or True: 
            # If coupled is false, I remove all spacial information. It works like simple RNN.
            # It's to see if there is additional information of space.
            # In our case, use coupled==True.
            concat_input = states[0]
            if self.masked:
                # If we work with masked links dataset, we concat inputs
                concat_input = tf.concat([states[0], inputs], 2)
            gathered = tf.gather(self.message(concat_input), self.v, axis=1)
            m = tf.transpose(gathered, [1, 0, 2])
            m = tf.math.unsorted_segment_sum(m, self.w, self.links)
            m = tf.transpose(m, [1, 0, 2])
            if not self.coupled:
                # Cut spacial information by multiplying by zeros. We don't use it
                m = m * tf.zeros_like(m) 
            new_inputs = tf.concat([inputs, m], axis=2)
            new_inputs = tf.reshape(new_inputs, [_batch_size * self.links, self.message_units+self.feature_dim])
        else:
            new_inputs = tf.reshape(inputs, [_batch_size * self.links, self.feature_dim])

        h, _ = self.gru.call(new_inputs, [new_states])
        h = tf.reshape(h,[_batch_size,self.links,self.units])

        return h, [h]
