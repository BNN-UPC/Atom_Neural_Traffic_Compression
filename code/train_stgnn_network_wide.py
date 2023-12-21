import datetime
import inspect
import os
from dataclasses import dataclass
from functools import partial
import json
from keras.layers import Multiply, Layer
import sys
sys.path.insert(1, '../data')
from utils import *
import cells
import numpy as np
import time as tt
import networkx as nx
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import pickle
import tensorflow_probability as tfp
from absl import flags, app, logging

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(0)
tf.keras.backend.set_floatx('float32')
np.random.seed(3)

tfd = tfp.distributions
tfb = tfp.bijectors

FLAGS = flags.FLAGS

# This dataset name should be replaced
dataset_name = "geant_w5_r40_maxt6063"
dataset_name = "abilene_w5_r40_maxt41741"
dataset_name = "KB_agh_w5_r20_maxt102799"

units = 50
seq_layer_size = 10
batch_size = 3

log_dataset_name = dataset_name+"_u"+str(units)+"_s"+str(seq_layer_size)+"_b"+str(batch_size)

flags.DEFINE_string("input_dir", "../data/files_to_be_compressed/tfrecords/", "Input file")
flags.DEFINE_string("input_file", dataset_name, "Input file")
flags.DEFINE_integer("window_size", 1, "...")
flags.DEFINE_integer("perc_training", 70, "...") 
flags.DEFINE_integer("batch_size", batch_size, "...") 
flags.DEFINE_integer("future", 1, "...")
flags.DEFINE_integer("buffer_size", 1000, "...")
flags.DEFINE_integer("repeat_train", 1, "...")
flags.DEFINE_integer("num_epoch", 2000, "...")
flags.DEFINE_integer("eval_freq", 100, "...")
flags.DEFINE_integer("units", units, "...") 
flags.DEFINE_integer("seq_layer_size", seq_layer_size, "...")
flags.DEFINE_bool('coupled', True, "...")
flags.DEFINE_string('logs_dir', "log/"+log_dataset_name+"_masked_GNN", "Model directory for logs an checkpoints")
flags.DEFINE_float('learning_rate', 0.0001, "Learning rate")
flags.DEFINE_bool('multistep', False, "...")
flags.DEFINE_enum('distribution','n',['n','t'],'Conditional distribution')
flags.DEFINE_enum('posterior','d',['n','d'],'Posterior distribution')
flags.DEFINE_enum("experiment","SpaceTimeGNN",['Experiment',
                                             'SpaceTimeGNN'],'...')

@dataclass
class Experiment:
    input_file: str
    input_dir: str
    logs_dir: str
    window_size: str
    perc_training: float
    num_train_samples = 100
    num_eval_samples = 100
    batch_size: int
    future: int
    repeat_train: int
    buffer_size: int
    num_epoch: int
    # model hparams
    units: int
    seq_layer_size: int
    coupled: bool
    multistep: bool
    inds = list()
    memory_train = list()
    model = None
    optimizer = None
    norm_layer = None
    scale_output = None
       
    def dataset(self):
        params = dict()
        with open(self.input_dir+self.input_file+"/"+self.input_file+"_params", 'r') as f:
            params = json.load(f)
        
        self.window_size = params["window_size"]
        num_samples = params["num_masked_samples"]
        self.buffer_size = num_samples
        self.scale_output = tf.cast(params["max_y"], tf.float32)
        self.num_train_samples = int((num_samples*self.perc_training)/100)
        self.num_eval_samples = num_samples-self.num_train_samples

        # Input sample has shape=(32, 24, 18, 1) for (batch, timestep, link, feature)
        # Output sample has shape=(32, 18) for (batch, link)
        ds = tf.data.TFRecordDataset(self.input_dir+self.input_file+"/"+self.input_file+'_masked.tfrecords')

        # Create a dictionary describing the features.
        data = {
            'link_uti': tf.io.VarLenFeature(tf.float32),
            'mask_pred_0': tf.io.VarLenFeature(tf.float32),
            'mask_pred_1': tf.io.VarLenFeature(tf.float32),
            'feat_pred_link': tf.io.VarLenFeature(tf.float32),
            'next_uti': tf.io.VarLenFeature(tf.float32)
        }

        def _parse_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            parsed_data = tf.io.parse_single_example(example_proto, data)
            x = dict()

            x["input_1"] = tf.sparse.to_dense(parsed_data["link_uti"])
            mask_pred_0 = tf.sparse.to_dense(parsed_data["mask_pred_0"])
            mask_pred_1 = tf.sparse.to_dense(parsed_data["mask_pred_1"])
            x["input_4"] = tf.sparse.to_dense(parsed_data["feat_pred_link"])
            next_uti = tf.sparse.to_dense(parsed_data["next_uti"])

            window_no_label = self.window_size-1 # -1 to exclude the label

            x["input_1"] = tf.reshape(x["input_1"], (window_no_label,  params["num_links"], 1))
            x["input_4"] = tf.reshape(x["input_4"], (params["num_links"], 1))
            x["input_2"] = tf.reshape(mask_pred_0, (params["num_links"], 1))
            x["input_3"] = tf.reshape(mask_pred_1, (params["num_links"], 1))
            x["input_5"] = tf.convert_to_tensor(mask_pred_1>0, dtype=bool)
            x["input_5"] = tf.reshape(x["input_5"], (params["num_links"], 1))

            # Repeat the y_label and mask as many times as window_no_label
            x["input_2"] = tf.repeat([x["input_2"]], repeats=[window_no_label], axis=0)
            x["input_3"] = tf.repeat([x["input_3"]], repeats=[window_no_label], axis=0)
            x["input_4"] = tf.repeat([x["input_4"]], repeats=[window_no_label], axis=0)
            return x, next_uti

        ds = ds.map(_parse_function)
        return ds
    
    def dataset_compress(self):
        # Read original dataset without masked values
        params = dict()
        with open(self.input_dir+self.input_file+"/"+self.input_file+"_params", 'r') as f:
            params = json.load(f)
        
        self.window_size = params["window_size"]

        # Input sample has shape=(32, 24, 18, 1) for (batch, timestep, link, feature)
        # Output sample has shape=(32, 18) for (batch, link)
        ds = tf.data.TFRecordDataset(self.input_dir+self.input_file+"/"+self.input_file+'.tfrecords')

        # Create a dictionary describing the features.
        data = {
            'link_uti': tf.io.VarLenFeature(tf.float32),
            'next_uti': tf.io.VarLenFeature(tf.float32),
        }

        def _parse_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            parsed_data = tf.io.parse_single_example(example_proto, data)
            x, y = tf.sparse.to_dense(parsed_data["link_uti"]), tf.sparse.to_dense(parsed_data["next_uti"])
            window_no_label = self.window_size-1 # -1 to exclude the label
            return tf.reshape(x, (window_no_label,  params["num_links"], 1)), y

        ds = ds.map(_parse_function)
        return ds

    def train_ds(self):
        ds = self.dataset()
        ds = ds.take(self.num_train_samples)
        ds = ds.shuffle(self.num_train_samples)
        ds = ds.cache()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.repeat(self.repeat_train)
        ds = ds.prefetch(1)
        return ds
    
    def compress_ds(self):
        ds = self.dataset_compress()
        ds = ds.cache()
        ds = ds.batch(1, drop_remainder=True)
        return ds

    def test_ds(self):
        ds = self.dataset()
        ds = ds.skip(self.num_train_samples)
        ds = ds.take(self.num_eval_samples)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.cache()
        return ds

    def final_test_ds(self):
        ds = self.dataset()
        ds = ds.skip(self.num_train_samples+self.num_eval_samples)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.cache()
        return ds


    def normalization_layer(self):
        '''
        Creats normalization layer computing mean and variance betwen future and average past
        :return:
        '''
        norm = tf.keras.layers.Normalization(axis=-1,trainable=False)
        ds = self.dataset()
        ds = ds.take(self.num_train_samples).map(
            lambda x: x["input_1"]
        )

        norm.adapt(ds)
        return norm

    def rnn(self, seq):
        with open(FLAGS["input_dir"].value+FLAGS["input_file"].value+"/links_to_remove.pkl", 'rb') as fp:
            list_zero_link_uti = pickle.load(fp)

        # Abilene dataset
        if "abilene" in self.input_file:
            topology_class = NetworkTopology()
            topology = nx.read_gml(FLAGS["input_dir"].value+FLAGS["input_file"].value+"/Abilene_topology.gml")
            topology_class.generate_topology_from_Graph(topology)

            coupled = self.coupled
            multistep = self.multistep
            units = self.units

            cell = cells.GraphCell(units, links=topology_class.numEdges, v=topology_class.first_tensor, w=topology_class.second_tensor,
                message_units=units,
                dropout=0.0,
                recurrent_dropout=0.0,
                masked=True,
                coupled=coupled)
        elif "geant" in self.input_file:
            topology_class = NetworkTopology()
            topology = nx.read_gml(FLAGS["input_dir"].value+FLAGS["input_file"].value+"/Geant_topology.gml")
            topology_class.generate_topology_from_Graph(topology)

            coupled = self.coupled
            multistep = self.multistep
            units = self.units

            cell = cells.GraphCell(units, links=topology_class.numEdges, v=topology_class.first_tensor, w=topology_class.second_tensor,
                message_units=units,
                dropout=0.0,
                recurrent_dropout=0.0,
                masked=True,
                coupled=coupled)

        elif "agh" in self.input_file:
            # The following lines of code must be used when training with campus network dataset
            A = AGH.adjacency()
            # This variable is in case we already removed some links
            num_removed = 0
            # Delete the links at 0 from correlation matrix and from the TS data
            for link_rmv in list_zero_link_uti:
                # Delete Row
                A = np.delete(A, int(link_rmv)-num_removed, 0)
                # Delete Column
                A = np.delete(A, int(link_rmv)-num_removed, 1)
                num_removed += 1

            first, second = np.nonzero(A)
            first_tensor = tf.convert_to_tensor(first, dtype=tf.int64)
            second_tensor = tf.convert_to_tensor(second, dtype=tf.int64)

            coupled = self.coupled
            multistep = self.multistep
            units = self.units

            # The following line of code must be used when training with AGH dataset
            cell = cells.GraphCell(units, links=len(A[0]), v=first_tensor, w=second_tensor,
                message_units=units,
                dropout=0.0,
                recurrent_dropout=0.0,
                masked=True,
                coupled=coupled)
        elif "sinx" in self.input_file:
            topology_class = NetworkTopology()
            topology = nx.read_gml(FLAGS["input_dir"].value+FLAGS["input_file"].value+"/topology_updated.gml")
            topology_class.generate_topology_from_Graph(topology)

            coupled = self.coupled
            multistep = self.multistep
            units = self.units

            cell = cells.GraphCell(units, links=topology_class.numEdges, v=topology_class.first_tensor, w=topology_class.second_tensor,
                message_units=units,
                dropout=0.0,
                recurrent_dropout=0.0,
                masked=True,
                coupled=coupled)

        hat = tf.keras.layers.RNN(cell)(seq)
        return hat

    @property
    def embed_size(self):
        return 2

    def class_model(self):
        x_shape = (None, None, 1)

        x_shape = (None, None, 1)
        inpts_link_uti = tf.keras.Input(shape=x_shape, batch_size=self.batch_size)
        norm_inpts_link_uti = inpts_link_uti

        x_shape = (None, None, 1)
        mask_pred_0 = tf.keras.Input(shape=x_shape, batch_size=self.batch_size)

        x_shape = (None, None, 1)
        mask_pred_1 = tf.keras.Input(shape=x_shape, batch_size=self.batch_size)

        x_shape = (None, None, 1)
        feat_y = tf.keras.Input(shape=x_shape, batch_size=self.batch_size)
        norm_feat_y = feat_y 

        x_shape = (None, 1)
        bool_mask = tf.keras.Input(shape=x_shape, batch_size=self.batch_size, dtype=bool)

        # The 3 indicates the axis
        concat_input = tf.concat([norm_inpts_link_uti, norm_feat_y, mask_pred_0, mask_pred_1], 3)
        hat = self.rnn(concat_input)
        # Here, hat has shape: shape=(batch_size, num_links, units)

        seq_layer = tf.keras.Sequential()
        seq_layer.add(tf.keras.layers.Dense(self.seq_layer_size))
        seq_layer.add(tf.keras.layers.Dense(self.embed_size))

        # Here, hat has shape: shape=(batch_size, num_links, hidden_state_size)
        hat = tf.keras.layers.TimeDistributed(seq_layer, input_shape=(None, self.units))(hat)

        loc = hat[..., 0]
        scale = hat[..., 1]
        
        rv = tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Masked(tfd.Laplace(
                    loc=t[0],
                    scale=1e-5 + tf.math.softplus(t[1])
                    # Uncomment the following when using single inference
                    # loc=t[..., :1],
                    # scale=1e-3 + tf.math.softplus(t[..., 1:])
                ), t[2])
        )((loc, scale, tf.squeeze(bool_mask)))# (hat) # when single inference

        # Custom loss layer for masked loss
        class CustomLossLayer(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs):
                rv = inputs[0]
                mask_pred = inputs[1]
                y_label = inputs[2]

                out = tfp.distributions.Masked(rv, mask_pred)

                #loss = -out.log_prob(y_label)
                return out # define the loss as output

        #m_loss  = CustomLossLayer(name = "CustomMaskedLossLayer")([rv, bool_mask, y_label])
        # out = tf.keras.layers.Masking(mask_pred_1)(rv)
        # out = tf.boolean_mask(rv, mask_pred_1, axis=1)

        self.model = tf.keras.Model(
            inputs=[inpts_link_uti, mask_pred_0, mask_pred_1, feat_y, bool_mask],
            outputs=rv,
            name=f'{type(self).__name__}'
        )

    def train_loop(self):
        log_dir = FLAGS.logs_dir+"/"

        # bug 34276
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=100, profile_batch=0)
        checkpoint_filepath = os.path.join(log_dir, 'checkpoint')
        summary_writer = tf.summary.create_file_writer(log_dir)
        try:
            os.mkdirs(checkpoint_filepath)
        except:
            pass

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            FLAGS.learning_rate,
            decay_steps=10,
            decay_rate=0.96,
            staircase=True)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        flags_callback = tf.keras.callbacks.LambdaCallback(
            on_train_begin=lambda _: FLAGS.append_flags_into_file(os.path.join(log_dir, 'flagfile.txt'))
        )

        self.class_model()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=0.9, epsilon=1e-05)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_metric = get_lr_metric(self.optimizer)

        # The model is trained using the negative log likelyhood loss.
        self.model.compile(optimizer=self.optimizer,
                loss=lambda y, rv_y: -rv_y.log_prob(y), metrics=[lr_metric, ChiSquared(), 'mse','mae'])

        train_set = self.train_ds()
        test_set = self.test_ds()

        h = self.model.fit(x=train_set,
                      epochs=self.num_epoch,
                      callbacks=[model_checkpoint_callback,
                      tensorboard_callback,flags_callback,
                      # Flag indicates '0' for model with no mask and '1' with model with mask
                      CustomCallback(self.model, test_set, summary_writer, FLAGS.eval_freq, 1)],
                      validation_data=test_set,
                      verbose=1)

class SpaceTimeGNN(Experiment):

    @property
    def embed_size(self):
        return 2

def scheduler(epoch):
    if epoch < 20:
        return FLAGS.learning_rate
    else:
        return FLAGS.learning_rate * tf.math.exp(0.005 * (20 - epoch))

def main(_):
    # python train_stgnn_network_wide.py
    print("***********")
    print("TRAINING USING THE GNN")
    print("***********")
    exp_cls = globals()[FLAGS.experiment]
    experiment_params = inspect.getfullargspec(exp_cls.__init__)
    e = exp_cls(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
    logging.info(e)

    # Clean logs directory
    if os.path.exists(FLAGS.logs_dir):
        os.system("rm -rf %s" % (FLAGS.logs_dir))

    if False:
        # If we use single sample inference, uncomment the (hat) from the model() function
        ds = e.compress_ds()

        for x,y in ds:
            break

        n_links = len(y[0])
        # Create new mask for predicted links. We store using one-hot encoding
        # In this iteration we have all links masked
        mask_pred_0 = np.zeros((n_links, 1))
        mask_pred_1 = np.ones((n_links, 1))
        y_label_feat = np.zeros((n_links, 1))
        mean_uti_x = np.mean(x)
        y_label_feat.fill(mean_uti_x)
        
        e.class_model()

        # Iterate over all links and predict over them. Make the loop be incremental
        for l in range(n_links):
            # If we use the model to obtain the probability distribution
            mask_pred_0.fill(1)
            mask_pred_1.fill(0)
  
            mask_pred_1[l] = 1
            mask_pred_0[l] = 0
            y_label_feat[l] = 0

            # Now we marked the link we want to make the prediction over
            t_mask_pred_1 = tf.convert_to_tensor(mask_pred_1, dtype=tf.float32)
            t_mask_pred_0 = tf.convert_to_tensor(mask_pred_0, dtype=tf.float32)

            # Now we have for the masked links 0 if it's marked for prediction,
            # the true link utilization values if they are not considered for prediction
            # or the mean utilization if it's not predicted over yet
            aux_feat_y = tf.convert_to_tensor(y_label_feat, dtype=tf.float32)

            # Repeat the feat_y and mask as many times as window_no_label
            window_no_label = 4
            aux_feat_y = tf.repeat([aux_feat_y], repeats=[window_no_label], axis=0)
            bool_mask = tf.convert_to_tensor(t_mask_pred_1>0, dtype=bool)
            t_mask_pred_0 = tf.repeat([t_mask_pred_0], repeats=[window_no_label], axis=0)
            t_mask_pred_1 = tf.repeat([t_mask_pred_1], repeats=[window_no_label], axis=0)
            yhat = e.model([x, t_mask_pred_0[tf.newaxis,...], t_mask_pred_1[tf.newaxis,...], aux_feat_y[tf.newaxis,...], bool_mask[tf.newaxis,...]])
            if l>=0:
                break
            # We use the real value as a feature
            y_label_feat[l,0] = y[0,l]

    else:
        pass

    e.train_loop()


if __name__ == '__main__':
    app.run(main)
