import datetime
import inspect
import os
from dataclasses import dataclass
from functools import partial
import json
import sys
sys.path.insert(1, '../data')
from utils import *
import cells
import numpy as np
import networkx as nx
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags, app, logging

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.random.set_seed(0)
tf.keras.backend.set_floatx('float32')
np.random.seed(7)

tfd = tfp.distributions
tfb = tfp.bijectors

FLAGS = flags.FLAGS

# 'Flag indicating SpaceTimeGNN(0) or GRU(2)'
pred_flag = 2

dataset_name = "geant_w5_r40_maxt6063"
dataset_name = "abilene_w5_r40_maxt41741"
dataset_name = "KB_agh_w5_r20_maxt102799"

units = 50
seq_layer_size = 10
batch_size = 256

log_dataset_name = dataset_name+"_u"+str(units)+"_s"+str(seq_layer_size)+"_b"+str(batch_size)

flags.DEFINE_string("input_dir", "../data/files_to_be_compressed/tfrecords/", "Input file")
flags.DEFINE_string("input_file", dataset_name, "Input file")
flags.DEFINE_integer("window_size", 1, "...")
flags.DEFINE_integer("perc_training", 70, "...") 
flags.DEFINE_integer("batch_size", batch_size, "...")
flags.DEFINE_integer("future", 1, "...")
flags.DEFINE_integer("buffer_size", 1000, "...") 
flags.DEFINE_integer("repeat_train", 1, "...")
flags.DEFINE_integer("num_epoch", 1000, "...")
flags.DEFINE_integer("eval_freq", 100, "...")
flags.DEFINE_integer("units", units, "...")
flags.DEFINE_integer("seq_layer_size", seq_layer_size, "...")
# flags.DEFINE_string('tag', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "Name of the run")
if pred_flag==0:
    flags.DEFINE_bool('coupled', True, "...")
    flags.DEFINE_string('logs_dir', "log/"+log_dataset_name+"_GNN", "Model directory for logs an checkpoints")
elif pred_flag==2:
    flags.DEFINE_string('logs_dir', "log/"+log_dataset_name+"_GRU", "Model directory for logs an checkpoints")
    flags.DEFINE_bool('coupled', False, "...") # Uncomment for GRU

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
       
    def dataset(self, flag_training):
        params = dict()
        with open(self.input_dir+self.input_file+"/"+self.input_file+"_params", 'r') as f:
            params = json.load(f)

        self.window_size = params["window_size"]
        self.buffer_size = params["num_samples"]
        self.num_train_samples = int((params["num_samples"]*self.perc_training)/100)
        self.num_eval_samples = params["num_samples"]-self.num_train_samples

        # Input sample has shape=(32, 24, 18, 1) for (batch, timestep, link, feature)
        # Output sample has shape=(32, 18) for (batch, link)
        ds = tf.data.TFRecordDataset(self.input_dir+self.input_file+"/"+self.input_file+'.tfrecords')

        # Create a dictionary describing the features.
        data = {
            'link_uti': tf.io.VarLenFeature(tf.float32),
            'next_uti': tf.io.VarLenFeature(tf.float32),
        }

        def _parse_function(example_proto, flag_training):
            # Parse the input tf.train.Example proto using the dictionary above.
            parsed_data = tf.io.parse_single_example(example_proto, data)
            x, y = tf.sparse.to_dense(parsed_data["link_uti"]), tf.sparse.to_dense(parsed_data["next_uti"])
            window_no_label = self.window_size-1 # -1 to exclude the label
            x = tf.reshape(x, (window_no_label,  params["num_links"], 1))
            x_log = tf.math.log(x)
            y_log = tf.math.log(y)
            if flag_training:
                # For training, we need the link utilizations after the log transformation
                return x_log, y_log
            else:
                # For compression/decompression, we need the raw link utilizations
                return x, y

        ds = ds.map(lambda x: _parse_function(x, flag_training))
        return ds

    def train_ds(self):
        ds = self.dataset(True)
        ds = ds.take(self.num_train_samples)
        ds = ds.shuffle(self.num_train_samples)
        ds = ds.cache()
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.repeat(self.repeat_train)
        ds = ds.prefetch(1)
        return ds
    
    def compress_ds(self):
        ds = self.dataset(False)
        # ds = ds.take(self.buffer_size)
        ds = ds.cache()
        ds = ds.batch(1, drop_remainder=True)
        return ds

    def test_ds(self):
        ds = self.dataset(True)
        ds = ds.skip(self.num_train_samples)
        ds = ds.take(self.num_eval_samples)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.cache()
        return ds

    def final_test_ds(self):
        ds = self.dataset(True)
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
        ds = self.dataset() # We do this to update the self.num_train_samples
        ds = ds.take(self.num_train_samples).map(
            lambda x, y: x
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
                masked=False,
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
                masked=False,
                coupled=coupled)
        # Campus network dataset
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
                masked=False,
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

    def model(self):
        with open(FLAGS["input_dir"].value+FLAGS["input_file"].value+"/links_to_remove.pkl", 'rb') as fp:
            list_zero_link_uti = pickle.load(fp)

        x_shape = (None, None, 1)

        inputs = tf.keras.Input(shape=x_shape, batch_size=self.batch_size)

        # bijector = tfb.Chain([
        #     #tfb.Exp(),
        #     tfb.Shift(norm_layer.mean),
        #     tfb.Scale(tf.sqrt(norm_layer.variance)),
        #     ])

        # z = bijector.inverse(inputs[..., 0])[..., tf.newaxis]

        hat = self.rnn(inputs)
        # Here, hat has shape: shape=(batch_size, num_links, units)

        seq_layer = tf.keras.Sequential()
        seq_layer.add(tf.keras.layers.Dense(self.seq_layer_size))
        #seq_layer.add(tf.keras.layers.Dense(self.embed_size*4))
        seq_layer.add(tf.keras.layers.Dense(self.embed_size))

        hat = tf.keras.layers.TimeDistributed(seq_layer, input_shape=(None, self.units))(hat)
        # Here, hat has shape: shape=(batch_size, num_links, self.embed_size)

        loc = hat[..., 0]
        scale = hat[..., 1]
 
        # lambda t: tfd.QuantizedDistribution(tfd.TransformedDistribution ...
        rv = tfp.layers.DistributionLambda(
            lambda t: tfd.Laplace(
                        loc=t[0],
                        scale=1e-5 + tf.math.softplus(t[1])
                        # Uncomment the following when using single inference
                        # loc=t[..., :1],
                        # scale=1e-3 + tf.math.softplus(t[..., 1:])
                    )
        )((loc, scale))# (hat) # when single inference

        model = tf.keras.Model(
            inputs=inputs,
            outputs=rv,
            name=f'{type(self).__name__}'
        )
        return model

    def train_loop(self):
        log_dir = FLAGS.logs_dir

        # bug 34276
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=50, profile_batch=0)
        checkpoint_filepath = os.path.join(FLAGS.logs_dir, 'checkpoint')
        summary_writer = tf.summary.create_file_writer(log_dir)
        try:
            os.mkdirs(checkpoint_filepath)
        except:
            pass
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        flags_callback = tf.keras.callbacks.LambdaCallback(
            on_train_begin=lambda _: FLAGS.append_flags_into_file(os.path.join(log_dir, 'flagfile.txt'))
        )
        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)

        model = self.model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate, beta_1=0.9, epsilon=1e-05)
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_metric = get_lr_metric(optimizer)

        model.compile(optimizer=optimizer,
            loss=lambda y, rv_y: -rv_y.log_prob(y), metrics=[lr_metric, ChiSquared(), 'mse','mae'])

        # logging.info(model.summary())
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0005, patience=5, verbose=1)

        train_set = self.train_ds()
        test_set = self.test_ds()

        h = model.fit(x=train_set,
                      callbacks=[tensorboard_callback,
                                 model_checkpoint_callback,
                                 # Flag indicates '0' for model with no mask and '1' with model with mask
                                 flags_callback,CustomCallback(model, test_set, summary_writer, FLAGS.eval_freq, 0)],
                      epochs=self.num_epoch,
                      validation_data=test_set,
                      verbose=1)
        return h, model

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
    # python train_gru_single_link.py
    if pred_flag==0:
        print("***********")
        print("TRAINING USING THE GNN")
        print("***********")
    elif pred_flag==2:
        print("###########")
        print("TRAINING USING THE GRU")
        print("###########")
    
    exp_cls = globals()[FLAGS.experiment]
    experiment_params = inspect.getfullargspec(exp_cls.__init__)
    e = exp_cls(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
    logging.info(e)

    # Clean logs directory
    if os.path.exists(FLAGS.logs_dir):
        os.system("rm -rf %s" % (FLAGS.logs_dir))

    if False:
        # If we use single sample inference, uncomment the (hat) from the model() function
        ds = e.train_ds()

        cnt_samples = 0
        for x, y in ds:
            # print(x, y)
            #x = tf.math.log(x)
            model = e.model()
            print("X: ", x)
            vals = np.arange(-1000,1000)
            yhat = model(x)
            print("yhat: ", yhat)
            # print("yhat.log_prob: ", yhat.log_prob(vals))
            #z = tfb.Exp()(yhat)
            #qtz_distrib = tfd.QuantizedDistribution(z)
            #prob_val = qtz_distrib.prob(vals[... , tf.newaxis])
            #print("yhat.prob: ", prob_val)
            #print(np.sum(prob_val))
            if cnt_samples>=0:
                break
            cnt_samples += 1
    else:
        pass

    h, model = e.train_loop()


if __name__ == '__main__':
    app.run(main)
