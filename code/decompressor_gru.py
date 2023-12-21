import keras
import os
import inspect
import glob
import tensorflow as tf
import numpy as np
import argparse
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
import sys
sys.path.insert(1, '../data')
from utils import *
import networkx as nx
import tempfile
import time as tt
import gc
import pickle
import shutil
from absl import flags, app, logging
import train_gru_single_link
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

FLAGS = flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.random.set_seed(0)
tf.keras.backend.set_floatx('float32')
np.random.seed(7)

arithm_statesize = 64

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-d', help='Dataset filename. Indicates tha dataset we want to compress', required=True)
parser.add_argument('-f', help='Flag indicating SpaceTimeGNN(0) or GRU(2)', required=True, type=int)
parser.add_argument('-m', help='Indicate folder name where to load the model from', required=True)

args = parser.parse_args()

delattr(flags.FLAGS, 'input_file')
flags.DEFINE_string("input_file", args.d, "Input file")
delattr(flags.FLAGS, 'batch_size')
flags.DEFINE_integer("batch_size", 1, "...")

# This is done to point to the proper logs dir to load the model
if args.f==0:
    delattr(flags.FLAGS, 'logs_dir')
    delattr(flags.FLAGS, 'coupled')
    flags.DEFINE_string('logs_dir', "log/"+args.m+"_GNN", "Model directory for logs an checkpoints")
    flags.DEFINE_bool('coupled', True, "...") # Uncomment for GRU
elif args.f==2:
    delattr(flags.FLAGS, 'logs_dir')
    delattr(flags.FLAGS, 'coupled')
    flags.DEFINE_string('logs_dir', "log/"+args.m+"_GRU", "Model directory for logs an checkpoints")
    flags.DEFINE_bool('coupled', False, "...") # Uncomment for GRU


@tf.function(jit_compile=True)
def model_inference(model, x, vals):
    x = tf.math.log(x)
    yhat = model(x)
    z = tfb.Exp()(yhat)
    qtz_distrib = tfd.QuantizedDistribution(z)
    prob_val = qtz_distrib.prob(vals[... , tf.newaxis])
    return prob_val

def main():
    # python decompressor_gru.py -d KB_agh_w5_r20_maxt102799 -f 2 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
    dataset_name = args.d
    original_data_dir = "../data/files_to_be_compressed/tfrecords/"+dataset_name+"/"
    input_file_tfrecord = original_data_dir+dataset_name+'.tfrecords'
    compressed_dir = "../data/compressed/"+dataset_name+"/"
    
        # Add extension to name according to prediction
    pred_flag = args.f
    pred_extension = "_GNN"
    if pred_flag==2:
        pred_extension = "_GRU"

    links_extension_dir = "links_"+args.m+pred_extension

    params = dict()
    with open(compressed_dir+dataset_name+"_params", 'r') as f:
        params = json.load(f)

    # Load ORIGINAL np array
    original = np.load(original_data_dir+"per_link_ORIGINAL_ts.npy")

    with open(original_data_dir+"links_to_remove.pkl", 'rb') as fp:
            list_zero_link_uti = pickle.load(fp)

    n_links = params["num_links"]
    num_time_steps = params["num_samples"]+params["window_size"]-1
    reconstructed_TS = np.zeros((n_links,num_time_steps), dtype = np.uint32)

    # -1 because we remove the label
    sequence_length = params['window_size']-1
    min_y = int(params["min_y"])
    max_y = int(params["max_y"])
    # +1 because we want to include the max_y-th element
    vals = np.arange(min_y,max_y+1)
    alphabet_size = int(max_y-min_y+1)

    # Load tf records
    experiment_params = inspect.getfullargspec(train_gru_single_link.SpaceTimeGNN.__init__)
    e = train_gru_single_link.SpaceTimeGNN(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
    logging.info(e)
    checkpoint_filepath = os.path.join(e.logs_dir, 'checkpoint')

    # Load model weights
    model = e.model()

    # No need to load the model if we use perfect prediction upper bound
    if pred_flag!=1:
        print("RESTORING MODEL: ", checkpoint_filepath)
        model.load_weights(checkpoint_filepath).expect_partial()

    # Iterate over dataset
    ds = e.compress_ds()

    # Create the uniform prob distribution
    prob = np.ones(alphabet_size, dtype = np.uint32)/alphabet_size
    cumul = np.zeros(alphabet_size+1, dtype = np.uint32)
    cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  

    f_in = [open(compressed_dir+links_extension_dir+"/link"+"_"+str(i)+'.compressed','rb') for i in range(n_links)]
    bitin = [arithmeticcoding_fast.BitInputStream(f_in[i]) for i in range(n_links)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(arithm_statesize, bitin[i]) for i in range(n_links)]

    # We store the times it takes to code a single time-step
    list_times = list()
    first_sample = True
    cnt_timestep = 0
    # Iterate over all samples and decompress them
    main_start = tt.time()
    for x, y in ds:
        start = tt.time()
        # tf.shape(x): tf.Tensor([batch_size time_steps num_links 1], shape=(4,), dtype=int32)
        if first_sample:
            # Decode the first time steps using uniform distribution. The last one will be encoded using 
            # the probabilities obtained from the model

            # Iterate over timesteps
            for t in range(sequence_length):
                # Iterate over links
                for l in range(n_links):
                    reconstructed_TS[l, cnt_timestep] = dec[l].read(cumul, alphabet_size)+min_y
                cnt_timestep += 1
            
            first_sample = False
        
        if pred_flag!=1:
            prob_val = model_inference(model, x, tf.cast(vals, tf.float32))
            
        list_diffs_label = list()
        # Iterate over all links
        for l in range(n_links):
            cumul.fill(0)
            if pred_flag==3:
                stdev = 0.1
                distrib = tfd.QuantizedDistribution(tfd.Laplace(loc=int(reconstructed_TS[l, cnt_timestep-1]), scale=stdev))
                prob = distrib.prob(vals)
                cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)
            else:
                cumul[1:] = np.cumsum(prob_val[:,l]*10000000 + 1, axis = 0)
            reconstructed_TS[l, cnt_timestep] = dec[l].read(cumul, alphabet_size)+min_y
        
        end = tt.time()
        list_times.append(end-start)
        
        if cnt_timestep%500==0:
            print("Decoded time-step ", pred_extension, ": ", cnt_timestep, "/", params["num_samples"]+params["window_size"]-1)    
            print("  Mean Decoding time per time-step (s): ", np.mean(list_times) )
            print("  Difference:", np.sum(np.abs(original[:,:cnt_timestep]-reconstructed_TS[:,:cnt_timestep])))
            list_times.clear()
            gc.collect()
        cnt_timestep += 1

    main_end = tt.time()
    print("Difference:", np.sum(np.abs(original-reconstructed_TS)))
    print("Total decoding time(in s): ", main_end-main_start)

    np.save(compressed_dir+"per_link_RECONSTRUCTED_ts"+pred_extension+".npy", reconstructed_TS)
    # close files
    for i in range(n_links):
        bitin[i].close()
        f_in[i].close()
                                        
if __name__ == "__main__":
    main()

