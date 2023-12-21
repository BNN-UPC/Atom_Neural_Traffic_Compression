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
import time as tt
import gc
#import models
import matplotlib.pyplot as plt
import tempfile
import shutil
from absl import flags, app, logging
import train_gru_single_link
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

FLAGS = flags.FLAGS

# This script uses the GNN model without masked links

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.random.set_seed(0)
tf.keras.backend.set_floatx('float32')
np.random.seed(7)

arithm_statesize = 64

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-d', help='Dataset filename. Indicates tha dataset we want to compress', required=True)
parser.add_argument('-f', help='Flag indicating SpaceTimeGNN(0) or GRU(2)', required=True, type=int)
parser.add_argument('-r', help='Indicate if remove old dir(1) or not(0)', required=True, type=int)
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
def model_inference(limit_size_cdf_hist, list_cdfs, model, x, vals, y):
    x = tf.math.log(x)
    yhat = model(x)
    z = tfb.Exp()(yhat)
    #z = tfb.Scale(99)(yhat)
    qtz_distrib = tfd.QuantizedDistribution(z)
    prob_val = qtz_distrib.prob(vals[... , tf.newaxis])
    cdf_values = tf.cond(list_cdfs < limit_size_cdf_hist, lambda: yhat.cdf(tf.math.log(y)), lambda: tf.zeros(tf.shape(y)))
    #cdf_values = tf.cond(list_cdfs < limit_size_cdf_hist, lambda: yhat.cdf(y), lambda: tf.zeros(tf.shape(y)))
    return prob_val, cdf_values, yhat.scale

def main():
    # python compressor_gru.py -d KB_agh_w5_r20_maxt102799 -f 2 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
    dataset_name = args.d
    original_data_dir = "../data/files_to_be_compressed/tfrecords/"+dataset_name+"/"
    input_file_tfrecord = original_data_dir+dataset_name+'.tfrecords'
    compressed_dir = "../data/compressed/"+dataset_name+"/"

    if not os.path.exists("../data/Images/"):
        os.makedirs("../data/Images/")
    
    if not os.path.exists("../data/Images/"+dataset_name):
        os.makedirs("../data/Images/"+dataset_name)
    
    # Add extension to name according to prediction
    pred_flag = args.f
    pred_extension = "_GNN"
    if pred_flag==2:
        pred_extension = "_GRU"

    links_extension_dir = "links_"+args.m+pred_extension

    # Load ORIGINAL np array
    original = np.load(original_data_dir+"per_link_ORIGINAL_ts.npy")

    if not os.path.exists(compressed_dir):
        os.makedirs(compressed_dir)
    # Clean old directory
    if args.r:
        os.system("rm -rf %s/*" % (compressed_dir))
    if not os.path.exists(compressed_dir+links_extension_dir):
        os.makedirs(compressed_dir+links_extension_dir)

    params = dict()
    with open(original_data_dir+dataset_name+"_params", 'r') as f:
        params = json.load(f)

    # We copy the params file to the compressed dir to have everything important together
    os.system("cp %s %s" % (original_data_dir+dataset_name+"_params", compressed_dir))

    # -1 because we remove the label
    sequence_length = params['window_size']-1
    # Set to 0 because when the links never have bw 0 there is an error
    # with the range of values to predict the distribution over
    min_y = int(params["min_y"])
    max_y = int(params["max_y"])
    # +1 because we want to include the max_y-th element
    vals = np.arange(min_y,max_y+1)
    alphabet_size = int(max_y-min_y+1)

    ################### Load tf records
    experiment_params = inspect.getfullargspec(train_gru_single_link.SpaceTimeGNN.__init__)
    e = train_gru_single_link.SpaceTimeGNN(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
    logging.info(e)
    checkpoint_filepath = os.path.join(e.logs_dir, 'checkpoint')

    # Load model weights
    model = e.model()

    # No need to load the model if we use perfect prediction upper bound
    if pred_flag!=1:
        print("  RESTORING MODEL: ", checkpoint_filepath)
        model.load_weights(checkpoint_filepath).expect_partial()

    # Iterate over dataset
    ds = e.compress_ds()

    # Create the uniform prob distribution
    prob = np.ones(alphabet_size, dtype = np.uint32)/alphabet_size
    cumul = np.zeros(alphabet_size+1, dtype = np.uint32)
    cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  

    n_links = params["num_links"]
    limit_size_cdf_hist = tf.cast(20000, tf.float32)
    # We encode the traffic for each link separately
    f_cmpr = [open(compressed_dir+links_extension_dir+"/link"+"_"+str(i)+'.compressed','wb') for i in range(n_links)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f_cmpr[i]) for i in range(n_links)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(arithm_statesize, bitout[i]) for i in range(n_links)]

    reconstructed_TS = np.zeros((n_links,params["num_samples"]+params["window_size"]-1), dtype = np.uint32)

    first_sample = True
    cdf_printed = False
    cnt_timestep = 0

    # We store the times it takes to code a single time-step
    list_times = list()
    list_all_times = list()
    # The following list is used for visualization purposes
    list_cdfs = list()
    list_scale = list()
    main_start = tt.time()
    # Iterate over all samples and compress them
    for x, y in ds:
        start = tt.time()
        # tf.shape(x): tf.Tensor([batch_size time_steps num_links 1], shape=(4,), dtype=int32)
        if first_sample:
            # Encode the first time steps using uniform distribution. The last one will be encoded using 
            # the probabilities obtained from the model
            # Iterate over batched samples. For now there is only 1 sample in the batch
            for b in range(len(x)):
                # Iterate over timesteps
                for t in range(len(x[b])):
                    # Iterate over links
                    for l in range(len(x[b,t])):
                        # Encode value using uniform probs
                        enc[l].write(cumul, int(x[b,t,l,0])-min_y)
                        reconstructed_TS[l, cnt_timestep] = int(x[b,t,l,0])
                    cnt_timestep += 1

            first_sample = False

        # tf.shape(prob_val) = tf.Tensor([alphabet_size   num_batch  num_links], shape=(3,), dtype=int32)
        #prob_val = yhat.prob(vals[..., tf.newaxis, tf.newaxis])
        prob_val, cdf_values, scale = model_inference(limit_size_cdf_hist, tf.cast(len(list_cdfs), tf.float32), model, 
                                x, tf.cast(vals, tf.float32), y)
        
        # The following list is used for visualization purposes
        if len(list_cdfs)<limit_size_cdf_hist:
            list_cdfs += list(cdf_values[0].numpy())
            list_scale += list(scale[0].numpy())
        elif not cdf_printed:
            plt.hist(list_cdfs, bins=50, edgecolor='black')
            plt.grid(axis='y')
            plt.ylabel("Count", fontsize=16)
            # plt.hist(list_cdfs,density=True,histtype='step', cumulative=True, bins=10000, edgecolor='black')
            # plt.ylabel("CDF", fontsize=16)
            plt.xlabel("y_dist.cdf(y)", fontsize=16)
            #plt.show()

            plt.tight_layout()
            plt.savefig("../data/Images/"+dataset_name+"/hist_cdf"+pred_extension+".pdf",bbox_inches='tight')
            plt.close()

            plt.hist(list_scale, bins=np.arange(0, 3, 0.05), color = "orange", edgecolor='black')
            plt.grid(axis='y')
            plt.ylabel("Count", fontsize=16)
            plt.xlabel("Scale value", fontsize=16)
            #plt.show()

            plt.tight_layout()
            plt.savefig("../data/Images/"+dataset_name+"/hist_scale"+pred_extension+".pdf",bbox_inches='tight')
            plt.close()

            cdf_printed = True

        list_diffs_label = list()
        # Iterate over batched samples
        for b in range(len(y)):
            # Iterate over all links
            for l in range(len(y[b])):
                cumul.fill(0)
                if pred_flag==3:
                    stdev = 1
                    distrib = tfd.QuantizedDistribution(tfd.Laplace(loc=int(x[b,sequence_length-1,l,0]), scale=stdev))
                    prob = distrib.prob(vals)
                    cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)
                else:
                    # Encode y value using the probs from the GNN
                    cumul[1:] = np.cumsum(prob_val[:,l]*10000000 + 1, axis = 0)

                enc[l].write(cumul, int(y[b,l])-min_y)
                reconstructed_TS[l, cnt_timestep] = int(y[b,l])

        end = tt.time()
        list_times.append(end-start)
        list_all_times.append(end-start)

        if cnt_timestep%500==0:
            print("Encoded time-step " , pred_extension, ": ", cnt_timestep, "/", params["num_samples"]+params["window_size"]-1)    
            print("  Mean Coding time per time-step (s): ", np.mean(list_times) )
            print("  Difference:", np.sum(np.abs(original[:,:cnt_timestep]-reconstructed_TS[:,:cnt_timestep])))
            list_times.clear()
            gc.collect()
        cnt_timestep += 1

    main_end = tt.time()
    # Store the compression cost in time for each bin
    np.save(compressed_dir+"/times_"+links_extension_dir+".npy", np.asarray(list_all_times))

    # We only store the cdf image if we are using the SpaceTimeGNN
    if pred_flag!=1 and not cdf_printed:
        plt.hist(list_cdfs, bins=50, edgecolor='black')
        plt.grid(axis='y')
        plt.ylabel("Count", fontsize=16)
        # plt.hist(list_cdfs,density=True,histtype='step', cumulative=True, bins=10000, edgecolor='black')
        # plt.ylabel("CDF", fontsize=16)
        plt.xlabel("y_dist.cdf(y)", fontsize=16)
        #plt.show()

        plt.tight_layout()
        plt.savefig("../data/Images/"+dataset_name+"/hist_cdf"+pred_extension+".pdf",bbox_inches='tight')
        plt.close()

        plt.hist(list_scale, bins=50, color = "orange", edgecolor='black')
        plt.grid(axis='y')
        plt.ylabel("Count", fontsize=16)
        plt.xlabel("Scale value", fontsize=16)
        #plt.show()

        plt.tight_layout()
        plt.savefig("../data/Images/"+dataset_name+"/hist_scale"+pred_extension+".pdf",bbox_inches='tight')
        plt.close()

    # close files
    for i in range(n_links):
        enc[i].finish()
        bitout[i].close()
        f_cmpr[i].close()
    
    print("Num total timesteps: ", cnt_timestep)
    print("Total coding time(in s): ", np.sum(list_all_times))
    print("Mean coding time per time bin(in s): ", np.mean(list_all_times))
          
                                        
if __name__ == "__main__":
    main()
