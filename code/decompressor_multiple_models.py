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
import pickle
sys.path.insert(1, '../data')
from utils import *
import networkx as nx
import tempfile
import time as tt
import shutil
import gc
from absl import flags, app, logging
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

FLAGS = flags.FLAGS

tf.random.set_seed(0)
tf.keras.backend.set_floatx('float32')
np.random.seed(7)

arithm_statesize = 64

# If set we order the compression by taking the links with lower std first
ordered_compression = "_ordered"
# If set we dont order the compression by taking the links with lower std first
#ordered_compression = "_no_order"

parser = argparse.ArgumentParser(description='Input')
parser.add_argument('-d', help='Dataset filename. Indicates tha dataset we want to compress', required=True)
parser.add_argument('-f', help='Flag indicating SpaceTimeGNN(0), Static AC or Overall Statistics(1), Overall Statistics per link (9), Adaptive AC(10), Adaptive AC per link (11)', required=True, type=int)
parser.add_argument('-m', help='Indicate folder name where to load the model from', required=True)

args = parser.parse_args()

# This is done to point to the proper logs dir to load the model
if args.f==0 or args.f==1 or args.f==9 or args.f==10 or args.f==11:
    import train_stgnn_network_wide
    delattr(flags.FLAGS, 'logs_dir')
    delattr(flags.FLAGS, 'coupled')
    delattr(flags.FLAGS, 'batch_size')
    flags.DEFINE_integer("batch_size", 1, "...")
    flags.DEFINE_string('logs_dir', "log/"+args.m+"_masked_GNN", "Model directory for logs an checkpoints")
    flags.DEFINE_bool('coupled', True, "...") # Uncomment for GRU

delattr(flags.FLAGS, 'input_file')
flags.DEFINE_string("input_file", args.d, "Input file")


@tf.function(jit_compile=True)
def model_inference(window_no_label, model, x, mask_pred_0, mask_pred_1, y_label_feat, vals, y):
    # Now we marked the link we want to make the prediction over
    t_mask_pred_1 = tf.convert_to_tensor(mask_pred_1, dtype=tf.float32)
    t_mask_pred_0 = tf.convert_to_tensor(mask_pred_0, dtype=tf.float32)

    # Now we have for the masked links 0 if it's marked for prediction,
    # the true link utilization values if they are not considered for prediction
    # or the mean utilization if it's not predicted over yet
    aux_feat_y = tf.convert_to_tensor(y_label_feat, dtype=tf.float32)

    # Repeat the feat_y and mask as many times as window_no_label
    aux_feat_y = tf.repeat([aux_feat_y], repeats=[window_no_label], axis=0)
    bool_mask = tf.convert_to_tensor(t_mask_pred_1>0, dtype=bool)
    t_mask_pred_0 = tf.repeat([t_mask_pred_0], repeats=[window_no_label], axis=0)
    t_mask_pred_1 = tf.repeat([t_mask_pred_1], repeats=[window_no_label], axis=0)
                
    yhat = model([tf.math.log(x), t_mask_pred_0[tf.newaxis,...], t_mask_pred_1[tf.newaxis,...], aux_feat_y[tf.newaxis,...], bool_mask[tf.newaxis,...]])
    z = tfb.Exp()(yhat)
    qtz_distrib = tfd.QuantizedDistribution(z)
    prob_val = qtz_distrib.prob(vals[... , tf.newaxis])
    return prob_val, yhat.parameters['distribution'].scale


def main():
    # python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
    dataset_name = args.d
    original_data_dir = "../data/files_to_be_compressed/tfrecords/"+dataset_name+"/"
    input_file_tfrecord = original_data_dir+dataset_name+'.tfrecords'
    compressed_dir = "../data/compressed/"+dataset_name+"/"
    
    # Add extension to name according to prediction
    pred_flag = args.f
    pred_extension = "_masked_GNN"
    links_extension_dir = "links_"+args.m+pred_extension+ordered_compression
    if pred_flag==1:
        pred_extension = "_overall_stats"
        links_extension_dir = "links"+pred_extension
    elif pred_flag==9:
        pred_extension = "_overall_stats_link"
        links_extension_dir = "links"+pred_extension
    elif pred_flag==10:
        pred_extension = "_adaptive_AC"
        links_extension_dir = "links"+pred_extension
    elif pred_flag==11:
        pred_extension = "_adaptive_AC_link"
        links_extension_dir = "links"+pred_extension

    params = dict()
    with open(compressed_dir+dataset_name+"_params", 'r') as f:
        params = json.load(f)

    # Load ORIGINAL np array
    original = np.load(original_data_dir+"per_link_ORIGINAL_ts.npy")

    n_links = params["num_links"]
    window_no_label = params["window_size"]-1

    num_time_steps = params["num_samples"]+params["window_size"]-1
    reconstructed_TS = np.zeros((n_links,num_time_steps), dtype = np.uint32)

    # -1 because we remove the label
    sequence_length = params['window_size']-1

    min_y = int(params["min_y"])
    max_y = int(params["max_y"])
    delta = tf.cast(1, tf.float32)
    # +1 because we want to include the max_y-th element
    vals = np.arange(min_y,max_y+1, dtype=np.uint32)
    # -min_y because we shift to 0 the vals
    alphabet_size = int(max_y-min_y+1)

    ################### Load tf records
    if pred_flag!=4:
        experiment_params = inspect.getfullargspec(train_stgnn_network_wide.SpaceTimeGNN.__init__)
        e = train_stgnn_network_wide.SpaceTimeGNN(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
        logging.info(e)
        checkpoint_filepath = os.path.join(e.logs_dir, 'checkpoint')

        # Load model weights
        e.class_model()

    # No need to load the model if we use perfect prediction upper bound
    if pred_flag==0:
        print("  RESTORING MODEL: ", checkpoint_filepath)
        e.model.load_weights(checkpoint_filepath).expect_partial()

    # Iterate over dataset
    ds = e.compress_ds()

    if pred_flag==1:
        # We use the generic probability distribution from entire dataset
        prob = np.load(original_data_dir+"/overall_statistics_probs.npy")
        cumul = np.zeros(alphabet_size+1, dtype = np.uint32)
        cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  
    elif pred_flag==9:
        print("::::::")
        print("::::::")
        total_link_uti_values = len(original[0])

        print("COMPUTING PROBABILITY DISTRIBUTION PER-LINK TAKING INTO ACCOUNT THE ENTIRE DATASET...")
        prob = np.zeros((n_links, alphabet_size), dtype = np.float32)
        for l in range(n_links):
            link_uti_hist = dict()
            # Iterate over all time bins of the link and store the frequencies
            for time_bin in range(total_link_uti_values):
                link_uti_val = str(int(original[l,time_bin]))
                if link_uti_val in link_uti_hist:
                    link_uti_hist[link_uti_val] += 1
                else:
                    link_uti_hist[link_uti_val] = 1

            for key, value in link_uti_hist.items():
                prob[l, int(key)-min_y] = float(value/total_link_uti_values)
            
            link_uti_hist.clear()
        
        print("Finished computing probability distribution")
        print("::::::")
        np.save(original_data_dir+"overall_statistics_probs_links.npy", prob)
        cumul = np.zeros((n_links, alphabet_size+1), dtype = np.uint32)
        for l in range(n_links):
            cumul[l,1:] = np.cumsum(prob[l]*10000000 + 1, dtype=np.uint32)
    else:
        # Create the uniform prob distribution
        prob = np.ones(alphabet_size, dtype = np.uint32)/alphabet_size
        cumul = np.zeros(alphabet_size+1, dtype = np.uint32)
        cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  

    f_in = [open(compressed_dir+links_extension_dir+"/link"+"_"+str(i)+'.compressed','rb') for i in range(n_links)]
    bitin = [arithmeticcoding_fast.BitInputStream(f_in[i]) for i in range(n_links)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(arithm_statesize, bitin[i]) for i in range(n_links)]

    # We store the times it takes to code a single time-step
    list_times = list()
    first_timestep = True
    cnt_timestep = 0
    total_link_uti_values = 0
    # Iterate over all samples and decompress them
    # The dataset in this case is the real one without log transformation
    main_start = tt.time()
    # We iterate over the dataset to use the X values
    for x,y in ds:
        start = tt.time()
        # tf.shape(x): tf.Tensor([batch_size time_steps num_links 1], shape=(4,), dtype=int32)
        if first_timestep:
            total_link_uti_values = len(x[0])*len(x[0,0])
            # Decode the first time steps using uniform distribution. The last one will be encoded using 
            # the probabilities obtained from the model
            # Iterate over timesteps
            for t in range(sequence_length):
                # Iterate over links
                for l in range(n_links):
                    if pred_flag==9:
                        reconstructed_TS[l, cnt_timestep] = dec[l].read(cumul[l], alphabet_size)+min_y
                    else:
                        reconstructed_TS[l, cnt_timestep] = dec[l].read(cumul, alphabet_size)+min_y
                cnt_timestep += 1
            
            first_timestep = False
        # Create new mask for predicted links. We store using one-hot encoding
        mask_pred_0 = np.zeros((n_links, 1))
        mask_pred_1 = np.zeros((n_links, 1))
        y_label_feat = np.zeros((n_links, 1))
        
        # We mark all links for prediction
        mask_pred_1.fill(1)

        # Iterate over all discarded links and predict over them. Make the loop be incremental
        for link in range(n_links):
            link_to_decompress = link
            # If we use the GNN
            if pred_flag==0:
                cumul.fill(0)

                prob_val, scale = model_inference(window_no_label, e.model, tf.cast(x, tf.float32), tf.cast(mask_pred_0, tf.float32), tf.cast(mask_pred_1, tf.float32), tf.cast(y_label_feat, tf.float32), tf.cast(vals, tf.float32), tf.cast(y, tf.float32))

                if "ordered" in ordered_compression:
                    list_tuple = list()

                    counter = 0
                    for s in scale[0]:
                        # Store the scale value, the real position and the mask
                        list_tuple.append((s.numpy(), counter, mask_pred_1[counter][0]))
                        counter += 1
                    
                    # Take the link with lowest scale first
                    list_tuple = sorted(list_tuple, key=lambda x: (x[0], x[1]))#, reverse=True)
                    for tup in list_tuple:
                        if tup[2]==1:
                            # Take the real position of the next link to compress
                            # This link should be masked as unkown!
                            link_to_decompress = tup[1]
                            break

                # Encode y value using the probs from the GNN
                cumul[1:] = np.cumsum(prob_val[:,link_to_decompress]*10000000 + 1, axis = 0)
                # We use the real value as a feature
                y_label_feat[link_to_decompress,0] = tf.math.log(y[0,link_to_decompress])
                # We mark the links as known
                mask_pred_1[link_to_decompress] = 0
                mask_pred_0[link_to_decompress] = 1
            elif pred_flag==10 and link==0: # Adaptive AC
                # We just compute once the prob distribution
                cumul.fill(0)
                prob.fill(0)
                flattened_x = np.squeeze(np.concatenate(x[0]))
                unique_elements, counts_elements = np.unique(flattened_x, return_counts=True)
                
                # Compute dynamic prob distribution
                for pos in range(len(unique_elements)):
                    prob[int(unique_elements[pos])-min_y] = float(counts_elements[pos]/total_link_uti_values)
                
                cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  
            elif pred_flag==11: # Adaptive AC
                # We just compute once the prob distribution
                cumul.fill(0)
                prob.fill(0)
                flattened_x = np.squeeze(np.concatenate(x[0,:,link_to_decompress]))
                unique_elements, counts_elements = np.unique(flattened_x, return_counts=True)
                
                # Compute dynamic prob distribution
                for pos in range(len(unique_elements)):
                    prob[int(unique_elements[pos])-min_y] = float(counts_elements[pos]/(params["window_size"]-1))
                
                cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  

            if pred_flag==9:
                # Decode real value using probs
                # we shift the link utilizations to go from 0 to max_uti
                reconstructed_TS[link_to_decompress, cnt_timestep] = dec[link_to_decompress].read(cumul[l], alphabet_size)+min_y
            else:
                # Decode real value using probs
                # we shift the link utilizations to go from 0 to max_uti
                reconstructed_TS[link_to_decompress, cnt_timestep] = dec[link_to_decompress].read(cumul, alphabet_size)+min_y

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

