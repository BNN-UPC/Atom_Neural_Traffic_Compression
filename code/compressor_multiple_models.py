import keras
import os
import inspect
import glob
import tensorflow as tf
import numpy as np
import argparse
import time
import contextlib
import arithmeticcoding_fast
import json
from tqdm import tqdm
import struct
import matplotlib.pyplot as plt
import time as tt
from memory_profiler import profile
import tempfile
import gzip
import gc
import psutil
import shutil
import multiprocessing
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
parser.add_argument('-r', help='Indicate if remove old dir(1) or not(0)', required=True, type=int)
parser.add_argument('-m', help='Indicate folder name where to load the model from', required=True)

args = parser.parse_args()

def worker_execute(args):
    os_pid = args[0]
    compressed_dir = args[1]
    links_extension_dir = args[2]

    process = psutil.Process(os_pid)
    max_memory = 0
    # When the files times_** exists, it means that the coding process finished
    while not os.path.exists(compressed_dir+"/times_"+links_extension_dir+".npy"):
        # Solution inspired by:          
        # https://stackoverflow.com/questions/30014295/how-to-get-the-percentage-of-memory-usage-of-a-process

        mem_perc = process.memory_percent() # Get total used memory
        mem_phys = psutil.virtual_memory().total # Get total physical memory (excluding swap)
        mem_raw = process.memory_info().rss # mem_raw/mem_phys == mem_perc
        if mem_raw>max_memory:
            max_memory = mem_raw

        time.sleep(1)
    
    np.save(compressed_dir+"/memory_usage_"+links_extension_dir+".npy", max_memory)

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
def model_inference(limit_size_cdf_hist, list_cdfs, window_no_label, model, x, mask_pred_0, mask_pred_1, y_label_feat, vals, y):
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
                
    yhat = model([tf.math.log(x), t_mask_pred_0[tf.newaxis,...], t_mask_pred_1[tf.newaxis,...], aux_feat_y[tf.newaxis,...], 
            bool_mask[tf.newaxis,...]])

    z = tfb.Exp()(yhat)

    qtz_distrib = tfd.QuantizedDistribution(z)
    prob_val = qtz_distrib.prob(vals[... , tf.newaxis])

    cdf_values = tf.cond(list_cdfs < limit_size_cdf_hist, lambda: yhat.cdf(tf.math.log(y)), lambda: tf.zeros(tf.shape(y)))
    return prob_val, cdf_values, yhat.parameters['distribution'].scale


# fp=open("../data/compressed/"+args.d+'/profile_logs_'+args.m+'.log','w+')
# @profile(stream=fp)
def main():
    #  python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 0 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
    dataset_name = args.d
    original_data_dir = "../data/files_to_be_compressed/tfrecords/"+dataset_name+"/"
    input_file_tfrecord = original_data_dir+dataset_name+'.tfrecords'
    compressed_dir = "../data/compressed/"+dataset_name+"/"

    if not os.path.exists("../data/Images/"):
        os.makedirs("../data/Images/")
    
    if not os.path.exists("../data/Images/"+dataset_name):
        os.makedirs("../data/Images/"+dataset_name)
    
    # Load ORIGINAL np array
    original = np.load(original_data_dir+"per_link_ORIGINAL_ts.npy")
    
    # Add extension to name according to prediction
    pred_flag = args.f
    pred_extension = "_masked_GNN"
    image_file_extension = args.m+pred_extension+ordered_compression
    links_extension_dir = "links_"+args.m+pred_extension+ordered_compression
    if pred_flag==1:
        pred_extension = "_overall_stats"
        image_file_extension = pred_extension
        links_extension_dir = "links"+pred_extension
    elif pred_flag==9:
        pred_extension = "_overall_stats_link"
        image_file_extension = pred_extension
        links_extension_dir = "links"+pred_extension
    elif pred_flag==10:
        pred_extension = "_adaptive_AC"
        image_file_extension = pred_extension
        links_extension_dir = "links"+pred_extension
    elif pred_flag==11:
        pred_extension = "_adaptive_AC_link"
        image_file_extension = pred_extension
        links_extension_dir = "links"+pred_extension

    if not os.path.exists(compressed_dir):
        os.makedirs(compressed_dir)
    # Clean old directory
    if args.r:
        os.system("rm -rf %s/*" % (compressed_dir))
    if not os.path.exists(compressed_dir+links_extension_dir):
        os.makedirs(compressed_dir+links_extension_dir)

    os.system("rm -rf %s" % (compressed_dir+"/times_"+links_extension_dir+".npy"))

    # Calling a monitoring process to track the memory usage
    argmts = []
    argmts.append((os.getpid(), compressed_dir, links_extension_dir))
    p1 = multiprocessing.Process(target=worker_execute, args=argmts)
    p1.start()

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
    scale_value = tf.cast(params["max_y"], tf.float32)
    delta = tf.cast(1, tf.float32)
    # +1 because we want to include the max_y-th element
    vals = np.arange(min_y,max_y+1, dtype=np.uint32)
    # -min_y because we shift to 0 the vals
    alphabet_size = int(max_y-min_y+1)

    # Load tf records
    experiment_params = inspect.getfullargspec(train_stgnn_network_wide.SpaceTimeGNN.__init__)
    e = train_stgnn_network_wide.SpaceTimeGNN(**{k: FLAGS[k].value for k in experiment_params[0][1:]})
    logging.info(e)
    checkpoint_filepath = os.path.join(e.logs_dir, 'checkpoint')

    # Load model weights
    e.class_model()

    if pred_flag==0:
        print("  RESTORING MODEL: ", checkpoint_filepath)
        e.model.load_weights(checkpoint_filepath).expect_partial()

    # Iterate over dataset
    ds = e.compress_ds()

    n_links = params["num_links"]
    window_no_label = params["window_size"]-1

    # In position 0 we store the model inference time and in position 1 we store the ac time
    times_inference_ac = np.zeros((params["num_samples"], 2))

    link_uti_hist = dict()
    if pred_flag==1:
        print("::::::")
        print("COMPUTING PROBABILITY DISTRIBUTION TAKING INTO ACCOUNT THE ENTIRE DATASET...")
        first_timestep = True

        prob = np.zeros(alphabet_size, dtype = np.float32)
        counter = 0
        total_link_uti_values = (params["num_samples"]+params["window_size"]-1)*n_links
        # Iterate over entire dataset and compute probability distribution for each link utilization
        for x,y in ds:
            if counter%1000==0:
                print("Computing probability timestep: ", counter)
            if first_timestep:
                # Iterate over timesteps
                for t in range(len(x[0])):
                    # Iterate over links
                    for l in range(len(x[0,t])):
                        link_uti_val = str(int(x[0,t,l,0]))
                        if link_uti_val in link_uti_hist:
                            link_uti_hist[link_uti_val] += 1
                        else:
                            link_uti_hist[link_uti_val] = 1

                first_timestep = False

            for l in range(n_links):
                link_uti_val = str(int(y[0,l]))
                if link_uti_val in link_uti_hist:
                    link_uti_hist[link_uti_val] += 1
                else:
                    link_uti_hist[link_uti_val] = 1

            counter += 1

        for key, value in link_uti_hist.items():
            prob[int(key)-min_y] = float(value/total_link_uti_values)
        
        del link_uti_hist
        gc.collect()
        print("Finished computing probability distribution")
        print("::::::")
        np.save(original_data_dir+"overall_statistics_probs.npy", prob)
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

    limit_size_cdf_hist = tf.cast(20000, tf.float32)
    # We encode the traffic for each link separately
    f_cmpr = [open(compressed_dir+links_extension_dir+"/link"+"_"+str(i)+'.compressed','wb') for i in range(n_links)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f_cmpr[i]) for i in range(n_links)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(arithm_statesize, bitout[i]) for i in range(n_links)]

    reconstructed_TS = np.zeros((n_links,params["num_samples"]+params["window_size"]-1), dtype = np.uint32)

    first_timestep = True
    cdf_printed = False
    cnt_timestep = 0

    # We store the times it takes to code a single time bin
    list_times = list()
    list_all_times = list()
    # Storing inference times of the model
    list_inf_times = list()
    # Storing coding times of the AC
    list_ac_times = list()

    # The following list is used for visualization purposes
    list_cdfs = list()
    list_cdfs_perf_last = list()
    list_scale = list()
    # Iterate over all samples and compress them
    # The dataset in this case is the real one without log transformation
    total_link_uti_values = 0
    # We iterate over the dataset to use the X values
    for x,y in ds:
        # tf.shape(x): tf.Tensor([batch_size time_steps num_links 1], shape=(4,), dtype=int32)
        start = tt.time()
        if first_timestep:
            total_link_uti_values = len(x[0])*len(x[0,0])
            # Encode the first time steps using uniform distribution. The last one will be encoded using 
            # the probabilities obtained from the model
            # Iterate over timesteps
            for t in range(len(x[0])):
                # Iterate over links
                for l in range(len(x[0,t])):
                    # Encode value using uniform probs
                    # we shift the link utilizations to go from 0 to max_uti
                    if pred_flag==9:
                        enc[l].write(cumul[l], int(x[0,t,l,0])-min_y)
                    else:
                        enc[l].write(cumul, int(x[0,t,l,0])-min_y)
                    reconstructed_TS[l, cnt_timestep] = int(x[0,t,l,0])
                cnt_timestep += 1

            first_timestep = False

        # Create new mask for predicted links. We store using one-hot encoding
        mask_pred_0 = np.zeros((n_links, 1))
        mask_pred_1 = np.zeros((n_links, 1))
        y_label_feat = np.zeros((n_links, 1))

        # We mark all links for prediction
        mask_pred_1.fill(1)
        list_tuple = list()

        # Iterate over all links and predict over them. Make the loop be incremental
        for link in range(n_links):
            link_to_compress = link
            # If we use the GNN
            if pred_flag==0:
                cumul.fill(0)

                inf_strt = tt.time()
                prob_val, cdf_values, scale = model_inference(limit_size_cdf_hist, tf.cast(len(list_cdfs), tf.float32), window_no_label, e.model, tf.cast(x, tf.float32), tf.cast(mask_pred_0, tf.float32), tf.cast(mask_pred_1, tf.float32), tf.cast(y_label_feat, tf.float32), tf.cast(vals, tf.float32), tf.cast(y, tf.float32))
                inf_end = tt.time()
                # Store the inference times for the GNN model
                list_inf_times.append(inf_end-inf_strt)
                # We use newaxis to give the batch dimension
                #yhat = e.model([tf.math.log(x), t_mask_pred_0[tf.newaxis,...], t_mask_pred_1[tf.newaxis,...], aux_feat_y[tf.newaxis,...], bool_mask[tf.newaxis,...]])
                if "ordered" in ordered_compression:
                    list_tuple.clear()

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
                            link_to_compress = tup[1]
                            break

                if len(list_cdfs)<limit_size_cdf_hist:
                    list_scale.append(scale[0,link_to_compress].numpy())
                    list_cdfs.append(cdf_values[0,link_to_compress].numpy())
                elif not cdf_printed:
                    plt.hist(list_cdfs, bins=50, edgecolor='black')
                    plt.grid(axis='y')
                    plt.ylabel("Count", fontsize=16)
                    plt.xlabel("y_dist.cdf(y)", fontsize=16)

                    plt.tight_layout()
                    plt.savefig("../data/Images/"+dataset_name+"/hist_cdf_"+image_file_extension+".pdf",bbox_inches='tight')
                    plt.close()

                    plt.hist(list_scale, bins=np.arange(0, 3, 0.05), color = "orange", edgecolor='black')
                    plt.grid(axis='y')
                    plt.ylabel("Count", fontsize=16)
                    plt.xlabel("Scale value", fontsize=16)

                    plt.tight_layout()
                    plt.savefig("../data/Images/"+dataset_name+"/hist_scale_"+image_file_extension+".pdf",bbox_inches='tight')
                    plt.close()

                    cdf_printed = True

                # Encode y value using the probs from the GNN
                cumul[1:] = np.cumsum(prob_val[:,link_to_compress]*10000000 + 1, axis = 0)
                # We use the real value as a feature
                y_label_feat[link_to_compress,0] = tf.math.log(y[0,link_to_compress])
                # We mark the links as known
                mask_pred_1[link_to_compress] = 0
                mask_pred_0[link_to_compress] = 1
            
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
                flattened_x = np.squeeze(np.concatenate(x[0,:,link_to_compress]))
                unique_elements, counts_elements = np.unique(flattened_x, return_counts=True)
                
                # Compute dynamic prob distribution
                for pos in range(len(unique_elements)):
                    prob[int(unique_elements[pos])-min_y] = float(counts_elements[pos]/(params["window_size"]-1))
                
                cumul[1:] = np.cumsum(prob*10000000 + 1, dtype=np.uint32)  
            if pred_flag==9:
                # Encode real value using probs
                # we shift the link utilizations to go from 0 to max_uti
                enc[link_to_compress].write(cumul[l], int(y[0,link_to_compress]-min_y))
                reconstructed_TS[link_to_compress, cnt_timestep] = int(y[0,link_to_compress])
            else:
                # Encode real value using probs
                # we shift the link utilizations to go from 0 to max_uti
                ac_strt = tt.time()
                enc[link_to_compress].write(cumul, int(y[0,link_to_compress]-min_y))
                ac_end = tt.time()
                
                # Storing ac coding times for the GNN model
                list_ac_times.append(ac_end-ac_strt)

                reconstructed_TS[link_to_compress, cnt_timestep] = int(y[0,link_to_compress])
                #break

        end = tt.time()
        list_times.append(end-start)
        list_all_times.append(end-start)

        # Storing mean times for model inference and ac coding phase
        pos_time = cnt_timestep-params["window_size"]+1
        times_inference_ac[pos_time,0] = np.mean(list_inf_times)
        times_inference_ac[pos_time,1] = np.mean(list_ac_times)
        list_inf_times.clear()
        list_ac_times.clear()

        if cnt_timestep%500==0:
            print("Encoded time bin " , image_file_extension, ": ", cnt_timestep, "/", params["num_samples"]+params["window_size"]-1)    
            print("  Mean Coding time per time bin (s): ", np.mean(list_times) )
            print("  Difference:", np.sum(np.abs(original[:,:cnt_timestep]-reconstructed_TS[:,:cnt_timestep])))
            list_times.clear()
            gc.collect()
        cnt_timestep += 1

    # Store the compression cost in time for each bin
    np.save(compressed_dir+"/times_"+links_extension_dir+".npy", np.asarray(list_all_times))

    # Store the model inference and AC coding times
    np.save(compressed_dir+"/inf_ac_times_"+links_extension_dir+".npy", times_inference_ac)

    # We only store the cdf image if we are using the SpaceTimeGNN
    if pred_flag==0 and not cdf_printed:
        plt.hist(list_cdfs, bins=50, edgecolor='black')
        plt.grid(axis='y')
        plt.ylabel("Count", fontsize=16)
        plt.xlabel("y_dist.cdf(y)", fontsize=16)

        plt.tight_layout()
        plt.savefig("../data/Images/"+dataset_name+"/hist_cdf_"+image_file_extension+".pdf",bbox_inches='tight')
        plt.close()

        plt.hist(list_scale, bins=50, color = "orange", edgecolor='black')
        plt.grid(axis='y')
        plt.ylabel("Count", fontsize=16)
        plt.xlabel("Scale value", fontsize=16)

        plt.tight_layout()
        plt.savefig("../data/Images/"+dataset_name+"/hist_scale_"+image_file_extension+".pdf",bbox_inches='tight')
        plt.close()

    # close files
    for i in range(n_links):
        enc[i].finish()
        bitout[i].close()
        f_cmpr[i].close()  
    print("Num total timesteps: ", cnt_timestep)
    print("Total coding time(in s): ", np.sum(list_all_times))
    print("Mean coding time per time bin(in s): ", np.mean(list_all_times))
    
    p1.join()

if __name__ == "__main__":
    main()

