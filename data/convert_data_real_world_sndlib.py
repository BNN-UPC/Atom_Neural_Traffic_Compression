import argparse
import os
import sys
import matplotlib.pyplot as plt
import json
import pandas as pd
from utils import *
from scipy.stats import pearsonr
from functools import partial
import pickle

# This script reads the csv created in script_conv_topologySNDLIB.py and converts it to tfrecords

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(7)
data_dir="./files_to_be_compressed/tfrecords/"


# Window size (including label)
window_size = 5
MAX_NUM_TIMESTEPS = 150000

def map_fn(x):
    return tf.split(x, axis=0, num_or_size_splits=[-1, 1])

@tf.function
def any_nan(x):
    return tf.reduce_any(tf.math.is_nan(x))

def filter_fn(*x):
    nans = tf.concat([any_nan(a) for a in tf.nest.flatten(*x)], axis=0)
    return tf.logical_not(tf.reduce_any(nans))

if __name__ == '__main__':
    # python convert_data_real_world_sndlib.py -r 1 -f geant
    # python convert_data_real_world_sndlib.py -r 1 -f abilene
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-r', help='Number of different mask repeats per time step', required=True, type=int)
    parser.add_argument('-f', help='Filename of the dataset to process: [geant|abilene]', required=True)
    args = parser.parse_args()
    file_name = args.f

    file_name = file_name + "_w" + str(window_size)

    print("+++++++++++++++++++++++++++++++")
    print("GENERATING ", file_name," TFRECORDS DATASET")
    print("+++++++++++++++++++++++++++++++")

    num_repetitions = args.r
    file_name += "_r"+str(num_repetitions)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if file_name.startswith("geant"):
        topology = nx.read_gml("./datasets/Geant/Geant_topology.gml")
        df = pd.read_csv('./datasets/Geant/geant_dataset.csv',  index_col='timestamp')  
    else:
        topology = nx.read_gml("./datasets/Abilene/Abilene_topology.gml")
        df = pd.read_csv('./datasets/Abilene/abilene_dataset.csv',  index_col='timestamp')  

    make_ds = partial(
        tf.keras.preprocessing.timeseries_dataset_from_array,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=1,
    )
    data = np.array(df.loc[:, :], dtype=np.float64)
    if "geant" in file_name:
        # Those rows that have at least one zero are removed
        data = data[np.all(data != 0, axis=1)] # KBytes/15min
        data = np.around(data)
    elif "abilene" in file_name:
        data = np.around((data * 100)) # Bytes/5min
    num_rows = len(data)
    
    # We store the position of the link that has always 0 link utilization
    list_zero_link_uti = dict()

    dataset = make_ds(data)
    # The filter_fn checks for Nans
    dataset = dataset.filter(filter_fn)
    # Remove those sequences where there is a value with 0
    if "geant" not in file_name:
        dataset = dataset.filter(lambda x: tf.reduce_all(x > 0))
    dataset = dataset.unbatch()
    dataset = dataset.map(lambda x: map_fn(x))
    dataset = dataset.map(lambda x, y: (x[..., tf.newaxis], tf.squeeze(y)))

    numEdges = len(topology.edges())

    # Create new mask for predicted links. We store using one-hot encoding
    mask_pred_one_hot = np.zeros((numEdges, 2))
    mask_pred = np.zeros((numEdges)) # We mark with 1 if the link is flagged or not with 0. 

    # Small loop to count the number of samples
    cnt_timestep = 0
    for elem in dataset:
        cnt_timestep += 1
        if cnt_timestep==MAX_NUM_TIMESTEPS:
            break

    if cnt_timestep<MAX_NUM_TIMESTEPS:
        MAX_NUM_TIMESTEPS = cnt_timestep
    
    file_name = file_name + "_maxt" + str(MAX_NUM_TIMESTEPS)

    if not os.path.exists(data_dir+file_name):
        os.makedirs(data_dir+file_name)
    else:
        os.system("rm -rf %s" % (data_dir+file_name))
        os.makedirs(data_dir+file_name)
    
    if not os.path.exists("./Images/"):
        os.makedirs("./Images/")

    if not os.path.exists("./Images/"+file_name):
        os.makedirs("./Images/"+file_name)

    # if not os.path.exists("./Images/"+file_name+"/LINK_PLOTS"):
    #     os.makedirs("./Images/"+file_name+"/LINK_PLOTS")

    tfrecords_file = data_dir+file_name+"/"+file_name+'.tfrecords'
    tfrecords_file_masked = data_dir+file_name+"/"+file_name+'_masked.tfrecords'
    
    writer = tf.io.TFRecordWriter(tfrecords_file)
    writer_masked = tf.io.TFRecordWriter(tfrecords_file_masked)

    # Here we store the evolution of the link states. It will be used to compare the compressor with the original file
    # We add +window_size-1 to include the values of the first sample
    link_utis_TS_evol_real = np.zeros((numEdges,MAX_NUM_TIMESTEPS), dtype = np.uint32)
    link_utis_TS_evol_log = np.zeros((numEdges,MAX_NUM_TIMESTEPS), dtype = np.float32)

    # Here we store the differences. It represents a time series of differences
    link_diff_utis_TS_real = np.zeros((numEdges,MAX_NUM_TIMESTEPS), dtype = np.int32)
    link_diff_utis_TS_log = np.zeros((numEdges,MAX_NUM_TIMESTEPS), dtype = np.float32)

    # We store the masks to avoid repetitions
    list_masks = list()

    cnt_timestep = 0
    sample_count = 0
    sample_count_masked = 0
    # Iterate over whole dataset and write the training/validation data splits to tfrecords
    for elem in dataset:

        if cnt_timestep==MAX_NUM_TIMESTEPS:
            break

        # If its the first sample, we add the first elements from the
        # time window to the link_utis_TS_evol_log
        if cnt_timestep==0:
            for sample_link_uti in elem[0]:
                # Iterate over all links and extract the utilization
                for pos in range(numEdges):
                    real_link_uti = sample_link_uti.numpy()[pos,0] #np.around(sample_link_uti.numpy()[pos,0], 0)
                    if "geant" in file_name:
                        real_link_uti = real_link_uti+1 # We shift all the values +1 because some links have very little traffic
                    log_link_uti = np.log(real_link_uti)

                    link_utis_TS_evol_real[pos, cnt_timestep] = real_link_uti
                    link_utis_TS_evol_log[pos,cnt_timestep] = log_link_uti
                    # Store the differences of link utilizations
                    if cnt_timestep==0:
                        link_diff_utis_TS_real[pos, cnt_timestep] = real_link_uti
                        link_diff_utis_TS_log[pos, cnt_timestep] = log_link_uti
                    else:
                        # Store the difference w.r.t. the last step with log transform and without
                        link_diff_utis_TS_real[pos, cnt_timestep] = real_link_uti-link_utis_TS_evol_real[pos, cnt_timestep-1]
                        link_diff_utis_TS_log[pos, cnt_timestep] = log_link_uti-link_utis_TS_evol_log[pos, cnt_timestep-1]

                cnt_timestep += 1

        list_masks.clear()

        # We create as many masks as indicated by the flag
        for mask_rep in range(num_repetitions):
            mask_pred.fill(0)
            mask_pred_one_hot.fill(0)

            # +1 is to include full mask of all 1s
            num_flagged_links = np.random.randint(0, numEdges+1)
            mask_pred[:num_flagged_links] = 1
            np.random.shuffle(mask_pred)
            mask_pred_string = np.array2string(mask_pred)

            # Repeate the process until we create a sample with non repeated mask
            while mask_pred_string in list_masks:
                mask_pred.fill(0)
                # print(cnt_timestep, mask_rep, mask_pred_string)
                # +1 is to include full mask of all 1s
                num_flagged_links = np.random.randint(0, numEdges+1)
                mask_pred[:num_flagged_links] = 1
                np.random.shuffle(mask_pred)
                mask_pred_string = np.array2string(mask_pred)

            list_masks.append(mask_pred_string)

            # If it's the first mask, we store the link evolution
            if mask_rep==0:
                # Iterate over all links from the label and extract the utilization
                for pos in range(numEdges):
                    real_link_uti = elem[1].numpy()[pos] #np.around(sample_link_uti.numpy()[pos,0], 0)
                    if "geant" in file_name:
                        real_link_uti = real_link_uti+1 # We shift all the values +1 because some links have very little traffic
                    log_link_uti = np.log(real_link_uti)

                    link_utis_TS_evol_log[pos, cnt_timestep] = log_link_uti
                    link_utis_TS_evol_real[pos, cnt_timestep] = real_link_uti

                    # Store the difference w.r.t. the last step with log transform and without
                    link_diff_utis_TS_real[pos, cnt_timestep] = real_link_uti-link_utis_TS_evol_real[pos, cnt_timestep-1]
                    link_diff_utis_TS_log[pos, cnt_timestep] = log_link_uti-link_utis_TS_evol_log[pos, cnt_timestep-1]

            if "geant" in file_name:
                y_label_feat = np.log(np.asarray(elem[1].numpy()+1, dtype=np.float32)) # We shift all the values +1 because some links have very little traffic
            else:
                y_label_feat = np.log(np.asarray(elem[1].numpy(), dtype=np.float32))
            
            # the corresponding non zero positions for links over which we will predict
            masked_links_pred = np.nonzero(mask_pred)[0]
            # We mark that we know all links
            mask_pred_one_hot[:,0] = 1

            # Masked links for prediction have the feature set to 0
            for aux_l in masked_links_pred:
                y_label_feat[aux_l] = 0
                # We unmark links
                mask_pred_one_hot[aux_l,1] = 1
                mask_pred_one_hot[aux_l,0] = 0

            # If it's the first mask, we store the link evolution in tfrecords
            if mask_rep==0:
                if "geant" in file_name:
                    # Write the sample without the masks, just the link features and the label
                    write_flat_sample_real_tfrecords(writer, elem[0]+1, elem[1]+1) # +1 to avoid log(0) in the GRU/GNN training and compression
                else:
                    write_flat_sample_real_tfrecords(writer,elem[0], elem[1])

            # Write the sample with the log, the mask and the features
            if "geant" in file_name:
                write_sample_tfrecords_flagged_links_masked(writer_masked, np.log(elem[0]+1), mask_pred_one_hot, np.log(elem[1]+1), y_label_feat) # We shift all the values +1 because some links have very little traffic
            else:
                write_sample_tfrecords_flagged_links_masked(writer_masked, np.log(elem[0]), mask_pred_one_hot, np.log(elem[1]), y_label_feat)
            sample_count_masked += 1

        if cnt_timestep%2000==0:
            print("Sample written #", cnt_timestep, " from ", MAX_NUM_TIMESTEPS)
        cnt_timestep += 1
        sample_count += 1

    #####################################
    ## VISUALIZATION PURPOSE ONLY

    # for col in range(numEdges):
    #     labels, counts = np.unique(data[:MAX_NUM_TIMESTEPS,col], return_counts=True)
    #     #plt.hist(counts, bins=np.arange(5)-0.2, edgecolor='black', log=True)
    #     plt.scatter(labels,counts)
    #     plt.yscale('log')

    #     plt.grid(axis='y')
    #     plt.ylabel("Count", fontsize=14)
    #     plt.xlabel("Column "+str(col)+" Utilizaion Values", fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig("./Images/"+file_name+"/LINK_PLOTS/column_"+str(col)+".pdf",bbox_inches='tight')
    #     plt.close()

    #     # Plot time series
    #     plt.rcParams["figure.figsize"] = (4.5,3.6)
    #     plt.plot([i for i in range(len(data[:MAX_NUM_TIMESTEPS,col]))], data[:MAX_NUM_TIMESTEPS,col], color="tab:blue")
    #     plt.yscale('log')

    #     plt.grid(axis='y', which="both")
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.ylabel("Bytes/5min", fontsize=12)
    #     plt.xlabel("Time bins (5 min)", fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig("./Images/"+file_name+"/LINK_PLOTS/TS_link_"+str(col)+".png",bbox_inches='tight', pad_inches = 0.01)
    #     #plt.show()
    #     plt.close()

    # We create the correlation matrix
    correlation_matrix = np.zeros((numEdges, numEdges))

    tick_values = list()
    # For each link, we compute the combination with all other links
    for i in range(numEdges):
        #tick_values.append(df.columns[i])
        tick_values.append(i)
        for j in range(numEdges):
            corr, _ = pearsonr(link_utis_TS_evol_log[i, :], link_utis_TS_evol_log[j, :])
            correlation_matrix[i, j] = corr

    with open(data_dir+file_name+"/pearson_correlation.pkl", 'wb') as fp:
        pickle.dump(correlation_matrix, fp, protocol=pickle.HIGHEST_PROTOCOL)

    plt.rcParams["figure.figsize"] = (4.5,3.6)
    fig, ax = plt.subplots()
    im = ax.imshow(correlation_matrix, cmap="seismic")
    im.set_clim(-1, 1)
    # ax.set_xticks(np.arange(len(tick_values)))
    # ax.set_yticks(np.arange(len(tick_values)))
    # ax.set_xticklabels(tick_values)
    # ax.set_yticklabels(tick_values)
    # plt.xticks(
    #     rotation=90, 
    #     horizontalalignment='left',
    # )
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    cbar.ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
    plt.tight_layout()
    plt.ylabel("Link identifier", fontsize=12)
    plt.xlabel("Link identifier", fontsize=12)
    #plt.show()
    plt.savefig("./Images/"+file_name+"/pearson_correlation.pdf",bbox_inches='tight',pad_inches = 0)
    plt.close()

    #####################################

    min_y = np.amin(link_utis_TS_evol_real) 
    max_y = np.amax(link_utis_TS_evol_real)

    print("**********************************")
    print("Total number of samples: ", sample_count)
    print("Total number of MASKED samples: ", sample_count_masked)
    print("Window size(with label): ", window_size)
    print("MIN and MAX Uti Values(before log): ", int(min_y), int(max_y))
    # Print diff historial BEFORE applying log
    plt.hist(link_diff_utis_TS_real.flatten(), bins=200, edgecolor='black', log=True)
    plt.grid(axis='y')
    plt.ylabel("Count BEFORE", fontsize=16)
    plt.xlabel("hist(data[t+1,col]-data[t,col])", fontsize=16)
    #plt.show()
    plt.tight_layout()
    plt.savefig("./Images/"+file_name+"/hist_diff_BEFORE_log.pdf",bbox_inches='tight')
    plt.close()

    # Print diff historial AFTER applying log
    plt.hist(link_diff_utis_TS_log.flatten(), bins=200, edgecolor='black', log=True)
    plt.grid(axis='y')
    plt.ylabel("Count AFTER", fontsize=16)
    plt.xlabel("hist(data[t+1,col]-data[t,col])", fontsize=16)
    #plt.show()
    plt.tight_layout()
    plt.savefig("./Images/"+file_name+"/hist_diff_AFTER_log.pdf",bbox_inches='tight')
    plt.close()

    min_diff = np.amin(link_diff_utis_TS_real)
    max_diff = np.amax(link_diff_utis_TS_real)
    print("MIN and MAX Diff Uti Values(before log/real): ", min_diff, max_diff)

    np.save(data_dir+file_name+"/per_link_ORIGINAL_ts.npy", link_utis_TS_evol_real)
    if file_name.startswith("geant"):
        nx.write_gml(topology, data_dir+file_name+"/Geant_topology.gml")
    else:
        nx.write_gml(topology, data_dir+file_name+"/Abilene_topology.gml")

    with open(data_dir+file_name+"/links_to_remove.pkl", 'wb') as fp:
        pickle.dump(list_zero_link_uti, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # Store general information
    params = dict()
    params["window_size"] = window_size
    params["num_samples"] = sample_count
    params["num_masked_samples"] = sample_count_masked
    params["min_y"] = int(min_y)
    params["max_y"] = int(max_y)
    params["num_links"] = int(numEdges)
    with open(data_dir+file_name+"/"+file_name+'_params', 'w') as f:
        json.dump(params, f, indent=4)