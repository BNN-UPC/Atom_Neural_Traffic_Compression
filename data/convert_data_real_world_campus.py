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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(7)
data_dir="./files_to_be_compressed/tfrecords/"

# Inficate if we work with KBytes units or raw dataset
flag_Kb = True
file_name = "agh"
if flag_Kb:
    file_name = "KB_agh"

# Window size (including label)
window_size = 5
file_name = file_name + "_w" + str(window_size)

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
    # python convert_data_real_world_campus.py -r 1
    print("+++++++++++++++++++++++++++++++")
    print("GENERATING ", file_name," TFRECORDS DATASET")
    print("+++++++++++++++++++++++++++++++")

    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-r', help='Number of different mask repeats per time step', required=True, type=int)
    args = parser.parse_args()

    num_repetitions = args.r
    file_name += "_r"+str(num_repetitions)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    A = AGH.adjacency()
    df = pd.read_csv('./datasets/stats_clean.csv',  index_col='timestamp')  
    columns = ['-'.join([n.replace('rtr', '') for n in x.split('-')]) for x in AGH._NAMES]

    make_ds = partial(
        tf.keras.preprocessing.timeseries_dataset_from_array,
        targets=None,
        sequence_length=window_size,
        sequence_stride=1,
        shuffle=False,
        batch_size=1,
    )
    data = np.array(df.loc[:, AGH._NAMES], dtype=np.float64)

    # We use the new_df to track the indices/dates to store the date per sample. I'll need the dates for the figures
    new_df = df[AGH._NAMES].copy()
    new_df = np.ceil(new_df/ 1e3)

    # data_KB =  np.around(data / 1e2)
    data_KB =  np.around(data / 1e3) # KByte/5min
    # data = 8. * data / 1e8  # 100 Mb/s
    num_rows = len(data)
    # Divide the entire data by median and apply np.log to all dataset. Remove normalization layer If I want to. Krzysztos doesn't have it and it works well 
    if not flag_Kb:
        # If we apply this division when working with Kb, we have many zeros
        median = np.nanmedian(data[:int(num_rows*0.7),:], axis=0)
        data = data/median

    # We store the position of the link that has always 0 link utilization
    list_zero_link_uti = dict()
    num_removed = 0
    for col in range(len(data_KB[0])):
        pos_col_zero = np.where(data_KB[:,col-num_removed] == 0)[0]
        # We delete the whole column if more than 95% of time steps are at 0
        if len(pos_col_zero)>int(num_rows*0.9):
            print("DELETING COLUMN: ", col-num_removed, columns[col-num_removed], num_rows, len(pos_col_zero))
            list_zero_link_uti[str(col)] = columns[int(col)-num_removed]
            data = np.delete(data, int(col)-num_removed, 1)
            data_KB = np.delete(data_KB, int(col)-num_removed, 1)
            new_df.drop(new_df.columns[int(col)-num_removed],axis=1,inplace=True)
            A = np.delete(A, int(col)-num_removed, 1)
            del columns[int(col)-num_removed]
            num_removed += 1

    dataset = make_ds(data)
    # The filter_fn checks for Nans
    dataset = dataset.filter(filter_fn)
    # Remove those sequences where there is a value with 0
    dataset = dataset.filter(lambda x: tf.reduce_all(x > 0))
    dataset = dataset.unbatch()
    dataset = dataset.map(lambda x: map_fn(x))
    if flag_Kb:
        # Convert everything to KBytes
        # dataset = dataset.map(lambda x, y: (tf.math.ceil(x/1e2), tf.math.ceil(y/1e2)))
        dataset = dataset.map(lambda x, y: (tf.math.ceil(x/1e3), tf.math.ceil(y/1e3)))
    dataset = dataset.map(lambda x, y: (x[..., tf.newaxis], tf.squeeze(y)))

    numEdges = len(A[0])

    # We create the correlation matrix
    correlation_matrix = np.zeros((numEdges, numEdges))

    # Create new mask for predicted links. We store using one-hot encoding
    mask_pred_one_hot = np.zeros((numEdges, 2))
    mask_pred = np.zeros((numEdges)) # We mark with 1 if the link is flagged or not with 0. 
    tick_values = list() # Used in the correlation matrix

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
    # Iterate over whole dataset and write the training set to tfrecords
    for elem in dataset:

        # The idea is to store a replica of the exact data we use during training/compression to later 
        # extract the timestamps for the figures of compression evolution for different percentages
        if cnt_timestep==0:
            for row_x in elem[0]:
                matching_rows = new_df.apply(lambda row: np.array_equal(row.values, tf.squeeze(row_x).numpy()), axis=1)
                if not os.path.isfile(data_dir+file_name+"/dataframe_with_timestamps.csv"):
                    new_df[matching_rows].to_csv(data_dir+file_name+"/dataframe_with_timestamps.csv")
                else:
                    new_df[matching_rows].to_csv(data_dir+file_name+"/dataframe_with_timestamps.csv", mode='a', header=False)

        matching_rows = new_df.apply(lambda row: np.array_equal(row.values, tf.squeeze(elem[1]).numpy()), axis=1)
        new_df[matching_rows].to_csv(data_dir+file_name+"/dataframe_with_timestamps.csv", mode='a', header=False)
 
        if cnt_timestep==MAX_NUM_TIMESTEPS:
            break

        # If its the first sample, we add the first elements from the
        # time window to the link_utis_TS_evol_log
        if cnt_timestep==0:
            for sample_link_uti in elem[0]:
                # Iterate over all links and extract the utilization
                for pos in range(numEdges):
                    real_link_uti = sample_link_uti.numpy()[pos,0] #np.around(sample_link_uti.numpy()[pos,0], 0)
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
            if mask_rep==0:
                # Iterate over all links from the label and extract the utilization
                for pos in range(numEdges):
                    real_link_uti = elem[1].numpy()[pos] #np.around(sample_link_uti.numpy()[pos,0], 0)
                    log_link_uti = np.log(real_link_uti)

                    link_utis_TS_evol_log[pos, cnt_timestep] = log_link_uti
                    link_utis_TS_evol_real[pos, cnt_timestep] = real_link_uti

                    # Store the difference w.r.t. the last step with log transform and without
                    link_diff_utis_TS_real[pos, cnt_timestep] = real_link_uti-link_utis_TS_evol_real[pos, cnt_timestep-1]
                    link_diff_utis_TS_log[pos, cnt_timestep] = log_link_uti-link_utis_TS_evol_log[pos, cnt_timestep-1]

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

            if mask_rep==0:
                # Write the sample without the masks, just the link features and the label
                write_flat_sample_real_tfrecords(writer, elem[0], elem[1])
            
            # Write the sample with the log, the mask and the features
            write_sample_tfrecords_flagged_links_masked(writer_masked, np.log(elem[0]), mask_pred_one_hot, np.log(elem[1]), y_label_feat)
        
            sample_count_masked += 1

        #####################################
        ## VISUALIZATION PURPOSE ONLY
        # plt.plot([i for i in range(len(elem[0][:,6,0]))], elem[0][:,6,0], color="green")
        # plt.yscale('log')

        # plt.grid(axis='y')
        # plt.ylabel("KByte/s", fontsize=14)
        # plt.xlabel("Time", fontsize=14)
        # plt.tight_layout()
        # plt.show()

        # correlation_matrix.fill(0)
        # tick_values.clear()
        # # For each link, we compute the combination with all other links
        # for i in range(numEdges):
        #     tick_values.append(columns[i])
        #     for j in range(numEdges):
        #         corr, _ = pearsonr(elem[0][:,i,0], elem[0][:,j,0])
        #         correlation_matrix[i, j] = corr

        # fig, ax = plt.subplots()
        # im = ax.imshow(correlation_matrix, cmap="seismic")
        # im.set_clim(-1, 1)
        # ax.set_xticks(np.arange(len(tick_values)))
        # ax.set_yticks(np.arange(len(tick_values)))
        # ax.set_xticklabels(tick_values)
        # ax.set_yticklabels(tick_values)
        # plt.xticks(
        #     rotation=90, 
        #     horizontalalignment='left',
        # )
        # ax.grid(False)
        # cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
        # plt.tight_layout()
        # plt.show()
        # plt.close()
        #####################################

        if cnt_timestep%2000==0:
            print("Sample written #", cnt_timestep, " from ", MAX_NUM_TIMESTEPS)
        cnt_timestep += 1
        sample_count += 1

    #####################################
    ## VISUALIZATION PURPOSE ONLY

    # for col in range(numEdges):
    #     labels, counts = np.unique(link_utis_TS_evol_real[col,:MAX_NUM_TIMESTEPS], return_counts=True)
    #     #plt.hist(counts, bins=np.arange(20)-0.5, edgecolor='black', log=True)
    #     # plt.hist(counts, bins=100, edgecolor='black', log=True)
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
    #     plt.plot([i for i in range(len(link_utis_TS_evol_real[col,:MAX_NUM_TIMESTEPS]))], link_utis_TS_evol_real[col,:MAX_NUM_TIMESTEPS], color="tab:blue")
    #     plt.yscale('log')

    #     plt.grid(axis='y', which="both")
    #     plt.xticks(fontsize=12)
    #     plt.yticks(fontsize=12)
    #     plt.ylabel("KBytes/5min", fontsize=12)
    #     plt.xlabel("Time bins (5 min)", fontsize=12)
    #     plt.tight_layout()
    #     plt.savefig("./Images/"+file_name+"/LINK_PLOTS/TS_link_"+str(col)+".png",bbox_inches='tight', pad_inches = 0.01)
    #     #plt.show()
    #     plt.close()

    # tick_values.clear()
    # correlation_matrix.fill(0)
    # # For each link, we compute the combination with all other links
    # for i in range(numEdges):
    #     #tick_values.append(columns[i])
    #     tick_values.append(i)
    #     for j in range(numEdges):
    #         corr, _ = pearsonr(link_utis_TS_evol_log[i, :], link_utis_TS_evol_log[j, :])
    #         correlation_matrix[i, j] = corr

    # with open(data_dir+file_name+"/pearson_correlation.pkl", 'wb') as fp:
    #     pickle.dump(correlation_matrix, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # fig, ax = plt.subplots()
    # im = ax.imshow(correlation_matrix, cmap="seismic")
    # im.set_clim(-1, 1)
    # ax.set_xticks(np.arange(len(tick_values)))
    # ax.set_yticks(np.arange(len(tick_values)))
    # ax.set_xticklabels(tick_values)
    # ax.set_yticklabels(tick_values)
    # # plt.xticks(
    # #     rotation=90, 
    # #     horizontalalignment='left',
    # # )
    # ax.grid(False)
    # cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    # cbar.ax.set_ylabel('Pearson Correlation Coefficient')
    # plt.tight_layout()
    # plt.ylabel("Link identifier", fontsize=14)
    # plt.xlabel("Link identifier", fontsize=14)
    # #plt.show()
    # plt.savefig("./Images/"+file_name+"/pearson_correlation.pdf",bbox_inches='tight', pad_inches = 0)
    # plt.close()

    #####################################

    min_y = np.amin(link_utis_TS_evol_real) #np.around(np.amin(elem[0]), 0)
    max_y = np.amax(link_utis_TS_evol_real) #np.around(np.amax(elem[0]), 0)

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