import argparse
import os
import sys
import numpy as np
import gzip
import shutil
import matplotlib.pyplot as plt
import zipfile
import pickle
from statsmodels.tsa.stattools import adfuller

           
if __name__ == '__main__':
    # python tfrecords_print_compress_info.py -d tfrcrd_sin_shift_distrib_LRG -t 0
    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-d', help='directory where we want to check the file sizes')
    parser.add_argument('-t', help='Indicate if to perform Dickey-Fuller test (1) or not (0)', required=True, type=int)
    parser.add_argument('-m', help='Indicates the model to compute its file size from the logs', required=False, type=str)
    args = parser.parse_args()

    dataset_name = args.d
    model_file = args.m
    original_data_dir = "../data/files_to_be_compressed/tfrecords/"+dataset_name+"/"
    input_file_tfrecord = original_data_dir+dataset_name+'.tfrecords'
    compressed_dir = "../data/compressed/"+dataset_name+"/"
    dir_comp_gzip = compressed_dir+"links_gzip/"

    if not os.path.exists("../data/Images/"):
        os.makedirs("../data/Images/")

    if not os.path.exists("../data/Images/"+dataset_name):
        os.makedirs("../data/Images/"+dataset_name)

    # We store the file sizes in bytes
    compressed_file_masked_gnn_ordered = 0
    compressed_file_masked_gnn_no_order = 0
    compressed_file_overall_stats = 0
    compressed_file_overall_stats_link = 0
    compressed_file_gru = 0
    comp_ratio_gru_gnn_ordered_gzip = np.zeros(4) # 0==RNN, 1==GNN, 2==GZIP, 3==GZIP (link)

    times_gru = np.array([])
    times_gnn = np.array([])
    pearson_correlation = np.array([])

    number_stationary_ts = 0
    counter_total_time_series = 0
    # Iterate over files_to_be_compressed to extract pearson correlation and original tm series per link
    for filename in os.listdir(original_data_dir):
        if filename.startswith("pearson_"):
            with open(original_data_dir+filename, 'rb') as fp:
                pearson_correlation = pickle.load(fp) 
        elif filename.endswith("ORIGINAL_ts.npy"):
            # Load ORIGINAL np array
            original = np.load(original_data_dir+filename)
            if args.t==1:
                print("Performing Augmented Dickey-Fuller test on: ", original_data_dir+filename)
                for link in range(len(original)):
                    print("Dickey-Fuller on link: ", link, "/", len(original))
                    counter_total_time_series += 1
                    result = adfuller(original[link])
                    if result[1]<=0.05:
                        number_stationary_ts += 1

    param_file = 0
    reconstructed_file_masked_GNN = 0
    for filename in os.listdir(compressed_dir):
        file_to_get_size = compressed_dir+filename
        if filename.endswith("_masked_GNN_ordered") and "_pc" not in filename:
            # Iterate over the compressed link files
            for comprsd_file in os.listdir(compressed_dir+filename):
                if comprsd_file.startswith("link_"):
                    compressed_file_masked_gnn_ordered += os.path.getsize(file_to_get_size+"/"+comprsd_file)/1000
        elif filename.startswith("times_") and filename.endswith("_masked_GNN_ordered.npy") and "_pc" not in filename:
            # We read the times of the GNN
            times_gnn = np.load(compressed_dir+filename)
        elif filename.startswith("times_") and filename.endswith("_GRU.npy"):
            # We read the times of the GNN
            times_gru = np.load(compressed_dir+filename)
        elif filename.endswith("_masked_GNN_no_order"):
            # Iterate over the compressed link files
            for comprsd_file in os.listdir(compressed_dir+filename):
                if comprsd_file.startswith("link_"):
                    compressed_file_masked_gnn_no_order += os.path.getsize(file_to_get_size+"/"+comprsd_file)/1000
        elif filename.endswith("_overall_stats"):
            # Iterate over the compressed link files
            for comprsd_file in os.listdir(compressed_dir+filename):
                if comprsd_file.startswith("link_"):
                    compressed_file_overall_stats += os.path.getsize(file_to_get_size+"/"+comprsd_file)/1000
        elif filename.endswith("_overall_stats_link"):
            # Iterate over the compressed link files
            for comprsd_file in os.listdir(compressed_dir+filename):
                if comprsd_file.startswith("link_"):
                    compressed_file_overall_stats_link += os.path.getsize(file_to_get_size+"/"+comprsd_file)/1000
        elif filename.endswith("_params"):
            param_file = os.path.getsize(file_to_get_size)/1000
        elif filename.endswith("RECONSTRUCTED_ts_masked_GNN.npy"):
            reconstructed_file_masked_GNN = np.load(compressed_dir+filename)
        elif filename.endswith("_GRU"):
            # Iterate over the compressed link files
            for comprsd_file in os.listdir(compressed_dir+filename):
                if comprsd_file.startswith("link_"):
                    compressed_file_gru += os.path.getsize(file_to_get_size+"/"+comprsd_file)/1000
    
    ##################################################################
    ##### Boxplot Pearson Correlation
    pearson_correlation = pearson_correlation[~np.eye(pearson_correlation.shape[0],dtype=bool)].reshape(pearson_correlation.shape[0],-1).flatten()
    
    plt.ylabel('Pearson correlation\nbetween links', fontsize=12)
    plt.ylim((-1.0, 1.0))
    #plt.title("EVALUATION on "+topology_eval_name+" topology with "+str(len(uti_init_SP))+" TMs")
    bp = plt.boxplot(x=(pearson_correlation),showfliers=True, labels=[dataset_name], widths=(0.3))
    for box in bp['boxes']:
        box.set(linewidth=2)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=2)
    for cap in bp['caps']:
        cap.set(linewidth=2)
    for median in bp['medians']:
        median.set(linewidth=2)

    plt.tight_layout()
    plt.grid(color='gray', axis='y')
    #plt.show()
    plt.savefig("../data/Images/"+dataset_name+"/pearson_boxplot.pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

    if args.t==1:
        print(number_stationary_ts, counter_total_time_series)
        plt.grid(color='gray', axis='y')
        plt.ylabel('Percentage of stationary\ntime series (%)', fontsize=12)
        plt.bar(x=(dataset_name), height=(number_stationary_ts/counter_total_time_series)*100, edgecolor='black')
        #plt.show()
        plt.savefig("../data/Images/"+dataset_name+"/dickey_fuller_test.pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.clf()

    # Show times info
    if len(times_gru)>0:
        print("GRU:")
        print("Total coding time(in s): ", np.sum(times_gru))
        print("Mean coding time per time bin(in s): ", np.mean(times_gru))
    if len(times_gnn)>0:
        print("GNN:")
        print("Total coding time(in s): ", np.sum(times_gnn))
        print("Mean coding time per time bin(in s): ", np.mean(times_gnn))

    ##################################################################
    ######### GZIP

    # Compress original file with gzip
    # with open(original_data_dir+"per_link_ORIGINAL_ts.npy", 'rb') as f_in:
    #     with gzip.open(original_data_dir+"per_link_ORIGINAL_ts.npy.gz", 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)

    original_file = os.path.getsize(original_data_dir+"per_link_ORIGINAL_ts.npy")/1000
    np_original_file = np.load(original_data_dir+"per_link_ORIGINAL_ts.npy")

    if os.path.exists(dir_comp_gzip):
        os.system("rm -rf %s" % (dir_comp_gzip))
    os.makedirs(dir_comp_gzip)
    
    gzip_file_link_level = 0
    # Iterate over all links and store a file per link
    for l in range(len(np_original_file)):
        file_name = dir_comp_gzip+"link_"+str(l)+".npy.gz"
        f = gzip.GzipFile(file_name, "wb")
        np.save(f, np_original_file[l])
        f.close()
        
        file_size = os.path.getsize(file_name)/1000
        gzip_file_link_level += file_size

    gzip_file_global = 0
    f = gzip.GzipFile(compressed_dir+"gzip_entire_dataset", "wb")
    np.save(f, np_original_file)
    f.close()
    file_size = os.path.getsize(compressed_dir+"gzip_entire_dataset")/1000
    gzip_file_global += file_size

    ##################################################################

    print("Difference masked GNN:", np.sum(np.abs(np_original_file-reconstructed_file_masked_GNN)))

    print("****************")
    print("File sizes (in KBytes)")
    print("****************")
    print("ORIGINAL FILE: ", original_file)
    print("####")
    #print("Compressed files using GNN (ordered): ", compressed_file_masked_gnn_ordered)

    #print("Size after compression (GNN+Params_file+Original_file): ", total)
    print("Size after compression (GZIPed link-level): ", gzip_file_link_level)
    print("Size after compression (GZIPed): ", gzip_file_global)
    file_sizes = ['ORIGINAL FILE', 'GZIP link-level', 'GZIP']
    values_file_sizes = [original_file, gzip_file_link_level, gzip_file_global]

    if compressed_file_masked_gnn_ordered>0:
        file_sizes.append("GNN\n")
        values_file_sizes.append(compressed_file_masked_gnn_ordered)
        print("Size after compression (GNN ordered): ", compressed_file_masked_gnn_ordered)

    if compressed_file_masked_gnn_no_order>0:
        file_sizes.append("GNN NO order\n")
        values_file_sizes.append(compressed_file_masked_gnn_no_order)
        print("Size after compression (GNN NO order): ", compressed_file_masked_gnn_no_order)

    if compressed_file_gru>0:
        file_sizes.append("RNN\n")
        values_file_sizes.append(compressed_file_gru)
        print("Size after compression (RNN): ", compressed_file_gru)

    if compressed_file_overall_stats>0:
        file_sizes.append("Overall\nstatistics")
        values_file_sizes.append(compressed_file_overall_stats)
        print("Size after compression (Overall Stats): ", compressed_file_overall_stats)
    
    if compressed_file_overall_stats_link>0:
        file_sizes.append("Overall\nstatistics\nlink")
        values_file_sizes.append(compressed_file_overall_stats_link)
        print("Size after compression (Overall Stats Link): ", compressed_file_overall_stats_link)

    values_file_sizes = np.round(values_file_sizes, 5)
    bars = plt.bar(file_sizes,values_file_sizes)
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+0.1, yval + 1, yval)

    plt.ylabel("File size (in KBytes)", fontsize=13)
    # plt.xticks(
    #     rotation=20, 
    #     horizontalalignment='right',
    #     fontweight='light',
    #     fontsize=10  
    # )
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("../data/Images/"+dataset_name+"/comp_sizes.pdf",bbox_inches='tight')
    plt.clf()

    # Rotation of the bars names
    # plt.show()
    
    print("####")
    print("Compression ratio GZIP link-level: ", original_file/gzip_file_link_level)
    print("Compression ratio GZIP: ", original_file/gzip_file_global)
    file_ratios = ['GZIP link-level', 'GZIP']
    values_ratios = [original_file/gzip_file_link_level, original_file/gzip_file_global]

    comp_ratio_gru_gnn_ordered_gzip[2] = original_file/gzip_file_global
    comp_ratio_gru_gnn_ordered_gzip[3] = original_file/gzip_file_link_level

    if compressed_file_masked_gnn_ordered>0:
        file_ratios.append("GNN\n")
        values_ratios.append(original_file/compressed_file_masked_gnn_ordered)
        print("Compression ratio GNN ordered: ", original_file/compressed_file_masked_gnn_ordered)
        comp_ratio_gru_gnn_ordered_gzip[1] = original_file/compressed_file_masked_gnn_ordered
    
    if compressed_file_masked_gnn_no_order>0:
        file_ratios.append("GNN NO order\n")
        values_ratios.append(original_file/compressed_file_masked_gnn_no_order)
        print("Compression ratio GNN NO order: ", original_file/compressed_file_masked_gnn_no_order)

    if compressed_file_gru>0:
        file_ratios.append("RNN\n")
        values_ratios.append(original_file/compressed_file_gru)
        print("Compression ratio RNN: ", original_file/compressed_file_gru)
        comp_ratio_gru_gnn_ordered_gzip[0] = original_file/compressed_file_gru

    if compressed_file_overall_stats>0:
        file_ratios.append("Overall\nstatistics")
        values_ratios.append(original_file/compressed_file_overall_stats)
        print("Compression ratio (Overall Stats): ", original_file/compressed_file_overall_stats)
    
    if compressed_file_overall_stats_link>0:
        file_ratios.append("Overall\nstatistics\nlink")
        values_ratios.append(original_file/compressed_file_overall_stats_link)
        print("Compression ratio (Overall Stats Link): ", original_file/compressed_file_overall_stats_link)
 
    with open(compressed_dir+"compression_ratios.pkl", 'wb') as fp:
        pickle.dump(comp_ratio_gru_gnn_ordered_gzip, fp, protocol=pickle.HIGHEST_PROTOCOL)

    values_ratios = np.round(values_ratios, 5)
    bars = plt.bar(file_ratios,values_ratios, color="tab:red")
    print("\nGNN ratio of compression ratio (w.r.t. GZIP):", comp_ratio_gru_gnn_ordered_gzip[1]/comp_ratio_gru_gnn_ordered_gzip[2])

    if isinstance(model_file, str):
        # Compute model file size (of only the weights)
        model_weights_size = os.path.getsize("../src/log/"+args.m+"/checkpoint.data-00000-of-00001")/1000
        print("Model weights size (in Kbytes):", model_weights_size)
    
    # access the bar attributes to place the text in the appropriate location
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+0.2, yval + .075, yval)
    
    plt.ylabel("Compression ratio", fontsize=13)
    plt.ylim((0.0, 7.2))
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("../data/Images/"+dataset_name+"/comp_ratios.pdf", bbox_inches='tight')
    plt.clf()