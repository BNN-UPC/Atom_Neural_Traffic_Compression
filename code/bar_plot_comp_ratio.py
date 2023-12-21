import argparse
import os
import sys
import numpy as np
import gzip
import shutil
import matplotlib.pyplot as plt
import zipfile
import pickle
import seaborn as sns
from itertools import cycle
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller

# This script is to compute the barplots of the compression ratios for all three real-world datasets
# Before executing this script, I need to execute the script tfrecords_print_compress_info.py!!!!!
# Have a look at section "#### Masked links (Paper Experiment)" from the readme

if __name__ == '__main__':
    # python bar_plot_comp_ratio.py
    list_datasets = ["abilene_w5_r40_maxt41741", "geant_w5_r40_maxt6063", "KB_agh_w5_r20_maxt102799"]
    compressed_dir = "../data/compressed/"

    pandas_df = pd.DataFrame(columns=['Compression ratio', 'Baseline', 'Dataset'])


    for dataset in list_datasets:
        print("Reading dataset ", dataset)

        original_file = os.path.getsize("../data/files_to_be_compressed/tfrecords/"+dataset+"/per_link_ORIGINAL_ts.npy")/1000
        gzip_file_global = 0
        compressed_file_masked_gnn_ordered = 0
        compressed_file_overall_stats = 0
        compressed_file_adaptive_ac = 0

        for filename in os.listdir(compressed_dir+dataset):
            if filename.endswith("gzip_entire_dataset"):
                file_size = os.path.getsize(compressed_dir+dataset+"/"+filename)/1000
                gzip_file_global += file_size
            elif filename.endswith("_overall_stats"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        compressed_file_overall_stats += os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
            elif filename.endswith("_masked_GNN_ordered") and "_pc" not in filename:
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        compressed_file_masked_gnn_ordered += os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
            elif filename.endswith("_adaptive_AC"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        compressed_file_adaptive_ac += os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000

        if "geant" in dataset:
            new_row = {"Compression ratio": float(original_file/compressed_file_masked_gnn_ordered), "Baseline": "Atom (ST-GNN)", "Dataset": "Geant"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/gzip_file_global), "Baseline": "GZIP", "Dataset": "Geant"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_overall_stats), "Baseline": "Static AC", "Dataset": "Geant"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_adaptive_ac), "Baseline": "Adaptive AC", "Dataset": "Geant"}
            pandas_df.loc[len(pandas_df.index)] = new_row
        elif "KB_agh" in dataset:
            new_row = {"Compression ratio": float(original_file/compressed_file_masked_gnn_ordered), "Baseline": "Atom (ST-GNN)", "Dataset": "Campus network"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/gzip_file_global), "Baseline": "GZIP", "Dataset": "Campus network"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_overall_stats), "Baseline": "Static AC", "Dataset": "Campus network"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_adaptive_ac), "Baseline": "Adaptive AC", "Dataset": "Campus network"}
            pandas_df.loc[len(pandas_df.index)] = new_row
        elif "abilene" in dataset:
            new_row = {"Compression ratio": float(original_file/compressed_file_masked_gnn_ordered), "Baseline": "Atom (ST-GNN)", "Dataset": "Abilene"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/gzip_file_global), "Baseline": "GZIP", "Dataset": "Abilene"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_overall_stats), "Baseline": "Static AC", "Dataset": "Abilene"}
            pandas_df.loc[len(pandas_df.index)] = new_row
            new_row = {"Compression ratio": float(original_file/compressed_file_adaptive_ac), "Baseline": "Adaptive AC", "Dataset": "Abilene"}
            pandas_df.loc[len(pandas_df.index)] = new_row
    
    # print(pandas_df)
    plt.rcParams["figure.figsize"] = (3.5, 2.8)
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['pdf.fonttype'] = 42
    plt.grid(color='gray', axis='y')
    plt.ylim((0.0, 5))

    ax = sns.barplot(x='Dataset', y='Compression ratio', hue='Baseline', data=pandas_df, palette="rocket", order=["Geant", "Campus network", "Abilene"]) 
    # ax.set_ylabel("Compression ratio")
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    # Ticks font size
    ax.yaxis.set_tick_params(labelsize = 8)
    ax.grid(which='major', axis='y', linestyle='-')
    # Loop over the bar

    for i, patch in enumerate(ax.patches):
        # Box 1
        print(i, patch)
        if i==6 or i==7 or i==8:
            patch.set_hatch('/'*3)
        if i==3 or i==4 or i==5:
            patch.set_hatch('.'*3)
        # if i==4 or i==5 or i==8 or i==11:
        #     patch.set_hatch('/'*3)
            
        # col = patch.get_facecolor()
        # #patch.set_edgecolor(col)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
        # patch.set_facecolor('None')

    plt.legend(loc='upper right', prop={'size': 8})
    #plt.show()
    plt.tight_layout()
    plt.savefig("../data/Images/barplot_comp_ratio.pdf", bbox_inches = 'tight',pad_inches = 0)
    # plt.clf()