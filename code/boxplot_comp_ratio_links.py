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

# This script is to compute the boxplots of the link-level compression ratios for all three real-world datasets
# Before executing this script, I need to execute the script tfrecords_print_compress_info.py!!!!!

if __name__ == '__main__':
    # python boxplot_comp_ratio_links.py
    list_datasets = ["abilene_w5_r40_maxt41741", "geant_w5_r40_maxt6063", "KB_agh_w5_r20_maxt102799"]
    compressed_dir = "../data/compressed/"

    geant = pd.DataFrame(columns=['Atom (RNN)', 'GZIP (link)', 'Static AC (link)', 'Adaptive AC (link)', 'Dataset'])
    agh = pd.DataFrame(columns=['Atom (RNN)', 'GZIP (link)', 'Static AC (link)', 'Adaptive AC (link)','Dataset'])
    abilene = pd.DataFrame(columns=['Atom (RNN)', 'GZIP (link)', 'Static AC (link)', 'Adaptive AC (link)','Dataset'])

    comp_ratios_gru = list()
    comp_ratios_overall_link = list()
    comp_ratios_gzip = list()
    comp_ratios_adaptive_ac_link = list()

    for dataset in list_datasets:
        print("Reading dataset ", dataset)

        np_original_file = np.load("../data/files_to_be_compressed/tfrecords/"+dataset+"/per_link_ORIGINAL_ts.npy")

        dir_orig_links = compressed_dir+dataset+"/original_links/"
        # Create dir to store original links one by file
        if not os.path.exists(dir_orig_links):
            os.makedirs(dir_orig_links)
        
        orig_file_size = np.zeros(len(np_original_file))
        # Iterate over all links and store a file per link
        for l in range(len(np_original_file)):
            file_name = dir_orig_links+"link_"+str(l)+".npy"
            np.save(file_name, np_original_file[l])
            
            orig_file_size[l] = os.path.getsize(dir_orig_links+"link_"+str(l)+".npy")/1000

        for filename in os.listdir(compressed_dir+dataset):
            if filename.endswith("_overall_stats_link"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        link_id = int(comprsd_file.split('.')[0].split('_')[1])
                        comp_file_size = os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
                        comp_ratios_overall_link.append(orig_file_size[l]/comp_file_size)
            elif filename.endswith("_GRU"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        link_id = int(comprsd_file.split('.')[0].split('_')[1])
                        comp_file_size = os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
                        comp_ratios_gru.append(orig_file_size[l]/comp_file_size)
            elif filename.endswith("_gzip"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        link_id = int(comprsd_file.split('.')[0].split('_')[1])
                        comp_file_size = os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
                        comp_ratios_gzip.append(orig_file_size[l]/comp_file_size)
            elif filename.endswith("_adaptive_AC_link"):
                # Iterate over the compressed link files
                for comprsd_file in os.listdir(compressed_dir+dataset+"/"+filename):
                    if comprsd_file.startswith("link_"):
                        link_id = int(comprsd_file.split('.')[0].split('_')[1])
                        comp_file_size = os.path.getsize(compressed_dir+dataset+"/"+filename+"/"+comprsd_file)/1000
                        comp_ratios_adaptive_ac_link.append(orig_file_size[l]/comp_file_size)

        for t_bin in range(len(comp_ratios_gru)):
            if "geant" in dataset:
                dataset_tag = "Geant"
                new_row = {"Static AC (link)": float(comp_ratios_overall_link[t_bin]), "Atom (RNN)": float(comp_ratios_gru[t_bin]), 'GZIP (link)': float(comp_ratios_gzip[t_bin]), 'Adaptive AC (link)': float(comp_ratios_adaptive_ac_link[t_bin]), "Dataset": dataset_tag}
                geant.loc[len(geant.index)] = new_row
            elif "KB_agh" in dataset:
                dataset_tag = "Campus network"
                new_row = {"Static AC (link)": float(comp_ratios_overall_link[t_bin]), "Atom (RNN)": float(comp_ratios_gru[t_bin]), 'GZIP (link)': float(comp_ratios_gzip[t_bin]), 'Adaptive AC (link)': float(comp_ratios_adaptive_ac_link[t_bin]), "Dataset": dataset_tag}
                agh.loc[len(agh.index)] = new_row
            elif "abilene" in dataset:
                dataset_tag = "Abilene"
                new_row = {"Static AC (link)": float(comp_ratios_overall_link[t_bin]), "Atom (RNN)": float(comp_ratios_gru[t_bin]), 'GZIP (link)': float(comp_ratios_gzip[t_bin]), 'Adaptive AC (link)': float(comp_ratios_adaptive_ac_link[t_bin]), "Dataset": dataset_tag}
                abilene.loc[len(abilene.index)] = new_row

        os.system("rm -rf %s" % (dir_orig_links))
        print("GRU, Overall and GZIP: ", np.mean(comp_ratios_gru), np.mean(comp_ratios_overall_link), np.mean(comp_ratios_gzip))
        comp_ratios_gru.clear()
        comp_ratios_overall_link.clear()
        comp_ratios_gzip.clear()

    cdf = pd.concat([geant, agh, abilene])
    mdf = pd.melt(cdf, id_vars=['Dataset'], var_name=['Topology'])      # MELT
    #plt.ylim((1.0, 11.0))
    plt.rcParams["figure.figsize"] = (3.5, 2.8)
    plt.rcParams.update({'font.size': 10})
    plt.rcParams['pdf.fonttype'] = 42
    plt.grid(color='gray', axis='y')

    ax = sns.boxplot(x="Dataset", y="value", hue="Topology", data=mdf, palette="rocket",  linewidth=1, showfliers = False)  # RUN PLOT
    ax.set_ylabel("Compression ratio")
    ax.set(xlabel=None)
    # Ticks font size
    ax.yaxis.set_tick_params(labelsize = 8)
    ax.grid(which='major', axis='y', linestyle='-')
    # Loop over the bar

    for i, patch in enumerate(ax.patches):
        # Box 1
        # print(i, patch)
        # if i==0 or i==1 or i==6 or i==9:
        #     patch.set_hatch('/'*5)
        if i==2 or i==3 or i==9 or i==13:
            patch.set_hatch('.'*3)
        if i==4 or i==5 or i==10 or i==14:
            patch.set_hatch('/'*3)
            
        # col = patch.get_facecolor()
        # #patch.set_edgecolor(col)
        patch.set_edgecolor("black")
        # patch.set_facecolor('None')

    #plt.text(1.25, 1.6, 'Adaptive AC', fontsize=6, rotation=60)
    plt.legend(loc='upper right', prop={'size': 8})
    #plt.show()
    plt.tight_layout()
    plt.savefig("../data/Images/boxplot_comp_ratio_per_link.pdf", bbox_inches = 'tight',pad_inches = 0)
    # plt.clf()