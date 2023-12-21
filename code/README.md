# Instructions to execute
1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3.8 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. We create the directories where we will store our datasets.
```ruby
mkdir data/datasets
mkdir data/files_to_be_compressed
mkdir data/files_to_be_compressed/tfrecords
```

## Preparing the real-world datasets
### Instructions to download and process the real-world datasets (ABILENE/Geant) 
The following steps indicate how to download and process the datasets from source. 

1. First, we need to download the ABILENE/GEANT datasets from [http://sndlib.zib.de/home.action](http://sndlib.zib.de/home.action) together with the ABILENE/GEANT topologies in xml format. Then, we need to place the downloaded files in the corresponding directory in './data/datasets' before we execute the following scripts.

2. We execute the following scripts from the '../data' directory to obtain the csv files. Notice that this step can take up to 10 minutes, depending on the CPU. With the flag '-r' we are indicating the number of different masks we created per time-bin. 
```ruby
1. data$: python script_conv_topologySNDLIB.py -f geant
2. data$: python script_conv_topologySNDLIB.py -f abilene
3. data$: python convert_data_real_world_sndlib.py -r 40 -f geant
4. data$: python convert_data_real_world_sndlib.py -r 40 -f abilene
```

### Instructions to download and process the campus network dataset
The following steps indicate how to download the campus network dataset and how to process it.

1. To prepare the campus network datasets, we only need to download it from [https://drive.google.com/file/d/1u6lmxAGGQr7h1lCKhxGScJ5Bu7aJkj5j/view?usp=sharing](https://drive.google.com/file/d/1u6lmxAGGQr7h1lCKhxGScJ5Bu7aJkj5j/view?usp=sharing) and unzip it in the "./data/datasets" directory.  

2. Then, we need to execute the following script to convert the csv to the desired tfrecords. At the top of the script, there is a variable flag_Kb to indicate if we use the raw dataset or we convert it to Kbytes. If we are working with raw dataset, we won't be able to compress/decompress as the range of link utilization values is very large and it doesn't fit in memory when computing the probability distribution. That's why it's interesting to have the flag flag_Kb set to True. Please be aware that this script can take several hours to finish executing. In the next step we indicate how to train and compress using the already-processed datasets used in the paper.

```ruby
data$: python convert_data_real_world_campus.py -r 20
```

### Preparing the datasets used in the paper
To avoid processing the entire datasets, we also provide the already-processed datasets used in our paper. These can be found in [https://drive.google.com/file/d/1CF8og3RKeE4jkT79Q8iD68ra-06ZxtMN/view?usp=sharing](https://drive.google.com/file/d/1CF8og3RKeE4jkT79Q8iD68ra-06ZxtMN/view?usp=sharing) and they should be unzipped and moved to "./data/files_to_be_compressed/tfrecords".

## Training the models
To execute the following scripts to train the models, we should do it from the "code" directory. 

1. First we will train Atom with implemented with the GRU model. To do this, we only need to execute the script train_gru_single_link.py as indicated below. In this file, at the top there is a variable "dataset_name" that indicates which dataset to train on. The name of the "dataset_name" should match the name of some directory in "../data/files_to_be_compressed/tfrecords". In addition to this, there are several hyperparameters that can be configured at the top of the file (e.g., number of units, batch size).

```ruby
code$: python train_gru_single_link.py
```

2. To train the ST-GNN version we will execute the following command. Similarly, at the top of the python script we can indicate which dataset to train on and we can set the parameters of our training.

```ruby
code$: python train_stgnn_network_wide.py
```

#### To facilitate the execution of our code, we already provide trained models in the "./log/" directory. Unzip and move them all to the "./log/" directory.

## Compressing/decompressing the datasets

Once the training process finished, we can proceed to compress the datasets using Atom.

### Single link scenario
Before proceeding to the compression/decompression, we should note that in the scripts we use "import train_gru_single_link" to load the model configuration and weights. This means that the global variables from train_gru_single_link.py will be imported and used in the compressor_gru.py script. Make sure the units, seq_layer_size and batch_size are set to match the parameters in the model you want to use to compress (i.e., match with the parameters of the model name in the flat -m).

1. To compress the datasets using Atom (GRU)/Adaptive AC/Static AC for each link independently, we will execute the commands below. We use the flags "-f 2" for the Atom (GRU), "-f 9" for Adaptive AC and "-f 11" for the Static AC. Note that the flag "-m" should point properly to the model directory that we want to use. In other words, it should point to the model directory within the "./log/" that we want to use removing the "_masked_GNN" from the name.

```ruby
code$: python compressor_gru.py -d KB_agh_w5_r20_maxt102799 -f 2 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 9 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 11 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
```

2. Once we have performed the compression, we proceed to decompress the dataset and we calculate the difference between the original dataset and the decompressed dataset. This is to make sure that we can recover 100% of the original data after decompression. Similarly, we also use the flags "-f 2" for the Atom (GRU), "-f 9" for Adaptive AC and "-f 11" for the Static AC.

```ruby
code$: python decompressor_gru.py -d KB_agh_w5_r20_maxt102799 -f 2 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 9 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 11 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
```

3. Finally, once we have decompressed the datasets and we have seen that we recovered the original dataset we can proceed to plot some figures to show the compression ratios of all three methods. 

```ruby
code$: python tfrecords_print_compress_info.py -d KB_agh_w5_r20_maxt102799 -t 0
code$: python boxplot_comp_ratio_links.py
```

### Network-wide scenario
1. To compress the datasets using Atom (STGNN)/Static AC/Adaptive AC using all links, we will execute the commands below. We use the flags "-f 0" for Atom (STGNN), "-f 1" for Static AC and "-f 10" for Adaptive AC to compress all links at the same time. Note that the flag "-m" should point properly to the model directory that we want to use. In other words, it should point to the model directory within the "./log/" that we want to use removing the "_masked_GNN" from the name.

```ruby
code$: python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 0 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 1 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python compressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 10 -r 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
```

2. Once we have performed the compression, we proceed to decompress the dataset and we calculate the difference between the original dataset and the decompressed dataset. This is to make sure that we can recover 100% of the original data after decompression. Similarly, we also use the flags "-f 0" for Atom (STGNN), "-f 1" for Static AC and "-f 10" for Adaptive AC.

```ruby
code$: python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 0 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 1 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
code$: python decompressor_multiple_models.py -d KB_agh_w5_r20_maxt102799 -f 10 -m KB_agh_w5_r20_maxt102799_u50_s10_b256
```

3. Finally, once we have decompressed the datasets and we have seen that we recovered the original dataset we can proceed to plot some figures to show the compression ratios of all three methods. 

```ruby
code$: python tfrecords_print_compress_info.py -d KB_agh_w5_r20_maxt102799 -t 0
code$: python bar_plot_comp_ratio.py
```