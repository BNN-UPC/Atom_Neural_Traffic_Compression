import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import xml.etree.ElementTree as ET
import tarfile
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

np.random.seed(7)
data_dir="./files_to_be_compressed/tfrecords/"

def add_timestep_to_dataframe(Graph, df, time_stamp):
    # This function adds one new row to the pandas df
    new_row = {"timestamp": time_stamp}

    for i in Graph:
        for j in Graph[i]:
            new_row[str(i)+"-"+str(j)] = Graph[i][j]['utilization']
    
    df.loc[len(df.index)] = new_row
    return df

if __name__ == '__main__':
    # python script_conv_topologySNDLIB.py -f geant
    # python script_conv_topologySNDLIB.py -f abilene

    parser = argparse.ArgumentParser(description='Input')
    parser.add_argument('-f', help='Filename of the dataset to process: [geant|abilene]', required=True)
    args = parser.parse_args()
    file_name = args.f

    if  file_name=="geant":
        print("+++++++++++++++++++++++++++++++")
        print("GENERATING CSV DATASET for the GEANT dataset")
        print("+++++++++++++++++++++++++++++++")
        topology = "./datasets/Geant/geant.xml"
        tms = "./datasets/Geant/directed-geant-uhlig-15min-over-4months-ALL.tgz"
    else:
        print("+++++++++++++++++++++++++++++++")
        print("GENERATING CSV DATASET for the ABILENE dataset")
        print("+++++++++++++++++++++++++++++++")
        topology = "./datasets/Abilene/abilene.xml"
        tms = "./datasets/Abilene/directed-abilene-zhang-5min-over-6months-ALL.tgz"
    

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists("./Images/"):
        os.makedirs("./Images/")

    df = pd.DataFrame(columns=['timestamp'])
    
    #######
    # Parse topology
    #######

    print(">>>>>>>>>>>>")
    print("PARSING TOPOLOGY")

    Gbase = nx.DiGraph()

    # Parse XML file
    tree = ET.parse(topology)

    x = None
    y = None

    # We store the nodes in ordered fashion
    list_nodes = list()
    count_nodes = 0
    # Iterate over nodes
    for elem in tree.iter():

        if elem.tag.endswith("node"):           
            # For each node, we iterate over it's sub-elements to get the coordinates
            for sub_elem in list(elem.iter()):
                #print(sub_elem)
                if sub_elem.tag.endswith("x"):
                    x = float(sub_elem.text)
                if sub_elem.tag.endswith("y"):
                    y = float(sub_elem.text)
            # print(elem.attrib.get('id'), x, y)
            Gbase.add_node(count_nodes, pos=(x,y))
            list_nodes.append(elem.attrib.get('id'))
            count_nodes += 1
    
    # Iterate over links
    for elem in tree.iter():
        # Iterate over all links
        if elem.tag.endswith("link"):
            # print(elem, elem.attrib)
            source = None
            target = None
            # For each link, we iterate over it's sub-elements
            for sub_elem in list(elem.iter()):
                #print(sub_elem)
                if sub_elem.tag.endswith("source"):
                    source = sub_elem.text
                if sub_elem.tag.endswith("target"):
                    target = sub_elem.text
            
            source = list_nodes.index(source) 
            target = list_nodes.index(target) 
            Gbase.add_edge(source, target)
            Gbase[source][target]["weight"] = 1
            Gbase.add_edge(target, source)
            Gbase[target][source]["weight"] = 1
            df[str(source)+"-"+str(target)] = " "
            df[str(target)+"-"+str(source)] = " "

    pos=nx.get_node_attributes(Gbase,'pos')
    nx.draw(Gbase, pos, with_labels = True)
    #plt.show()
    #plt.savefig(img_dir+"/topology.pdf",bbox_inches='tight')
    plt.close()
    
    if file_name=="geant":
        nx.write_gml(Gbase, "./datasets/Geant/Geant_topology.gml")
    else:
        nx.write_gml(Gbase, "./datasets/Abilene/Abilene_topology.gml")
    topology_class = NetworkTopology()
    topology_class.generate_topology_from_Graph(Gbase)
    topology_class.compute_SP_routing()

    traffic_matrix = np.zeros((len(list_nodes), len(list_nodes)))

    #######
    # Parse TM
    #######
    print("************")
    print("PARSING TMs")

    with tarfile.open(tms) as tf:
        count = 0
        for entry in tf: 
            if entry.name.endswith(".xml"):
                fileobj = tf.extractfile(entry)
                traffic_matrix.fill(0)

                # Print the xml file name
                # print(entry.name)
                
                time_stamp = entry.name.split("5min")[2][1:-4]

                # Parse XML file
                tree = ET.parse(fileobj)

                # Iterate over childs of XML file to obtain the TM
                for elem in tree.iter():
                    # If the elem is a traffic demand
                    if "demand" in elem.tag and "demands" not in elem.tag and "demandValue" not in elem.tag:
                        source = None
                        destination = None
                        demandValue = None

                        df_col = elem.attrib.get('id')
                        # For each demand, we obtain the source, destination and traffic demand
                        for sub_elem in list(elem.iter()):
                            if sub_elem.tag.endswith("source"):
                                source = sub_elem.text
                            elif sub_elem.tag.endswith("target"):
                                destination = sub_elem.text
                            elif sub_elem.tag.endswith("demandValue"):
                                demandValue = float(sub_elem.text)
                        #print(source, destination, demandValue)
                        src_tm = list_nodes.index(source) 
                        dst_tm = list_nodes.index(destination) 
                        traffic_matrix[src_tm, dst_tm] = demandValue

                topology_class.apply_tm_2_routing(traffic_matrix)
                df = add_timestep_to_dataframe(topology_class.Graph, df, time_stamp)

                count += 1
                if count%1000==0:
                    print(count, entry.name)

    # Order in increasing order of timestamp
    sorted_df = df.sort_values(by=['timestamp'])
    if file_name=="geant":
        sorted_df.to_csv("./datasets/Geant/geant_dataset.csv", index=False) 
        np.save("./datasets/Geant/routing.npy", topology_class.routing_path)
    else:
        sorted_df.to_csv("./datasets/Abilene/abilene_dataset.csv", index=False) 
        np.save("./datasets/Abilene/routing.npy", topology_class.routing_path)