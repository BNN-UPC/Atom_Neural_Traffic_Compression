import numpy as np
import networkx as nx
import tensorflow as tf
import functools
import operator

np.random.seed(21)

@tf.function
def chi_squared_uniform_tf(u):
    u = tf.convert_to_tensor(u)
    k=tf.constant(50)
    value_range = tf.constant([0.,1.], dtype=u.dtype)
    x = tf.histogram_fixed_width(u,nbins=k, value_range=value_range)
    n = tf.shape(u)[0]
    n = tf.cast(n, u.dtype)
    k = tf.cast(k, u.dtype)
    x = tf.cast(x, u.dtype)

    expected_counts = n/k
    chi_squared = tf.reduce_sum(tf.square(x-expected_counts)/expected_counts)
    return chi_squared

@tf.function(experimental_compile=True)
def mccdf(samples, y):
    return tf.reduce_mean(
        tf.cast(samples < tf.cast(y, samples.dtype), samples.dtype),
        axis=0
    )

class CustomCallback(tf.keras.callbacks.Callback):
    """ Callback to plot the cdf after training on the test set.
        We also compute the Chi Square Metric
    """
    def __init__(self, model, valid_data, file_writer, eval_epoch, flag_model):
        """ Save params in constructor
        """
        self.model = model
        self.test_data = valid_data
        self.list_cdf = list()
        self.file_writer = file_writer
        self.eval_epoch = eval_epoch
        # Flag model indicates '0' for model with no mask
        # and '1' with model with mask
        self.flag_model = flag_model

    def on_epoch_end(self, epoch, logs={}):
        self.list_cdf.clear()

        if epoch%self.eval_epoch==0:
            for x,y in self.test_data:
                out = self.model(x, training=False)

                cdf_values = out.cdf(y)
                if self.flag_model==0:
                    self.list_cdf = self.list_cdf + list(cdf_values.numpy())
                else:
                    self.list_cdf = self.list_cdf + list(cdf_values[tf.squeeze(x["input_5"])].numpy())
            
            chi_sqr = chi_squared_uniform_tf(np.asarray(self.list_cdf))
            with self.file_writer.as_default():
                tf.summary.histogram(name='pred_dist.cdf(y_true)', data=self.list_cdf, step=epoch)
                tf.summary.scalar(name="Chi_square_callback", data=chi_sqr.numpy(), step=epoch)
                self.file_writer.flush()
            

class ChiSquared(tf.keras.metrics.Metric):
    # Stateful metric implementation that keeps the counts accross batches

    def __init__(self, bins=50, num_mc=512, dtype=tf.float32, name='chi_squared_uniform'):
        super(ChiSquared, self).__init__(name=name)
        # add_weight is used to create state variables
        self.histogram = self.add_weight(name='histogram', initializer='zeros', shape=(bins,), dtype=tf.int32)
        self.counts = self.add_weight(name='counts', initializer='zeros', shape=(), dtype=tf.int32)
        self._num_mc = num_mc
        self._bins=bins
        self._dtype = dtype


    def update_state(self, y_true, y_pred, sample_weight=None):
        # uses the targets y_true and the model predictions y_pred to update the state variables
        #samples = y_pred.sample(self._num_mc)
        u = y_pred.cdf(y_true)
        #u = mccdf(cdf_vals, y_true)
        value_range = tf.constant([0., 1.], dtype=y_true.dtype)
        self.histogram.assign_add(tf.histogram_fixed_width(u, nbins=self._bins, value_range=value_range))
        self.counts.assign_add(tf.cast(tf.reduce_prod(tf.shape(y_true)),tf.int32))

    def result(self):
        # uses the state variables to compute the final results
        k = tf.cast(self._bins, self._dtype)
        x = tf.cast(self.histogram, self._dtype)
        counts = tf.cast(self.counts, self._dtype)
        expected_counts = counts / k
        chi_squared = tf.reduce_sum(tf.square(x - expected_counts) / expected_counts)
        return chi_squared

    def reset_state(self):
        # reinitializes the state of the metric
        self.histogram.assign(tf.zeros_like(self.histogram))
        self.counts.assign(tf.zeros_like(self.counts))


def get_lr_metric(optimizer):
    # Used as metrics in compile() to print the lr for each epoch
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

class NetworkTopology:
    def __init__(self):
        self.flag = None
        self.first = list()
        self.second = list()
        self.first_tensor = list()
        self.second_tensor = list()
        self.edge_2_id = dict()
        self.id_2_edge = dict()
        # For each edge we store a pair of (src, dst) nodes to indicate the paths crossing the edge
        self.crossing_paths = dict()
        self.Graph = None
        self.numNodes = None
        self.routing_path = None
        self.routing_edges = None
    
    def generate_topology_from_Graph(self, graph):
        self.Graph = graph.copy()
        self.clear()
        self.numEdges = self.Graph.number_of_edges()
        self.numNodes = self.Graph.number_of_nodes()

        self.initialize_dicts()
        self.first_second()
    
    def compute_crossing_paths(self, routing):
        self.crossing_paths.clear()

        for n1 in range (self.numNodes):
            for n2 in range (self.numNodes):
                if (n1 != n2):
                    currentPath = routing[n1, n2]
                    i = 0
                    j = 1

                    # For each edge from the path, we store that (n1, n2) is crossing the edge
                    while (j < len(currentPath)):
                        firstNode = currentPath[i]
                        secondNode = currentPath[j]
                        if str(firstNode)+':'+str(secondNode) in self.crossing_paths:
                            self.crossing_paths[str(firstNode)+':'+str(secondNode)].append((n1, n2))
                        else:
                            self.crossing_paths[str(firstNode)+':'+str(secondNode)] = list()
                            self.crossing_paths[str(firstNode)+':'+str(secondNode)].append((n1, n2))
                        i = i + 1
                        j = j + 1

    def mp_traffic_inject(self, dir_routing):
        self.first.clear()
        self.second.clear()

        routing = np.load(dir_routing, allow_pickle=True)
        self.compute_crossing_paths(routing)
        
        for i in self.Graph:
            for j in self.Graph[i]:
                neighbour_edges = self.Graph.edges(j)
                # Take output links of node 'j'

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        receives_traffic_main_edge = False

                        # Iterate over all paths of receiver edge
                        for path in self.crossing_paths[str(m) +':'+ str(n)]:
                            # If the receiver edge shares at least one path with "injector" edge,
                            # we consider it in the MP
                            if path in self.crossing_paths[str(i) +':'+ str(j)]:
                                receives_traffic_main_edge = True
                                break
                        if receives_traffic_main_edge:
                            self.first.append(self.edge_2_id[str(i) +':'+ str(j)])
                            self.second.append(self.edge_2_id[str(m) +':'+ str(n)])

        self.first_tensor = tf.convert_to_tensor(self.first, dtype=tf.int64)
        self.second_tensor = tf.convert_to_tensor(self.second, dtype=tf.int64)

    def initialize_routing(self):
        # Initialize the link weights
        for i in range (self.numNodes):
            for j in range (self.numNodes):
                if i!=j:
                    if self.Graph.has_edge(i, j):
                        
                        if self.flag!=5:
                            # Replicate nodes to have bidirectional topology
                            self.Graph.add_edge(j, i)
                            self.Graph[j][i]['weight'] = 1 #np.random.randint(0, 21)
                    
                        self.Graph[i][j]['weight'] = 1 #np.random.randint(0, 21)    

    def clear(self):
        self.first.clear()
        self.second.clear()
        self.edge_2_id.clear()
        self.id_2_edge.clear()
        self.crossing_paths.clear()

    def initialize_dicts(self): 
        self.numEdges = self.Graph.number_of_edges()
        position = 0
        for i in self.Graph:
            for j in self.Graph[i]:                 
                self.edge_2_id[str(i)+':'+str(j)] = position
                self.id_2_edge[str(position)] = (i, j)
                position += 1

    def first_second(self):
        # Link (1, 2) recibe trafico de los links que inyectan en el nodo 1
        # un link que apunta a un nodo envía mensajes a todos los links que salen de ese nodo

        for i in self.Graph:
            for j in self.Graph[i]:
                neighbour_edges = self.Graph.edges(j)
                # Take output links of node 'j'

                for m, n in neighbour_edges:
                    if ((i != m or j != n) and (i != n or j != m)):
                        self.first.append(self.edge_2_id[str(i) +':'+ str(j)])
                        self.second.append(self.edge_2_id[str(m) +':'+ str(n)])

        self.first_tensor = tf.convert_to_tensor(self.first, dtype=tf.int64)
        self.second_tensor = tf.convert_to_tensor(self.second, dtype=tf.int64)

    def compute_SP_routing(self):
        self.routing_path = np.zeros((self.numNodes, self.numNodes), dtype=object)
        self.routing_edges = np.zeros((self.numNodes*(self.numNodes-1), self.numEdges))
        path_len = list()

        flow_count = 0
        for n1 in range (self.numNodes):
            for n2 in range (self.numNodes):
                if (n1 != n2):
                    currentPath = nx.shortest_path(self.Graph, source=n1, target=n2, weight='weight')
                    self.routing_path[n1,n2] = currentPath
                    i = 0
                    j = 1

                    while (j < len(currentPath)):
                        firstNode = currentPath[i]
                        secondNode = currentPath[j]
                        edge_pos = self.edge_2_id[str(firstNode)+':'+str(secondNode)]
                        self.routing_edges[flow_count,edge_pos] = 1  
                        i = i + 1
                        j = j + 1
                    
                    flow_count += 1
                    path_len.append(len(currentPath))        

    def init_uti_zero(self):
        for i in self.Graph:
            for j in self.Graph[i]:
                self.Graph[i][j]['utilization'] = 0

    def allocate_to_destination_sp(self, src, dst, bw_allocate): 
        currentPath = self.routing_path[src,dst]
        
        i = 0
        j = 1

        while (j < len(currentPath)):
            firstNode = currentPath[i]
            secondNode = currentPath[j]

            self.Graph[firstNode][secondNode]['utilization'] += bw_allocate  
            i = i + 1
            j = j + 1

def write_sample_tfrecords(writer, list_link_uti_window, t):
    # Normalize data
    norm_link_utis = np.asarray(list_link_uti_window, dtype=np.float64)
    # Flatten the list except for the last timestep
    flatten_norm_link_utis = functools.reduce(operator.iconcat, norm_link_utis[:-1], [])
    example = tf.train.Example(features=tf.train.Features(feature={
        "link_uti": _float_features(flatten_norm_link_utis),
        "next_uti": _float_features(norm_link_utis[-1]) # ylabel
        }))
    # print(len(flatten_norm_link_utis), len(norm_link_utis[-1]))
    writer.write(example.SerializeToString())

def write_flat_sample_real_tfrecords(writer, list_link_uti_window, next_uti):
    # This function is similar like the one above but we used it with the real world dataset.
    # This is because the samples come here already flattened and divided.
    # Flatten the list
    flatten_norm_link_utis = functools.reduce(operator.iconcat, list_link_uti_window, [])
    example = tf.train.Example(features=tf.train.Features(feature={
        "link_uti": _float_features(flatten_norm_link_utis),
        "next_uti": _float_features(next_uti) # ylabel
        }))
    # print(len(flatten_norm_link_utis), len(norm_link_utis[-1]))
    writer.write(example.SerializeToString())

def write_sample_tfrecords_flagged_links_masked(writer, fl_list_link_uti, mask_pred_one_hot, y_label, y_label_feat):
    flatten_norm_link_utis = functools.reduce(operator.iconcat, fl_list_link_uti[:], [])
    mask_pred_0 = mask_pred_one_hot[...,0]
    mask_pred_1 = mask_pred_one_hot[...,1]
    # Flatten the list except for the last timestep
    example = tf.train.Example(features=tf.train.Features(feature={
        "link_uti": _float_features(flatten_norm_link_utis),
        "mask_pred_0": _float_features(mask_pred_0), # mark the links with 1 over link which we will predict
        "mask_pred_1": _float_features(mask_pred_1),
        "feat_pred_link": _float_features(y_label_feat), 
        "next_uti": _float_features(y_label) # ylabel
    }))
    writer.write(example.SerializeToString())

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_features(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class AGH:
    """
    Topology of AGH network
    """
    _NAMES = ['ucirtr-b6',
              'b1rtr-b6',
              'ftjrtr-b1',
              'ucirtr-cyfronet',
              'b6rtr-ftj',
              'b6rtr-b1',
              'ftjrtr-cyfronet',
              'ucirtr-b1',
              'b6rtr-ms',
              'ftjrtr-b6',
              'ucirtr-ftj',
              'ftjrtr-uci',
              'b1rtr-uci',
              'b1rtr-ftj',
              'b6rtr-uci',
              'cyfronet-ucirtr',
              'cyfronet-ftjrtr',
              'ms-b6rtr']

    @staticmethod
    def adjacency():
        columns = ['-'.join([n.replace('rtr', '') for n in x.split('-')]) for x in AGH._NAMES]
        A = np.zeros((len(columns), len(columns)), dtype=np.float32)
        for i, e1 in enumerate(columns):
            for j, e2 in enumerate(columns):
                if e1.split('-')[1] == e2.split('-')[0]:
                    A[i, j] = 1.
        return A
