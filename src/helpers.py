import json
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import sparse
from texttable import Texttable

def parameter_parser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Wiki Chameleons.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """
    parser = argparse.ArgumentParser(description = "Run FSCNMF.")

    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "./input/chameleon_edges.csv",
	                help = "Edge list csv.")

    parser.add_argument("--feature-path",
                        nargs = "?",
                        default = "./input/chameleon_features.json",
	                help = "Node features csv.")

    parser.add_argument("--output-path",
                        nargs = "?",
                        default = "./output/chameleon_fscnmf.csv",
	                help = "Target embedding csv.")

    parser.add_argument("--features",
                        nargs = "?",
                        default = "sparse",
	                help = "Feature matrix structure.")

    parser.add_argument("--dimensions",
                        type = int,
                        default = 32,
	                help = "Number of dimensions. Default is 32.")

    parser.add_argument("--order",
                        type = int,
                        default = 2,
	                help = "Order of approximation. Default is 2.")

    parser.add_argument("--iterations",
                        type = int,
                        default = 500,
	                help = "Number of iterations. Default is 500.")

    parser.add_argument("--alpha_1",
                        type = float,
                        default = 1000.0,
	                help = "Alignment parameter for adjacency matrix. Default is 1000.0.")

    parser.add_argument("--alpha_2",
                        type = float,
                        default = 1.0,
	                help = "Adjacency basis factor matrix regularization parameter. Default is 1.0.")

    parser.add_argument("--alpha_3",
                        type = float,
                        default = 1.0,
	                help = "Adjacency feature factor matrix regularization parameter. Default is 1.0.")

    parser.add_argument("--beta_1",
                        type = float,
                        default = 1000.0,
	                help = "Alignment parameter for feature matrix. Default is 1000.0.")

    parser.add_argument("--beta_2",
                        type = float,
                        default = 1.0,
	                help = "Node basis factor matrix regularization parameter. Default is 1.0.")

    parser.add_argument("--beta_3",
                        type = float,
                        default = 1.0,
	                help = "Node feature factor matrix regularization parameter. Default is 1.0.")

    parser.add_argument("--gamma",
                        type = float,
                        default = 0.5,
	                help = "Down sampling rate for frequent features. Default is 0.5.")

    parser.add_argument("--lower-control",
                        type = float,
                        default = 10**-15,
	                help = "Numeric overflow control. Default is 10**-15.")
    
    return parser.parse_args()

def normalize_adjacency(graph):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    A = sparse.csr_matrix(nx.adjacency_matrix(graph),dtype=np.float32)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=A.shape,dtype=np.float32))
    A = A.dot(degs)
    return A

def read_graph(edge_path, order):
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param edge_path: Path to the ege list.
    :param order: Order of approximations.
    :return out_A: Target matrix.
    """
    print("Target matrix creation started.")
    graph = nx.from_edgelist(pd.read_csv(edge_path).values.tolist())
    A = normalize_adjacency(graph)
    if order > 1:
        powered_A, out_A = A, A
        
        for power in tqdm(range(order-1)):
            powered_A = powered_A.dot(A)
            out_A = out_A + powered_A
    else:
        out_A = A
    print("Factorization started.")
    return out_A

def read_features(feature_path):
    """
    Method to get node feaures.
    :param feature_path: Path to the node features.
    :return X: Node features.
    """
    features = pd.read_csv(feature_path)
    features = np.array(features)[:,1:]
    return features

def read_sparse_features(feature_path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """

    features = json.load(open(feature_path))
    index_1 = [int(k) for k,v in features.items() for fet in v]
    index_2 = [int(fet) for k,v in features.items() for fet in v]
    values = [1.0]*len(index_1) 

    nodes = [int(k) for k,v in features.items()]
    node_count = max(nodes)+1

    feature_count = max(index_2)+1
    features = sparse.csr_matrix(sparse.coo_matrix((values,(index_1,index_2)),shape=(node_count,feature_count),dtype=np.float32))
    return features

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())
