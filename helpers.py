import argparse
import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
from texttable import Texttable

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Wiki Giraffes.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by node ID.
    """

    parser = argparse.ArgumentParser(description = "Run FSCNMF.")


    parser.add_argument('--edge-path',
                        nargs = '?',
                        default = './input/giraffe_edges.csv',
	                help = 'Input folder with jsons.')

    parser.add_argument('--feature-path',
                        nargs = '?',
                        default = './input/giraffe_features.csv',
	                help = 'Input folder with jsons.')

    parser.add_argument('--output-path',
                        nargs = '?',
                        default = './output/giraffe_fscnmf.csv',
	                help = 'Input folder with jsons.')

    parser.add_argument('--dimensions',
                        type = int,
                        default = 32,
	                help = 'Number of dimensions. Default is 128.')

    parser.add_argument('--workers',
                        type = int,
                        default = 4,
	                help = 'Number of workers. Default is 4.')

    parser.add_argument('--order',
                        type = int,
                        default = 2,
	                help = 'Number of workers. Default is 4.')

    parser.add_argument('--iterations',
                        type = int,
                        default = 500,
	                help = 'Number of dimensions. Default is 10.')

    parser.add_argument('--alpha_1',
                        type = float,
                        default = 1000.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--alpha_2',
                        type = float,
                        default = 1.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--alpha_3',
                        type = float,
                        default = 1.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--beta_1',
                        type = float,
                        default = 1000.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--beta_2',
                        type = float,
                        default = 1.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--beta_3',
                        type = float,
                        default = 1.0,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--gamma',
                        type = float,
                        default = 0.5,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')

    parser.add_argument('--lower-control',
                        type = float,
                        default = 10**-15,
	                help = 'Down sampling rate for frequent features. Default is 0.0001.')
    
    return parser.parse_args()

def normalize_adjacency(graph):
    """
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    A = sparse.csr_matrix(nx.adjacency_matrix(graph),dtype=np.float32)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs,(ind,ind)),shape=A.shape,dtype=np.float32))
    A = A.dot(degs)
    return A

def read_graph(edge_path, order):
    """
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
    """
    features = pd.read_csv(feature_path)
    X = np.array(features)[:,1:]
    return X

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)

    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),v] for k,v in args.iteritems()])
    print t.draw()
