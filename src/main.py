from fscnmf import FSCNMF
from helpers import parameter_parser, read_graph, read_features, tab_printer

def learn_model(args):
    """
    Method to create adjacency matrix powers, read features, and learn embedding.
    :param args: Arguments object.
    """
    A = read_graph(args.edge_path, args.order)
    X = read_features(args.feature_path)
    model = FSCNMF(A, X, args)
    model.optimize()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    learn_model(args)
