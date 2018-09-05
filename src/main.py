from fscnmf import FSCNMF
from helpers import parameter_parser, read_graph, read_features

def learn_model(args):
    """
    """
    A = read_graph(args.edge_path, args.order)
    X = read_features(args.feature_path)
    model = FSCNMF(A, X, args)
    model.optimize()
    model.save_embedding()

if __name__ == "__main__":
    args = parameter_parser()
    learn_model(args)
