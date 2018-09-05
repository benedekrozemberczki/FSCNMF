FSCNMF
============================================
<p align="justify">
An implementation of "Fusing Structure and Content via Non-negative Matrix Factorization for Embedding Information Networks". FSCNMF is a graph embedding algorithm which learns an embedding of nodes and fuses the node representations with node attributes. The procedure places nodes in an abstract feature space where information about a fixed order procimity is preserved and attributes of neighbours within the proximity are also part of the representation. FSCNMF learns the joint feature-proximal representations using regularized non-negative matrix factorization. In our implementation we assumed that the proximity matrix used in the approximation is sparse, hence the solution runtime can be linear in the number of nodes for low proximity. For a large proximity order value (which is larger than the graph diameter) the runtime is quadratic. We did not make any assumptions about the sparsity of the feature matrix.

This repository provides an implementation for FSCNMF as described in the paper:
> FSCNMF: Fusing Structure and Content via Non-negative Matrix Factorization for Embedding Information Networks.
> Sambaran Bandyopadhyay, Harsh Kara, Aswin Kannan and M N Murty
> arXiv, 2018.
>https://arxiv.org/pdf/1804.05313.pdf


### Requirements

The codebase is implemented in Python 2.7. package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
texttable         1.2.1
scipy             1.1.0
argparse          1.1.0
```

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Wikipedia Giraffes` is included in the  `input/` directory.

### Options

Learning of the embedding is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --input STR                   Input graph path.                                 Default is `data/politician_edges.csv`.
  --embedding-output STR        Embeddings path.                                  Default is `output/embeddings/politician_embedding.csv`.
  --cluster-mean-output STR     Cluster centers path.                             Default is `output/cluster_means/politician_means.csv`.
```

#### Model options

```
  --initial-learning-rate FLOAT   Initial learning rate.                                        Default is 0.001.
  --minimal-learning-rate FLOAT   Final learning rate.                                          Default is 0.0001.
  --annealing-factor FLOAT        Annealing factor for learning rate.                           Default is 1.0.
  --initial-gamma FLOAT           Initial clustering weight coefficient.                        Default is 0.1.
  --lambd FLOAR                   Smoothness regularization penalty.                            Default is 0.0625.
  --cluster-number INT            Number of clusters.                                           Default is 20.
  --overlap-weighting STR         Weight construction technique for regularization.             Default is `normalized_overlap`.
  --regularization-noise FLOAT    Uniform noise max and min on the feature vector distance.     Default is 10**-8.
```

### Examples

The following commands learn a graph embedding and cluster center and writes them to disk. The node representations are ordered by the ID.

Creating a GEMSEC embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.

```
python src/embedding_clustering.py
```
Creating a DeepWalk embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.

```
python src/embedding_clustering.py --model DeepWalk
```

Turning off the model saving.

```
python src/embedding_clustering.py --dump-matrices False
```

Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/embedding_clustering.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --log-output output/cluster_means/company_means.csv --cluster-mean-output output/logs/company.json
```

Creating a clustered embedding of the default dataset in 32 dimensions, 20 sequences per source node with length 160 and 10 cluster centers.

```
python src/embedding_clustering.py --dimensions 32 --num-of-walks 20 --random-walk-length 160 --cluster-number 10
```
