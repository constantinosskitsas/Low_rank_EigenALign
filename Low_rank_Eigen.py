
import os
import parser
import pickle
import networkx as nx
import bipartite_Matching
def parse_args():
    parser.add_argument('--input', nargs='?', default='data/arenas_combined_edges.txt', help="Edgelist of combined input graph")

    parser.add_argument('--output', nargs='?', default='emb/arenas990-1.npy',
                        help='Embeddings path')
    parser.add_argument('--S1', type=int, default=100,
                        help='Weight of S1')
    parser.add_argument('--S2', type=int, default=10,
                        help='Weight of S2')
    parser.add_argument('--S3', type=int, default=5,
                        help='Weight of S2')
    return parser.parse_args()


def main(args):
    dataset_name = args.output.split("/")
    if len(dataset_name) == 1:
        dataset_name = dataset_name[-1].split(".")[0]
    else:
        dataset_name = dataset_name[-2]
    true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" #can be changed if desired
    print ("true alignments file: ", true_alignments_fname)
    true_alignments = None
    if os.path.exists(true_alignments_fname):
        with open(true_alignments_fname, "rb") as true_alignments_file:
            true_alignments = pickle.load(true_alignments_file)
    nx_graph = nx.read_edgelist('data/arenas_combined_edges.txt', nodetype = int, comments="%")
    nx_graph1= nx.read_edgelist('data/arenas_combined_edges.txt', nodetype = int, comments="%")
    adj_matrix = nx.adjacency_matrix(nx_graph).todense()
    A = nx.Graph(adj_matrix)
    adj_matrix1 = nx.adjacency_matrix(nx_graph1).todense()
    B = nx.Graph(adj_matrix1)

if __name__ == "__main__":
    args = parse_args()
    main(args)
