"""Sort images by an automatically generated ID before photo-identification"""

from collections import Counter
from pathlib import Path
from typing import Tuple, List, Optional
import os
import shutil

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def prep_images(image_dir) -> None:
    """Copy all images to a tempory directory and return encounter information"""
    images, encounters = process_images(image_dir)
    save_encounter_info(image_dir, encounters, images)

def process_images(image_root: str) -> Tuple[List[str], List[str]]:
    """Copy all images to a tempory directory and return encounter information"""
    image_list = []
    encounter_list = []

    tmp_dir = os.path.join(image_root, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    
    i = 0
    for path, dirs, files in os.walk(image_root, topdown=True):
        dirs[:] = [d for d in dirs if d not in 'tmp']
        dirs[:] = [d for d in dirs if 'cluster' not in d]
        for file in files:
            if not file.lower().endswith('.jpg'):
                continue
                
            image_list.append(file)
            
            full_path = os.path.join(path, file)
            p = Path(full_path)
            encounter = p.parts[-2].replace(' (CROPPED)', '')
            encounter_list.append(encounter)
            
            shutil.copy(full_path, tmp_dir)
            i += 1
            
    print(f'Copied {i} images to:', tmp_dir)
    
    return image_list, encounter_list

def save_encounter_info(output_dir: str, encounters: List[str], images: List[str]) -> None:
    encounter_df = pd.DataFrame(dict(encounter=encounters, image=images))
    encounter_path = os.path.join(output_dir, 'encounter_info.csv')
    encounter_df.to_csv(encounter_path, index=False)
    print('Saved encounter information to:', encounter_path)

# def main() -> None:
#     """Main function to run autosort."""

#     # parse command line arguments
#     args = None

#     # check if the user wants to run the preparation step
#     if args['prep']:
#         print('Preparing images for autosorting.')
#         prep_images()
    
#     # check if the user wants to run the extraction step
#     if args['extract']:
#         print('Extracting features from images.')
#         extract()

#     # load the configuration file
#     with open('config.yaml', 'r') as f:
#         config = yaml.safe_load(f)

#     root = config['image_root']

#     # cluster the feature vectors into proposed individuals
#     fnames, features = load_features(root)
#     cluster_ids = cluster_images(features, config['cluster_algo'], config['match_threshold'])

#     # quick summary of the clustering results
#     report_cluster_results(cluster_ids)

#     # we want to subdivide the clusters by encounter for easier viewing
#     encounter_path = os.path.join(root, 'encounter_info.csv')
#     encounter_info = pd.read_csv(encounter_path)

#     # create a dataframe proposed id and encounter for each image
#     cluster_df = pd.DataFrame({'image': fnames, 'autosort_id': cluster_ids})
#     cluster_df = cluster_df.merge(encounter_info)

#     # sort the images into folders based on the proposed individuals
#     sort_images(cluster_df, root)

#     shutil.rmtree(os.path.join(root, 'tmp'))

def load_features(image_root: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load features from disk."""

    # the features are stored in a dictionary with image names as keys
    feature_path = os.path.join(image_root, 'features', 'features.npy')
    feature_dict = np.load(feature_path, allow_pickle=True).item()

    # unpack the dictionary into arrays
    image_names = np.array(list(feature_dict.keys()))
    feature_array = np.array(list(feature_dict.values()))

    return image_names, feature_array

def cluster_images(
        features: np.ndarray, 
        algo: str, 
        match_threshold: float
    ):

    lab = 'Hierachical Clustering' if algo == 'hac' else 'Network Clustering'
    print(f'Clustering {features.shape[0]} features with {lab}.')

    if algo == 'hac':
        results = cluster_hac(features, match_threshold)
    elif algo == 'network':
        results = cluster_network(features, match_threshold)
    else:
        raise ValueError(f'Unknown clustering algorithm: {algo}')
    
    return results

def cluster_hac(feature_layer: np.ndarray, match_threshold: float) -> np.ndarray:
    """Cluster features using hierarchical agglomerative clustering."""

    distance_threshold = 1 - match_threshold

    # single linkage is necessary when using cosine distance
    hac = AgglomerativeClustering(
        distance_threshold=distance_threshold, 
        n_clusters=None, metric='cosine', 
        linkage='complete'
    ).fit(feature_layer)

    cluster_labels = hac.labels_

    return cluster_labels

# def cluster_network_old(features: np.ndarray, threshold: float) -> np.ndarray:

#     # similarity = 1 - pairwise_distances(features, metric='cosine')
#     # matches = np.where(similarity > threshold, similarity, 0)

#     distance = pairwise_distances(features, metric='cosine')
#     matches = (distance < threshold) * 1

#     # Get connected components from the graph
#     nx_graph = nx.from_numpy_array(matches)
#     connected_components = list(nx.connected_components(nx_graph))

#     # Create a mapping from node index to cluster index
#     file_count, _ = features.shape 
#     cluster_labels = np.zeros(file_count, dtype=int)
#     for cluster_idx, cluster in enumerate(connected_components):
#         for node in cluster:
#             cluster_labels[node] = cluster_idx

#     return cluster_labels
class ClusterResults:
    def __init__(self, cluster_labels):
        self.cluster_labels = cluster_labels
        self.cluster_idx = [None]
        self.filenames = None
        self.cluster_count = len(set(cluster_labels))
        self.cluster_sizes = Counter(cluster_labels).values()
        self.false_positive_df = None  # type: Optional[pd.DataFrame]
        self.graph = nx.Graph()  # Initialize with empty graph instead of None
        self.bad_clusters = []
        self.bad_cluster_idx = []

    def plot_suspicious(self):
        graph = self.graph
        # Get connected components from the graph
        if graph is None or graph.number_of_nodes() == 0:
            print("No graph data available to plot suspicious connections.")
            return
        connected_components = [graph.subgraph(c) for c 
                                in nx.connected_components(self.graph)]

        subplot_count = len(self.bad_clusters)
        n_col = 5
        n_row = int(np.ceil(subplot_count / n_col))
        width = 1.5
        height = 1.5

        fig, axes = plt.subplots(n_row, n_col, tight_layout=True,
                                figsize=(n_col * width, n_row * height))
        flat = axes.flatten()

        for i, idx in enumerate(self.bad_cluster_idx):
            
            ax = flat[i]

            # remove self loops
            G = connected_components[idx].copy()
            G.remove_edges_from(nx.selfloop_edges(G))

            # modularity is the warning sign for a bad cluster
            community = nx.community.louvain_communities(G) # pyright: ignore[reportAttributeAccessIssue]

            layout = nx.spring_layout(G)
            nx.draw_networkx_edges(G, pos=layout, ax=ax, edge_color='C7', 
                                   alpha=0.3)
            # color each node based on the louvain_communities
            community = nx.community.louvain_communities(G) # pyright: ignore[reportAttributeAccessIssue]
            color_map = {}
            for idx, comm in enumerate(community):
                for node in comm:
                    color_map[node] = idx
            node_colors = [color_map[node] for node in G.nodes]
            nx.draw_networkx_nodes(G, layout, node_size=20, edgecolors='k',
                                   node_color=node_colors, cmap='tab10', ax=ax) # pyright: ignore[reportArgumentType]

            label = self.bad_clusters[i]
            ax.set_title(label, fontsize=10, loc='center')

        # delete unused axes
        for idx in range(subplot_count, len(flat)):
            fig.delaxes(flat[idx])

        s = 'Matches between images\nSingle links between clusters are suspicious'
        fig.suptitle(s, fontsize=12)

        plt.tight_layout()
        plt.show()

def cluster_network(features, match_threshold):

    MODULARITY_THRESHOLD = 0.3

    similarity = 1 - pairwise_distances(features, metric='cosine')
    matches = (similarity > match_threshold) 
    # matches = np.where(distance < distance_threshold, distance, 0)

    # Get connected components from the graph
    G = nx.from_numpy_array(matches)
    connected_components = (G.subgraph(c) for c in nx.connected_components(G))

    # Create a mapping from node index to cluster index
    file_count, _ = features.shape 
    cluster_labels = np.empty(file_count, dtype=object)
    cluster_indices = np.empty(file_count, dtype=int)

    # Assign clusters to the cluster_labels array
    df_list = []
    bad_clusters = []
    bad_cluster_idx = []
    for cluster_idx, subgraph in enumerate(connected_components):

        cluster_label = f'ID_{cluster_idx:04d}'
        for node in subgraph:
            cluster_labels[node] = cluster_label
            cluster_indices[node] = cluster_idx

        # modularity is the warning sign for a bad cluster
        community = nx.community.louvain_communities(subgraph) # type: ignore
        modularity = nx.community.quality.modularity(subgraph, community) # pyright: ignore[reportAttributeAccessIssue]

        if modularity > MODULARITY_THRESHOLD:
            bad_clusters.append(cluster_label)
            bad_cluster_idx.append(cluster_idx)
            
        for community_idx, comm in enumerate(community):
            for node in comm:
                row = pd.DataFrame({
                    'cluster_id': [cluster_label],
                    'modularity': modularity,
                    # 'filename': fnames[node],
                    'community': community_idx
                })
                df_list.append(row)

    if bad_clusters:
        w = f'Following clusters may contain false positives:\n{bad_clusters}'
        print(w)

    df = pd.concat(df_list, ignore_index=True)

    results = ClusterResults(cluster_labels)
    # results.filenames = fnames
    results.graph = G
    results.false_positive_df = df
    results.bad_clusters = bad_clusters
    results.bad_cluster_idx = bad_cluster_idx
    results.cluster_idx = format_ids(cluster_indices)

    return results

def format_ids(ids: np.ndarray) -> List:
    return [f'ID-{i:04d}' for i in ids]

def report_cluster_results(cluster_labs: np.ndarray) -> None:

    # quick summary of the cluster_labs results
    label, count = np.unique(cluster_labs, return_counts=True)
    print(f'Found {len(label)} clusters.')
    print(f'Largest cluster has {np.max(count)} images.')

def sort_images(id_df, input_dir: str, output_dir: str) -> None:
    """Sort images into folders based on cluster and encounter."""

    if not os.path.isdir(input_dir):
        raise ValueError('input_dir', input_dir, 'is not a valid directory')

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    grouped = id_df.groupby(['autosort_id', 'encounter'])
    i = 0
    j = 0
    for (clust_id, enc_id), mini_df in grouped:

        i += 1
        cluster_dir = os.path.join(output_dir, clust_id)
        os.makedirs(cluster_dir, exist_ok=True)

        encounter_dir = os.path.join(cluster_dir, enc_id)
        os.makedirs(encounter_dir, exist_ok=True)

        for img in mini_df['image']:
            j += 1
            old_path = os.path.join(input_dir, img)
            shutil.copy(old_path, encounter_dir)
        
    print(f'Sorted {j} images into {i} folders.')