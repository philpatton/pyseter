
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def find_neighbors(ref_feat, query_feat):
    """Find the most similar images to the reference set to the query set."""
    neighborhood = NearestNeighbors(n_neighbors=500, metric="cosine")
    neighborhood.fit(ref_feat)

    distances, indices = neighborhood.kneighbors(query_feat)
    return distances, indices

def insert_new_id(distances, predictions, threshold=0.5):
    """Insert new_individual into the predictions at the threshold."""

    # find position where we should slot new individual
    new_id_position = np.array(
        [np.searchsorted(s, threshold) for s in distances]
    )

    # insert "new_individual" into predictions and 0.5 into scores
    new_preds = [np.insert(t, i, 'new_individual') 
                 for t, i in zip(predictions, new_id_position)]
    new_dist = [np.insert(t, i, 0.5) 
                 for t, i in zip(distances, new_id_position)]

    return new_dist, new_preds

def pool_predictions(predictions, distances):
    """Remove redundant predictions, i.e., those of the same individual."""

    # find position of unique entry to eliminate redundant predictions 
    unique_index = [np.unique(a, return_index=True)[1] for a in predictions]

    # convert index to string 
    pooled_pred = [t[np.sort(i)] for t, i in zip(predictions, unique_index)]
    pooled_dist = [t[np.sort(i)] for t, i in zip(distances, unique_index)]

    return pooled_dist, pooled_pred


def identify(reference_dict, query_dict, id_df, proposed_id_count=10, return_scores=True):
    """Identify individuals in the query set."""

    # unpack the dictionaries 
    reference_files = np.array(list(reference_dict.keys()))
    reference_feats = np.array(list(reference_dict.values()))   

    query_files = np.array(list(query_dict.keys()))
    query_feats = np.array(list(query_dict.values()))  

    # this is the true id of every id in the reference dataset
    ids = id_df.set_index('image').loc[reference_files, 'individual_id'].values

    # takes about 19 seconds
    distance_matrix, index_matrix = find_neighbors(reference_feats, query_feats)

    # get the corresponding labels for each reference image
    predicted_ids = ids[index_matrix]

    # insert the prediction "new_individual" at the threshold
    distances, ids = insert_new_id(distance_matrix, predicted_ids, threshold=0.5)

    # remove redundant predictions and take the maximum 
    pooled_distances, pooled_ids = pool_predictions(ids, distances)

    final_predictions = [t[:proposed_id_count].tolist() for t in pooled_ids]
    final_distances = [t[:proposed_id_count].tolist() for t in pooled_distances]

    final_predictions = np.array(final_predictions)
    final_scores = 1 - np.array(final_distances)

    score_df = pd.DataFrame(final_scores, index=query_files)
    score_df = score_df.stack().reset_index()
    score_df.columns = ['image', 'rank', 'score']

    pred_df = pd.DataFrame(final_predictions, index=query_files)
    pred_df = pred_df.stack().reset_index()
    pred_df.columns = ['image', 'rank', 'predicted_id']

    pred_df = pred_df.merge(score_df)

    return pred_df
