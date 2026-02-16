"""Compare unidentified individuals in a query set with a reference set of known IDs."""

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


def predict_ids(reference_dict, query_dict, id_df, proposed_id_count=10):
    """Predict the identities of individuals in the query set

    Return a DataFrame of the most `proposed_id_count` most similar individuals
    in the reference set for each query image, along with their cosine 
    similarity. 

    Parameters
    ----------
    reference_dict : dict
        Dictionary where the key is the reference image's name. The value 
        associate with each key is a NumPy array of shape (M, ) where M is the 
        number of features in the feature vector.
    query_dict : dict
        Dictionary where the key is the query image's name. The value 
        associate with each key is a NumPy array of shape (M, ) where M is the 
        number of features in the feature vector.
    id_df : pd.DataFrame
        DataFrame containing the identities, `individual_id`, and image file 
        name, `image`, for every image in the reference set.
    proposed_id_count : integer
        The number of proposed IDs to return for each query image.
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyseter.identify import predict_ids
    >>> 
    >>> ref_dict = {'image1': np.array([0.1, 0.11])}
    >>> query_dict = {'image2': np.array([0.1, 0.12])}
    >>> id_df = pd.DataFrame({'image': 'image1', 'individual_id': 'a'})
    >>> 
    >>> results = predict_ids(ref_dict, query_dict, id_df, proposed_id_count=1)
    >>> len(results)
    1

    """

    # unpack the dictionaries 
    reference_files = np.array(list(reference_dict.keys()))
    reference_feats = np.array(list(reference_dict.values()))   

    query_files = np.array(list(query_dict.keys()))
    query_feats = np.array(list(query_dict.values()))  

    # this is the true id of every id in the reference dataset
    ids = id_df.set_index('image').loc[reference_files, 'individual_id'].to_numpy()

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
    pred_df['rank'] = pred_df['rank'] + 1

    return pred_df

def update_reference_features(reference_dict, query_dict, confirmed_id_df):
    """Update the reference feature dict with confirmed matches

    Return a dictionary where the keys are the updated reference images and the
    values contain the updated feature array.

    Parameters
    ----------
    reference_dict : dict
        Dictionary where the key is the reference image's name. The value 
        associate with each key is a NumPy array of shape (M, ) where M is the 
        number of features in the feature vector.
    query_dict : dict
        Dictionary where the key is the query image's name. The value 
        associate with each key is a NumPy array of shape (M, ) where M is the 
        number of features in the feature vector.
    id_df : pd.DataFrame
        DataFrame containing the identities, `individual_id`, and image file 
        name, `image`, for every image in the reference set.
    proposed_id_count : integer
        The number of proposed IDs to return for each query image.
    """
    
    # reorder the query_dict to make it match the order of the confirmed_match_df
    confirmed_images = confirmed_id_df.image
    confirmed_feature_dict = {i: query_dict[i] for i in confirmed_images}

    # unpack both dicts 
    reference_features = np.array(list(reference_dict.values()))
    confirmed_features = np.array(list(confirmed_feature_dict.values()))

    # stack the reference features
    updated_features = np.vstack((reference_features, confirmed_features))

    # new dict
    reference_images = np.array(list(reference_dict.keys()))
    updated_images = np.concatenate((reference_images, confirmed_images))
    updated_dict = dict(zip(updated_images, updated_features))

    return updated_dict