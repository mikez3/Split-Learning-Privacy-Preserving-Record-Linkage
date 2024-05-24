
# %%

# Import libs
import pandas as pd
import numpy as np
import cupy as cp
from nameparser import HumanName
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist
import cudf
from tqdm import tqdm
from cupyx.scipy.spatial.distance import cdist as cucdist
import dask.array as da
import sys

import json

# Convert the command line argument from a string to a list of integers
json_filename = sys.argv[1]
with open(json_filename, 'r') as f:
    test_indices = json.load(f)

def jaccard_distance_words(ngrams1, ngrams2):   
    # Convert the sets of n-grams into binary vectors
    unique_ngrams = list(ngrams1.union(ngrams2))

    vec1 = [int(ng in ngrams1) for ng in unique_ngrams]
    vec2 = [int(ng in ngrams2) for ng in unique_ngrams]
    # print(vec1, vec2)
    
    # Calculate Jaccard distance using cdist
    distance = cdist([vec1], [vec2], metric='jaccard')[0][0]
    # distance = cucdist([vec1], [vec2], metric='jaccard')[0][0]
    # distance = cdist(vec1, vec2, metric='jaccard')
    
    return distance
    # return 0

method = 'edit_distance' 
metric = 'cosine'
# euclidean,cityblock,seuclidean,sqeuclidean,cosine,correlation,hamming,jaccard,jensenshannon,chebyshev,canberra,braycurtis,-NOT-mahalanobis, custom_distance

# Read csvs
print(len(test_indices))
total_test_rows = [215, 858, 2000, 5000, 10000]
refrows = int(sys.argv[2])
part = int(sys.argv[3])
dfA_original_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None, nrows=12000)
dfA_original_test = dfA_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfB_original_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None, nrows=12000)
dfB_original_test = dfB_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_original_test = dfA_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_original_test = dfB_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

ref = pd.read_csv('clean_ref.csv', names=['name'], nrows=refrows)

# Extract the first and last names from the parsed names
ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)
ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)
ref = ref.drop(['ParsedName', 'name'], axis=1)

for test_rows in total_test_rows:
    if isinstance(metric, str):
        filename = '/Data/test_samples/part'+str(part+1)+'_'+metric+'_n'+str(test_rows)+'_ref'+str(refrows)+'.csv'
    elif callable(metric):
        filename = '/Data/test_samples/part'+str(part+1)+'_'+ str(metric.__name__)+'_n'+str(test_rows)+'_ref'+str(refrows)+'.csv'

    # Select the rows specified by test_indices
    dfA_test = dfA_original_test.loc[test_indices]
    dfA_test = dfA_original_test.iloc[:test_rows]
    # Same for dfB
    dfB_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None, nrows=12000)
    dfB_test = dfB_original_test.loc[test_indices]
    dfB_test = dfB_original_test.iloc[:test_rows]

    print('Test Data: test_rows =',len(dfA_test), 'ref =',len(ref))
    print(len(dfA_test), len(dfB_test), len(ref))

    if (method=='edit_distance'):
        # A-B
        # Α-ref
        distances_AtoRef_names = []
        distances_AtoRef_lastnames = []
        distances_AtoRef_middlenames_first = []
        distances_AtoRef_middlenames_last = []

        for nameA in dfA_test.itertuples(index=False):
            # FirstName A - FirstName ref
            distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
            # LastName A - LastName ref
            distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

            # MiddleName A - Firstname ref
            distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

            # MiddleName A - Lastname ref
            distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])


        distances_AtoRef_names = np.array(distances_AtoRef_names)
        distances_AtoRef_lastnames = np.array(distances_AtoRef_lastnames)
        distances_AtoRef_middlenames_first = np.array(distances_AtoRef_middlenames_first)
        distances_AtoRef_middlenames_last = np.array(distances_AtoRef_middlenames_last)

        # B-ref
        edit_distances_BtoRef_names = []
        edit_distances_BtoRef_lastnames = []
        edit_distances_BtoRef_middlenames_first = []
        edit_distances_BtoRef_middlenames_last = []

        for nameB in dfB_test.itertuples(index=False):
        # for nameB in dfB.itertuples(index=False):
            # FirstName B - FirstName ref
            edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
            # LastName B - LastName ref
            edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
            # MiddleName B - FirstName ref
            edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

            # MiddleName B - Lastname ref
            edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

        distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
        distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
        distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
        distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

    chunk_size = round(test_rows/2)
    if (test_rows < 1000):
        chunk_size = 200
    num_chunks = len(distances_AtoRef_names) // chunk_size + 1
    with open(filename, 'w') as f:
        for i in tqdm(range (num_chunks)):
                start_i = i * chunk_size
                end_i = (i + 1) * chunk_size
                for j in tqdm(range (num_chunks)):
                    start_j = j * chunk_size
                    end_j = (j + 1) * chunk_size
                    if end_j > test_rows:
                        continue
                    # -------------------------------------
                    # Convert arrays to Dask arrays
                    distances_AtoRef_names_dask = da.from_array(distances_AtoRef_names[start_i:end_i])
                    distances_BtoRef_names_dask = da.from_array(distances_BtoRef_names[start_j:end_j])
                    distances_AtoRef_lastnames_dask = da.from_array(distances_AtoRef_lastnames[start_i:end_i])
                    distances_BtoRef_lastnames_dask = da.from_array(distances_BtoRef_lastnames[start_j:end_j])
                    distances_AtoRef_middlenames_first_dask = da.from_array(distances_AtoRef_middlenames_first[start_i:end_i])
                    distances_BtoRef_middlenames_first_dask = da.from_array(distances_BtoRef_middlenames_first[start_j:end_j])
                    distances_AtoRef_middlenames_last_dask = da.from_array(distances_AtoRef_middlenames_last[start_i:end_i])
                    distances_BtoRef_middlenames_last_dask = da.from_array(distances_BtoRef_middlenames_last[start_j:end_j])
                    
                    distances_AtoRef_names_dask = da.from_array(distances_AtoRef_names[start_i:end_i])
                    distances_names = da.map_blocks(lambda x, y: cdist(x, y, metric), distances_AtoRef_names_dask, distances_BtoRef_names_dask).compute()
                    distances_names = cp.asarray(distances_names)

                    distances_lastnames = da.map_blocks(lambda x, y: cdist(x, y, metric), distances_AtoRef_lastnames_dask, distances_BtoRef_lastnames_dask).compute()
                    distances_lastnames = cp.asarray(distances_lastnames)

                    distances_middlenames_first = da.map_blocks(lambda x, y: cdist(x, y, metric), distances_AtoRef_middlenames_first_dask, distances_BtoRef_middlenames_first_dask).compute()
                    distances_middlenames_first = cp.asarray(distances_middlenames_first)

                    distances_middlenames_last = da.map_blocks(lambda x, y: cdist(x, y, metric), distances_AtoRef_middlenames_last_dask, distances_BtoRef_middlenames_last_dask).compute()
                    distances_middlenames_last = cp.asarray(distances_middlenames_last)
                    # _-------------------------------------
                    distances = cp.column_stack((cp.asarray(distances_names).ravel(), cp.asarray(distances_lastnames).ravel(), cp.asarray(distances_middlenames_first).ravel(), cp.asarray(distances_middlenames_last).ravel()))
                    comparisons_with_labels = cudf.DataFrame((dfA_test['id'].values[start_i:end_i, None] == dfB_test['id'].values[start_j:end_j]).astype(int).ravel(), columns=['label'])
                    # save
                    distances_with_labels = cp.column_stack([cp.asarray(comparisons_with_labels['label']), cp.asarray(distances)])
                    dataAB = cudf.DataFrame(distances_with_labels)
                    dataAB.to_csv(f, index=False, header=None)