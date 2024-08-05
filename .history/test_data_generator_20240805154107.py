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
from nltk.util import bigrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.metrics.distance import jaccard_distance
from numba import njit

import more_itertools
import time

@njit(cache=True)

def custom_distance(pointA, pointB):
    return np.sum((np.maximum(pointA, pointB) - np.minimum(pointA, pointB)) <= 1)

def jaccard_distance_words(ngrams1, ngrams2):   

    unique_ngrams = list(ngrams1.union(ngrams2))

    vec1 = [int(ng in ngrams1) for ng in unique_ngrams]
    vec2 = [int(ng in ngrams2) for ng in unique_ngrams]

    distance = cdist([vec1], [vec2], metric='jaccard')[0][0]

    return distance

metric = 'cosine'

method = 'edit_distance' 

total_test_rows = [5000]

refrows = 2000
chunk_size = 0
round_digits = 3

dfA_original_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None)
dfA_original_test = dfA_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfB_original_test = pd.read_csv('Data/BIASB_200000.csv', delimiter="|", header=None)
dfB_original_test = dfB_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_original_test = dfA_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_original_test = dfB_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

ref = pd.read_csv('clean_ref.csv', names=['name'], nrows=refrows)

ref['ParsedName'] = ref['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref["FirstName"] = ref["ParsedName"].apply(lambda x: x.first)

ref["LastName"] = ref["ParsedName"].apply(lambda x: x.last)

ref = ref.drop(['ParsedName', 'name'], axis=1)

for part in range(3,4):
    for test_rows in total_test_rows:
        test_indices = pd.read_csv('test_indices_'+str(part)+'.csv', header=None)[0].tolist()[:test_rows]
        print("len test_indices = ",len(test_indices))
        if isinstance(metric, str):
            filename = '/media/mike/corsair/correct_data/new_dataset/only_tests/round3_n'+str(test_rows)+'_ref'+str(refrows)+'_'+str(part)+'.csv'

        elif callable(metric):
            filename = '/media/mike/corsair/correct_data/new_dataset/only_tests/jacc_part'+str(part)+'_clean_ref_'+ str(metric.__name__)+'_n'+str(test_rows)+'_ref'+str(refrows)+'.csv'

        print(filename)

        dfA_test = dfA_original_test.loc[test_indices]

        print(dfA_test.shape)

        dfB_test = dfB_original_test.loc[test_indices]

        print(dfB_test.shape)

        print('Test Data: test_rows =',len(dfA_test), 'ref =',len(ref))
        print(len(dfA_test), len(dfB_test), len(ref))
        start_time = time.time()
        if (method=='edit_distance'):

            distances_AtoRef_names = []
            distances_AtoRef_lastnames = []
            distances_AtoRef_middlenames_first = []
            distances_AtoRef_middlenames_last = []

            for nameA in dfA_test.itertuples(index=False):

                distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])

                distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

                distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            chunk_size = 0
            if (chunk_size == 0):
                distances_AtoRef_names = np.array(distances_AtoRef_names)

            elif (chunk_size != 0):
                raise RuntimeError("XAXAXAXA OLA APO THN ARXH")
                total = []
                for vector in distances_AtoRef_names:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_AtoRef_names = np.array(total)

            if (chunk_size == 0):
                distances_AtoRef_lastnames = np.array(distances_AtoRef_lastnames)
            elif (chunk_size != 0):
                total = []
                for vector in distances_AtoRef_lastnames:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_AtoRef_lastnames = np.array(total)

            if (chunk_size == 0):
                distances_AtoRef_middlenames_first = np.array(distances_AtoRef_middlenames_first)
            elif (chunk_size != 0):
                total = []
                for vector in distances_AtoRef_middlenames_first:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_AtoRef_middlenames_first = np.array(total)

            if (chunk_size == 0):
                distances_AtoRef_middlenames_last = np.array(distances_AtoRef_middlenames_last)
            elif (chunk_size != 0):
                total = []
                for vector in distances_AtoRef_middlenames_last:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_AtoRef_middlenames_last = np.array(total)

            distances_BtoRef_names = []
            distances_BtoRef_lastnames = []
            distances_BtoRef_middlenames_first = []
            distances_BtoRef_middlenames_last = []

            for nameB in dfB_test.itertuples(index=False):

                distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])

                distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])

                distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            if (chunk_size == 0):
                distances_BtoRef_names = np.array(distances_BtoRef_names)
            elif (chunk_size != 0):
                total = []
                for vector in distances_BtoRef_names:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_BtoRef_names = np.array(total)

            if (chunk_size == 0):
                distances_BtoRef_lastnames = np.array(distances_BtoRef_lastnames)
            elif (chunk_size != 0):
                total = []
                for vector in distances_BtoRef_lastnames:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_BtoRef_lastnames = np.array(total)

            if (chunk_size == 0):
                distances_BtoRef_middlenames_first = np.array(distances_BtoRef_middlenames_first)
            elif (chunk_size != 0):
                total = []
                for vector in distances_BtoRef_middlenames_first:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_BtoRef_middlenames_first = np.array(total)

            if (chunk_size == 0):
                distances_BtoRef_middlenames_last = np.array(distances_BtoRef_middlenames_last)
            elif (chunk_size != 0):
                total = []
                for vector in distances_BtoRef_middlenames_last:
                    chunk_sum = []
                    for chunk in more_itertools.chunked(vector, chunk_size):
                        chunk_sum.append(sum(chunk))
                    total.append(chunk_sum)
                distances_BtoRef_middlenames_last = np.array(total)

        if (method=='jaccard'):

            ref_FirstName_bigrams = ref.FirstName.apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))
            ref_LastName_bigrams = ref.LastName.apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))

            dfA_FirstName_bigrams = dfA_test.FirstName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))
            dfA_MiddleName_bigrams = dfA_test.MiddleName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))
            dfA_LastName_bigrams = dfA_test.LastName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))

            dfB_FirstName_bigrams = dfB_test.FirstName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))
            dfB_MiddleName_bigrams = dfB_test.MiddleName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))
            dfB_LastName_bigrams = dfB_test.LastName.str.upper().apply(lambda x: set(bigrams(pad_both_ends(x,n=2))))

            distances_AtoRef_names = []
            distances_AtoRef_lastnames = []
            distances_AtoRef_middlenames_first = []
            distances_AtoRef_middlenames_last = []

            distances_AtoRef_names = [[jaccard_distance(set1, set2) for set2 in ref_FirstName_bigrams] for set1 in dfA_FirstName_bigrams]
            distances_AtoRef_lastnames = [[jaccard_distance(set1, set2) for set2 in ref_LastName_bigrams] for set1 in dfA_LastName_bigrams]
            distances_AtoRef_middlenames_first = [[jaccard_distance(set1, set2) for set2 in ref_FirstName_bigrams] for set1 in dfA_MiddleName_bigrams]
            distances_AtoRef_middlenames_last = [[jaccard_distance(set1, set2) for set2 in ref_LastName_bigrams] for set1 in dfA_MiddleName_bigrams]

            distances_AtoRef_names = np.array(distances_AtoRef_names)
            distances_AtoRef_lastnames = np.array(distances_AtoRef_lastnames)
            distances_AtoRef_middlenames_first = np.array(distances_AtoRef_middlenames_first)
            distances_AtoRef_middlenames_last = np.array(distances_AtoRef_middlenames_last)

            distances_BtoRef_names = []
            distances_BtoRef_lastnames = []
            distances_BtoRef_middlenames_first = []
            distances_BtoRef_middlenames_last = []

            distances_BtoRef_names = [[jaccard_distance(set1, set2) for set2 in ref_FirstName_bigrams] for set1 in dfB_FirstName_bigrams]
            distances_BtoRef_lastnames = [[jaccard_distance(set1, set2) for set2 in ref_LastName_bigrams] for set1 in dfB_LastName_bigrams]
            distances_BtoRef_middlenames_first = [[jaccard_distance(set1, set2) for set2 in ref_FirstName_bigrams] for set1 in dfB_MiddleName_bigrams]
            distances_BtoRef_middlenames_last = [[jaccard_distance(set1, set2) for set2 in ref_LastName_bigrams] for set1 in dfB_MiddleName_bigrams]

            distances_BtoRef_names = np.array(distances_BtoRef_names)
            distances_BtoRef_lastnames = np.array(distances_BtoRef_lastnames)
            distances_BtoRef_middlenames_first = np.array(distances_BtoRef_middlenames_first)
            distances_BtoRef_middlenames_last = np.array(distances_BtoRef_middlenames_last)

        chunk_size = 1000

        num_chunks = len(distances_AtoRef_names) // chunk_size + 1

        print(num_chunks)

        with open(filename, 'w') as f:
            for i in tqdm(range (num_chunks)):

                    start_i = i * chunk_size
                    end_i = (i + 1) * chunk_size

                    if end_i > test_rows:
                        continue

                    for j in tqdm(range (num_chunks)):

                        start_j = j * chunk_size
                        end_j = (j + 1) * chunk_size
                        if end_j > test_rows:

                            continue

                        distances_AtoRef_names_dask = da.from_array(distances_AtoRef_names[start_i:end_i])
                        distances_BtoRef_names_dask = da.from_array(distances_BtoRef_names[start_j:end_j])
                        distances_AtoRef_lastnames_dask = da.from_array(distances_AtoRef_lastnames[start_i:end_i])
                        distances_BtoRef_lastnames_dask = da.from_array(distances_BtoRef_lastnames[start_j:end_j])
                        distances_AtoRef_middlenames_first_dask = da.from_array(distances_AtoRef_middlenames_first[start_i:end_i])
                        distances_BtoRef_middlenames_first_dask = da.from_array(distances_BtoRef_middlenames_first[start_j:end_j])
                        distances_AtoRef_middlenames_last_dask = da.from_array(distances_AtoRef_middlenames_last[start_i:end_i])
                        distances_BtoRef_middlenames_last_dask = da.from_array(distances_BtoRef_middlenames_last[start_j:end_j])

                        distances_AtoRef_names_dask = da.from_array(distances_AtoRef_names[start_i:end_i])

                        distances_names = da.map_blocks(lambda x, y: cp.around(cdist(x, y, metric),round_digits), distances_AtoRef_names_dask, distances_BtoRef_names_dask).compute()
                        distances_names = cp.asarray(distances_names)

                        distances_lastnames = da.map_blocks(lambda x, y: cp.around(cdist(x, y, metric),round_digits), distances_AtoRef_lastnames_dask, distances_BtoRef_lastnames_dask).compute()
                        distances_lastnames = cp.asarray(distances_lastnames)

                        distances_middlenames_first = da.map_blocks(lambda x, y: cp.around(cdist(x, y, metric),round_digits), distances_AtoRef_middlenames_first_dask, distances_BtoRef_middlenames_first_dask).compute()
                        distances_middlenames_first = cp.asarray(distances_middlenames_first)

                        distances_middlenames_last = da.map_blocks(lambda x, y: cp.around(cdist(x, y, metric),round_digits), distances_AtoRef_middlenames_last_dask, distances_BtoRef_middlenames_last_dask).compute()
                        distances_middlenames_last = cp.asarray(distances_middlenames_last)

                        distances = cp.column_stack((cp.asarray(distances_names).ravel(), cp.asarray(distances_lastnames).ravel(), cp.asarray(distances_middlenames_first).ravel(), cp.asarray(distances_middlenames_last).ravel()))

                        comparisons_with_labels = cudf.DataFrame((dfA_test['id'].values[start_i:end_i, None] == dfB_test['id'].values[start_j:end_j]).astype(int).ravel(), columns=['label'])

                        distances_with_labels = cp.column_stack([cp.asarray(comparisons_with_labels['label']), cp.asarray(distances)])

                        dataAB = cudf.DataFrame(distances_with_labels)

                        dataAB.to_csv(f, index=False, header=None)
        end_time = time.time()
        total_time = end_time - start_time  
        print(f"Total time for part {part}, test_rows {test_rows}: {total_time:.2f} seconds")   

