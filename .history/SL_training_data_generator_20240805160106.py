import pandas as pd
import numpy as np

from nameparser import HumanName
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist
import os, cudf
import sys
import subprocess
import json
import time
import gc
import more_itertools

from numba import njit

@njit
def custom_distance(pointA, pointB):
    return np.sum((np.maximum(pointA, pointB) - np.minimum(pointA, pointB)) <= 1)
def jaccard_distance_words(ngrams1, ngrams2):   

    unique_ngrams = list(ngrams1.union(ngrams2))

    vec1 = [int(ng in ngrams1) for ng in unique_ngrams]
    vec2 = [int(ng in ngrams2) for ng in unique_ngrams]

    distance = cdist([vec1], [vec2], metric='jaccard')[0][0]

    return distance
method = "edit_distance"
metric = 'cosine'

ref_original = pd.read_csv('clean_ref.csv', names=['name'])
ref_original['ParsedName'] = ref_original['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref_original["FirstName"] = ref_original["ParsedName"].apply(lambda x: x.first)
ref_original["LastName"] = ref_original["ParsedName"].apply(lambda x: x.last)
ref_original = ref_original.drop(['ParsedName', 'name'], axis=1)

dfA_original = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None)
dfA_original = dfA_original.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfB_original = pd.read_csv('Data/BIASB_200000.csv', delimiter="|", header=None)
dfB_original = dfB_original.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_original = dfA_original[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_original = dfB_original[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

total_rows = [2000]
total_refrows = [2000]

seed = 42
np.random.seed(seed)

for part in range(3,4):
    indices = pd.read_csv('train_indices_'+str(part)+'.csv', header=None)[0].tolist()
    shuffled_indices = np.random.permutation(indices)
    indicesA = shuffled_indices[:total_rows[0]].tolist()
    indicesB = shuffled_indices[total_rows[0]:(2*total_rows[0])].tolist()

    for nrows in total_rows:
        for refrows in total_refrows:
            start_time = time.time()
            ref = ref_original[:refrows]
            if isinstance(metric, str):
                csv_name = 'part'+str(part)+'_clean_ref_'+metric+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
            elif callable(metric):
                csv_name = 'part'+str(part)+'_clean_ref_'+str(metric.__name__)+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
            print(csv_name)

            dfA = dfA_original.loc[indicesA]
            print('Training Data:', 'nrows =', len(dfA), ', refrows =', len(ref))
            dfA2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])

            dfA2['id'] = dfA['id']
            dfA2['FirstName'] = dfA['FirstName']
            dfA2['LastName'] = dfA['LastName'] + '!'
            dfA2['MiddleName'] = dfA['MiddleName']

            edit_distances_AtoRef_names = []
            edit_distances_AtoRef_lastnames = []
            edit_distances_AtoRef_middlenames_first = []
            edit_distances_AtoRef_middlenames_last = []

            for nameA in dfA.itertuples(index=False):
                edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
            edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
            edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
            edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)
            print(len(edit_distances_AtoRef_names))

            edit_distances_BtoRef_names = []
            edit_distances_BtoRef_lastnames = []
            edit_distances_BtoRef_middlenames_first = []
            edit_distances_BtoRef_middlenames_last = []

            for nameB in dfA2.itertuples(index=False):
                edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
            edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
            edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
            edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

            distances = np.column_stack((np.around(np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric)),3).ravel(), np.around(np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric)),3).ravel()))
            print(distances.shape)
            edit_distances_AtoRef_names = edit_distances_BtoRef_names = edit_distances_AtoRef_lastnames = edit_distances_BtoRef_lastnames = None
            del edit_distances_AtoRef_names , edit_distances_BtoRef_names , edit_distances_AtoRef_lastnames , edit_distances_BtoRef_lastnames
            distances = np.column_stack((distances, np.around(np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric)),3).ravel()))
            print(distances.shape)

            edit_distances_AtoRef_middlenames_first = edit_distances_BtoRef_middlenames_first = None
            del edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first
            distances = np.column_stack((distances, np.around(np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric)),3).ravel()))
            print(distances.shape)

            edit_distances_AtoRef_middlenames_last = edit_distances_BtoRef_middlenames_last = None
            del edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last

            if np.isnan(distances).any():
                print("Result contains NaN values, stopping the run")
                nan_count = np.sum(np.isnan(distances))
                print(f"Result contains {nan_count} NaN values")
                time.sleep(5)
                sys.exit()

            comparisons_with_labels = pd.DataFrame((dfA['id'].values[:, None] == dfA2['id'].values).astype(int).ravel(), columns=['label'])
            distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
            print(distances_with_labels.dtype)
            distances = None
            del distances

            del globals()['comparisons_with_labels']

            trainA = cudf.DataFrame(distances_with_labels)
            distances_with_labels = None
            del distances_with_labels

            trainA.to_csv('Data/trainingA_'+csv_name, index=False, header=None, chunksize=50000)

            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)
            print(f"Elapsed time to create FL training data: {elapsed_time} seconds")
            trainA = None
            del trainA

            start_time = time.time()
            dfB = dfB_original.loc[indicesB]
            dfB2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])

            dfB2['id'] = dfB['id']
            dfB2['FirstName'] = dfB['FirstName']
            dfB2['LastName'] = dfB['LastName'] + '!'
            dfB2['MiddleName'] = dfB['MiddleName']

            edit_distances_AtoRef_names = []
            edit_distances_AtoRef_lastnames = []
            edit_distances_AtoRef_middlenames_first = []
            edit_distances_AtoRef_middlenames_last = []

            for nameA in dfB.itertuples(index=False):
                edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
            edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
            edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
            edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)

            edit_distances_BtoRef_names = []
            edit_distances_BtoRef_lastnames = []
            edit_distances_BtoRef_middlenames_first = []
            edit_distances_BtoRef_middlenames_last = []

            for nameB in dfB2.itertuples(index=False):
                edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])
                edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
            edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
            edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
            edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

            distances = np.column_stack((np.around(np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric)),3).ravel(), np.around(np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric)),3).ravel()))
            print(distances.shape)

            edit_distances_AtoRef_names = edit_distances_BtoRef_names = edit_distances_AtoRef_lastnames = edit_distances_BtoRef_lastnames = None
            del edit_distances_AtoRef_names , edit_distances_BtoRef_names , edit_distances_AtoRef_lastnames , edit_distances_BtoRef_lastnames
            distances = np.column_stack((distances, np.around(np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric)),3).ravel()))
            print(distances.shape)

            edit_distances_AtoRef_middlenames_first = edit_distances_BtoRef_middlenames_first = None
            del edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first
            distances = np.column_stack((distances, np.around(np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric)),3).ravel()))
            print(distances.shape)

            edit_distances_AtoRef_middlenames_last = edit_distances_BtoRef_middlenames_last = None
            del edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last

            if np.isnan(distances).any():
                print("Result contains NaN values, stopping the run")
                nan_count = np.sum(np.isnan(distances))
                print(f"Result contains {nan_count} NaN values")
                time.sleep(5)
                sys.exit()

            comparisons_with_labels = pd.DataFrame((dfB['id'].values[:, None] == dfB2['id'].values).astype(int).ravel(), columns=['label'])
            distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
            distances = None
            del distances

            trainA = cudf.DataFrame(distances_with_labels)
            distances_with_labels = None
            del distances_with_labels

            trainA.to_csv('/media/mike/corsair/correct_data/new_dataset/trainingB_'+csv_name, index=False, header=None, chunksize=50000)
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)
            print(f"Elapsed time to create FL training data: {elapsed_time} seconds")
            trainA = dfB2 = None
            del trainA, dfB2

