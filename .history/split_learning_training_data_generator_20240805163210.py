# Generates TRAIN +TEST data for FL

# %%
# Import libs
import pandas as pd
import numpy as np
# import cupy as cp
from nameparser import HumanName
from rapidfuzz.distance import Levenshtein
from scipy.spatial.distance import cdist
import os, cudf
import sys
import subprocess
import json
import time

# %%
metric = 'cosine'
# euclidean,cityblock,seuclidean,sqeuclidean,cosine,correlation,hamming,jaccard,jensenshannon,chebyshev,canberra,braycurtis,-NOT-mahalanobis, custom_distance

# %%
# Read csvs
ref_original = pd.read_csv('reference_set.csv', names=['name'])
# %%
# Extract the first and last names from the parsed names
ref_original['ParsedName'] = ref_original['name'].str.replace('_', ' ').apply(lambda x: HumanName(x))
ref_original["FirstName"] = ref_original["ParsedName"].apply(lambda x: x.first)
ref_original["LastName"] = ref_original["ParsedName"].apply(lambda x: x.last)
ref_original = ref_original.drop(['ParsedName', 'name'], axis=1)

# Read the file
dfA_original = pd.read_csv('Data/BIASA_200000.csv',nrows=12000, delimiter="|", header=None)
dfA_original = dfA_original.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfB_original = pd.read_csv('Data/BIASB_200000.csv', nrows=12000, delimiter="|", header=None)
dfB_original = dfB_original.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_original = dfA_original[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_original = dfB_original[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

total_rows = [500, 2000]
total_refrows = [200, 2000]
for part in range(3):
    for nrows in total_rows:
        for refrows in total_refrows:
            start_time = time.time()
            ref = ref_original[:refrows]
            if isinstance(metric, str):
                csv_name = 'part'+str(part)+'_reference_set_'+metric+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
            elif callable(metric):
                csv_name = 'part'+str(part)+'_reference_set_'+str(metric.__name__)+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"

            # Shuffle the rows
            dfA = dfA_original.sample(frac=1, random_state=part)
            dfB = dfB_original.loc[dfA.index]

            # Select the first nrows rows and drop unnecessary columns
            dfA = dfA.iloc[:nrows]
            dfB = dfB.iloc[:nrows]
            print('Federated Data:', 'nrows =', len(dfA), ', refrows =', len(ref))
            # %%
            dfA2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])
            dfB2 = pd.DataFrame(columns=['id', 'FirstName', 'LastName', 'MiddleName'])
            # A
            dfA2['id'] = dfA['id']
            dfA2['FirstName'] = dfA['FirstName']
            dfA2['LastName'] = dfA['LastName'] + '!'
            dfA2['MiddleName'] = dfA['MiddleName']

            # B
            dfB2['id'] = dfB['id']
            dfB2['FirstName'] = dfB['FirstName']
            dfB2['LastName'] = dfB['LastName'] + '!'
            dfB2['MiddleName'] = dfB['MiddleName']

            # %%
            # A-A2
            # Α-ref
            edit_distances_AtoRef_names = []
            edit_distances_AtoRef_lastnames = []
            edit_distances_AtoRef_middlenames_first = []
            edit_distances_AtoRef_middlenames_last = []

            for nameA in dfA.itertuples(index=False):
                # Onoma A - Onoma ref
                edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo A - Eponymo ref
                edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

                # MiddleName A - Firstname ref
                edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # MiddleName A - Lastname ref
                edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])


            edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
            edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
            edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
            edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)

            print(len(edit_distances_AtoRef_names))

            # A2-ref
            edit_distances_BtoRef_names = []
            edit_distances_BtoRef_lastnames = []
            edit_distances_BtoRef_middlenames_first = []
            edit_distances_BtoRef_middlenames_last = []
            for nameB in dfA2.itertuples(index=False):
                # Onoma B - Onoma ref
                edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo B - Eponymo ref
                edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                # MiddleName B - FirstName ref
                edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # MiddleName B - Lastname ref
                edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
            edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
            edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
            edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)
            # %%
            distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
            distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
            distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
            distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))

            distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))

            if np.isnan(distances).any():
                print("Result contains NaN values, stopping the run")
                nan_count = np.sum(np.isnan(distances))
                print(f"Result contains {nan_count} NaN values")
                sys.exit()

            comparisons_with_labels = pd.DataFrame((dfA['id'].values[:, None] == dfA2['id'].values).astype(int).ravel(), columns=['label'])
            # save
            distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
            dataA = cudf.DataFrame(distances_with_labels)
            # %%
            # B-B2
            # B-ref
            edit_distances_AtoRef_names = []
            edit_distances_AtoRef_lastnames = []
            edit_distances_AtoRef_middlenames_first = []
            edit_distances_AtoRef_middlenames_last = []

            for nameA in dfB.itertuples(index=False):
                # Onoma A - Onoma ref
                edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo A - Eponymo ref
                edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

                # MiddleName A - Firstname ref
                edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # # MiddleName A - Lastname ref
                edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])


            edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
            edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
            edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
            edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)

            # B2-ref
            edit_distances_BtoRef_names = []
            edit_distances_BtoRef_lastnames = []
            edit_distances_BtoRef_middlenames_first = []
            edit_distances_BtoRef_middlenames_last = []

            for nameB in dfB2.itertuples(index=False):
                # Onoma B - Onoma ref
                edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo B - Eponymo ref
                edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                # MiddleName B - FirstName ref
                edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # MiddleName B - Lastname ref
                edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
            edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
            edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
            edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

            # %%
            distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
            distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
            distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
            distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))

            distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))
            if np.isnan(distances).any():
                print("Result contains NaN values, stopping the run")
                nan_count = np.sum(np.isnan(distances))
                print(f"Result contains {nan_count} NaN values")
                sys.exit()

            comparisons_with_labels = pd.DataFrame((dfB['id'].values[:, None] == dfB2['id'].values).astype(int).ravel(), columns=['label'])
            distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
            dataB = cudf.DataFrame(distances_with_labels)

            # %%
            # A-B
            # Α-ref
            edit_distances_AtoRef_names = []
            edit_distances_AtoRef_lastnames = []
            edit_distances_AtoRef_middlenames_first = []
            edit_distances_AtoRef_middlenames_last = []

            for nameA in dfA.itertuples(index=False):
                # Onoma A - Onoma ref
                edit_distances_AtoRef_names.append([Levenshtein.distance(nameA.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo A - Eponymo ref
                edit_distances_AtoRef_lastnames.append([Levenshtein.distance(nameA.LastName.upper(), nameRef) for nameRef in ref['LastName']])

                # MiddleName A - Firstname ref
                edit_distances_AtoRef_middlenames_first.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # MiddleName A - Lastname ref
                edit_distances_AtoRef_middlenames_last.append([Levenshtein.distance(nameA.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])

            edit_distances_AtoRef_names = np.array(edit_distances_AtoRef_names)
            edit_distances_AtoRef_lastnames = np.array(edit_distances_AtoRef_lastnames)
            edit_distances_AtoRef_middlenames_first = np.array(edit_distances_AtoRef_middlenames_first)
            edit_distances_AtoRef_middlenames_last = np.array(edit_distances_AtoRef_middlenames_last)

            # B-ref
            edit_distances_BtoRef_names = []
            edit_distances_BtoRef_lastnames = []
            edit_distances_BtoRef_middlenames_first = []
            edit_distances_BtoRef_middlenames_last = []

            for nameB in dfB.itertuples(index=False):
                # Onoma B - Onoma ref
                edit_distances_BtoRef_names.append([Levenshtein.distance(nameB.FirstName.upper(), nameRef) for nameRef in ref['FirstName']])
                # Eponymo B - Eponymo ref
                edit_distances_BtoRef_lastnames.append([Levenshtein.distance(nameB.LastName.upper(), nameRef) for nameRef in ref['LastName']])
                # MiddleName B - FirstName ref
                edit_distances_BtoRef_middlenames_first.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['FirstName']])

                # MiddleName B - Lastname ref
                edit_distances_BtoRef_middlenames_last.append([Levenshtein.distance(nameB.MiddleName.upper(), nameRef) for nameRef in ref['LastName']])
            edit_distances_BtoRef_names = np.array(edit_distances_BtoRef_names)
            edit_distances_BtoRef_lastnames = np.array(edit_distances_BtoRef_lastnames)
            edit_distances_BtoRef_middlenames_first = np.array(edit_distances_BtoRef_middlenames_first)
            edit_distances_BtoRef_middlenames_last = np.array(edit_distances_BtoRef_middlenames_last)

            # %%
            distances_names = np.array(cdist(edit_distances_AtoRef_names, edit_distances_BtoRef_names, metric))
            distances_lastnames = np.array(cdist(edit_distances_AtoRef_lastnames, edit_distances_BtoRef_lastnames, metric))
            distances_middlenames_first = np.array(cdist(edit_distances_AtoRef_middlenames_first, edit_distances_BtoRef_middlenames_first, metric))
            distances_middlenames_last = np.array(cdist(edit_distances_AtoRef_middlenames_last, edit_distances_BtoRef_middlenames_last, metric))
            
            print('edit end')

            distances = np.column_stack(((distances_names).ravel(), (distances_lastnames).ravel(), (distances_middlenames_first).ravel(), (distances_middlenames_last).ravel()))

            comparisons_with_labels = pd.DataFrame((dfA['id'].values[:, None] == dfB['id'].values).astype(int).ravel(), columns=['label'])
            distances_with_labels = np.column_stack([comparisons_with_labels['label'], distances])
            dataAB = cudf.DataFrame(distances_with_labels)
            # %%
            test = dataAB
            trainA = dataA
            trainB = dataB
            df = cudf.concat([test, trainA, trainB])

            df.to_csv('Data/'+csv_name, index=False, header=None, chunksize=100000)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time to create FL data: {elapsed_time} seconds")
            test = dataAB = trainA = dataA = trainB = dataB = None
            del test, dataAB, trainA, dataA, trainB, dataB
            if (nrows==2000):
                # Calculate the rest of the indices
                all_indices = set(range(len(dfA_original)))
                used_indices = set(dfA.index.tolist())
                unused_indices = list(all_indices - used_indices)
                with open('unused_indices.json', 'w') as f:
                    json.dump(unused_indices, f)
                # Wait for the file to be written
                time.sleep(1)
                # Pass the unused indices to another script
                unused_indices_str = list(map(str, unused_indices))
                subprocess.run(['python', 'test_data_generatror.py', 'unused_indices.json', str(refrows), str(part)])