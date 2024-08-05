import pandas as pd
from rapidfuzz.distance import Levenshtein
import cupy as cp

import cudf
from tqdm import tqdm

import time

metric = 'cosine'
method = 'edit_distance'

total_test_rows = [2000]

dfA_original_test = pd.read_csv('Data/BIASA_200000.csv', delimiter="|", header=None)
dfA_original_test = dfA_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')
dfB_original_test = pd.read_csv('Data/BIASB_200000.csv', delimiter="|", header=None)
dfB_original_test = dfB_original_test.drop([3,4,5,6], axis=1).rename({0:'LastName', 1:'FirstName', 2:'MiddleName',7:'id'}, axis='columns')

dfA_original_test = dfA_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)
dfB_original_test = dfB_original_test[['id', 'FirstName', 'LastName', 'MiddleName']].astype(str)

for part in range(1,2):
    for test_rows in total_test_rows:
        test_indices = pd.read_csv('test_indices_'+str(part)+'.csv', header=None)[0].tolist()[:test_rows]
        print("len test_indices = ",len(test_indices))
        filename = 'Data/local_test_samples/part'+str(part)+'_'+metric+'_n'+str(test_rows)+'.csv'
        print(filename)

        dfA_test = dfA_original_test.loc[test_indices]
        dfB_test = dfB_original_test.loc[test_indices]
        start_time = time.time()

        dfA_test['FirstName'] = dfA_test['FirstName'].str.upper()
        dfA_test['LastName'] = dfA_test['LastName'].str.upper()
        dfA_test['MiddleName'] = dfA_test['MiddleName'].str.upper()

        dfB_test['FirstName'] = dfB_test['FirstName'].str.upper()
        dfB_test['LastName'] = dfB_test['LastName'].str.upper()
        dfB_test['MiddleName'] = dfB_test['MiddleName'].str.upper()

        chunk_size = 1000
        num_chunks = len(dfA_test) // chunk_size + 1
        with open(filename, 'w') as f:
            for i in tqdm(range (num_chunks)):
                start_i = i * chunk_size
                end_i = (i + 1) * chunk_size
                dfA_chunk = dfA_test.iloc[start_i:end_i]
                if end_i > test_rows:
                    continue
                for j in tqdm(range (num_chunks)):
                    start_j = j * chunk_size
                    end_j = (j + 1) * chunk_size
                    if end_j > test_rows:
                        continue
                    dfB_chunk = dfB_test.iloc[start_j:end_j]
                    distances_names = []
                    distances_lastnames = []
                    distances_middlenames = []
                    labels = []
                    for nameA in dfA_chunk.itertuples(index=False):
                        for nameB in dfB_chunk.itertuples(index=False):
                                distances_names.append(Levenshtein.distance(nameA.FirstName, nameB.FirstName))
                                distances_lastnames.append(Levenshtein.distance(nameA.LastName, nameB.LastName))
                                distances_middlenames.append(Levenshtein.distance(nameA.MiddleName, nameB.MiddleName))
                                label = int(nameA.id == nameB.id)
                                labels.append(label)
                    distances_names = cp.asarray(distances_names)
                    distances_lastnames = cp.asarray(distances_lastnames)
                    distances_middlenames = cp.asarray(distances_middlenames)
                    labels = cp.asarray(labels)
                    distances = cp.column_stack((cp.asarray(distances_names).ravel(), cp.asarray(distances_lastnames).ravel(), cp.asarray(distances_middlenames).ravel()))
                    data = cudf.DataFrame(cp.column_stack([labels, cp.column_stack((distances_names.ravel(), distances_lastnames.ravel(), distances_middlenames.ravel())).astype(float)]))
                    data.to_csv(f, index=False, header=None, chunksize=100000)
        end_time = time.time()
        total_time = end_time - start_time  
        print(f"Total time for part {part}, test_rows {test_rows}: {total_time:.2f} seconds")