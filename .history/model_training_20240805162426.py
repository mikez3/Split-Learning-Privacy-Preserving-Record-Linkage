import pandas as pd
from cuml import SVC
import joblib
from tqdm import tqdm
import time

records = 2000
ref = 200
chunk_size = 2

training_dataA = pd.read_csv("/media/mike/corsair/correct_data/new_dataset/trainingA_round3_chunk2_proth_dokimh_2000_records_ref200.csv", header=None)

n=records**2
x_trainA = training_dataA.iloc[:, 1:].values
y_trainA = training_dataA.iloc[:, 0].values

data_path = '/media/mike/corsair/correct_data/new_dataset/only_tests/no_ref_tests/test_local_5000_1.csv'
model_name = '/home/mike/ΠΤΥΧΙΑΚΗ/NVFlare_project/my_nvflare_example/FL-record-linkage2/trained_models/shuffle/last_dance/local_linear_c100_svm_n2000_1.joblib'
model2 = joblib.load(model_name)   

chunksize = 40000000
kernel = 'rbf'
c=100
train_records = 5000
clients = ['A','B','local']
for client in clients:
    for part in range (1,4):
        data = pd.read_csv('/media/mike/corsair/correct_data/new_dataset/only_tests/no_ref_tests/train_'+client+'_'+str(train_records)+'_'+str(part)+'.csv', header=None)
        X_data = data.iloc[:, 1:].values  
        y_data = data.iloc[:, 0].values   
        start_time = time.time()  
        svmA = SVC(kernel=kernel, C=c)
        svmA.fit(X_data, y_data)
        end_time = time.time()  
        training_time = end_time - start_time  
        print(f"Training time for client {client}, part {part}: {training_time:.2f} seconds")
        joblib.dump(svmA, '/home/mike/ΠΤΥΧΙΑΚΗ/NVFlare_project/my_nvflare_example/FL-record-linkage2/trained_models/shuffle/last_dance/skata_'+client+'_'+str(kernel)+'_c'+str(c)+'_svm_n'+str(train_records)+'_'+str(part)+'.joblib')