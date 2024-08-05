import pandas as pd
from cuml import SVC
import joblib
import time

records = 2000
ref = 200
chunk_size = 2
chunksize = 40000000
kernel = 'rbf'
c=100
train_records = 5000
clients = ['A','B']

for client in clients:
    for part in range (1,4):
        data = pd.read_csv('Data/train_'+client+'_'+str(train_records)+'_'+str(part)+'.csv', header=None)
        data = pd.read_csv('Data/training'+client+'_round3_n'+str(train_records)+'_ref'+str(ref)+'_'+str(part)+'.csv', header=None)

        csv_name = 'part'+str(part)+'_reference_set_'+metric+"_n"+str(nrows)+"_ref"+str(refrows)+".csv"
        X_data = data.iloc[:, 1:].values  
        y_data = data.iloc[:, 0].values   
        start_time = time.time()  
        svmA = SVC(kernel=kernel, C=c)
        svmA.fit(X_data, y_data)
        end_time = time.time()  
        training_time = end_time - start_time  
        print(f"Training time for client {client}, part {part}: {training_time:.2f} seconds")
        joblib.dump(svmA, '/home/mike/ΠΤΥΧΙΑΚΗ/NVFlare_project/my_nvflare_example/FL-record-linkage2/trained_models/shuffle/last_dance/skata_'+client+'_'+str(kernel)+'_c'+str(c)+'_svm_n'+str(train_records)+'_'+str(part)+'.joblib')