import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
import joblib
import sys
import multiprocessing as mp
import psutil
import datetime
import warnings
from sklearn.model_selection import train_test_split
from Models import transfer_learning_model
warnings.filterwarnings('ignore')

def fine_tune_DNN(tr, tr_obs, refl_obs, train_size, RTM, n_iterations, data_type):
    X = refl_obs
    y = tr_obs
    n_iterations = n_iterations
     
    var_start = True
    for iteration in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=iteration)
        print(tr, f'{RTM}_{tr}_iteration {iteration+1}_{data_type}, {str(int(train_size*100))}% train size: {len(X_train)}, test size: {len(X_test)}')
        
        file_name = f'{RTM}_{tr}'
        pred_ann = transfer_learning_model(X_train, y_train, X_test, y_test, tr, file_name, data_type, iteration)
        
        y_test.reset_index(drop = True, inplace = True)
        y_test['pred_fine-tune'] = pred_ann
        y_test['iteration'] = iteration+1
        
        if var_start:
            iterative_pred = y_test
            var_start = False
        else:
            iterative_pred = pd.concat([iterative_pred,y_test],axis = 0)
        
    iterative_pred.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/4_{tr}_{RTM}_{str(int(train_size*100))}% obs fine-tune_{data_type}.csv',index = False)
    return  



tr = sys.argv[1]         # tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
data_type = sys.argv[2]   # when tr = "Chla+b", "Ccar", "LMA", data_types = ['sites','PFT','temporal'].  when tr = "EWT", data_types = ['sites','PFT']

data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_{data_type}.csv")
tr_obs, refl_obs = data.loc[:,'Dataset ID':], data.loc[:,'450':'2400']

start_t = datetime.datetime.now()
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))

RTMs = ['Leaf-SIP','PROSPECT']
ts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for model in RTMs:
    for train_size in ts:
        p = pool.apply_async(func=fine_tune_DNN,args= (tr, tr_obs, refl_obs, train_size, model, 10, data_type))

pool.close()
pool.join()
end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/3600, 'hours')