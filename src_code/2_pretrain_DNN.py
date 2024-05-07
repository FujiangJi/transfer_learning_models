import pandas as pd
import numpy as np
from Models import fine_tune_synthetic_data
import joblib
import multiprocessing as mp
import psutil
import datetime
import warnings
import sys
warnings.filterwarnings('ignore')


start_t = datetime.datetime.now()
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))

RTMs = {'1_SIP':'Leaf-SIP','2_PROSPECT':'PROSPECT'}

models = ['1_SIP','2_PROSPECT']
tr_name = ['Chla+b', 'Ccar', 'EWT', 'LMA']

for model in models:
    for tr in tr_name:
        LUT_X = pd.read_csv(f'/scratch/fji7/transfer_learning_paper/1_saved_LUT/{model}_reflectance_LUT.csv')
        LUT_y = pd.read_csv(f'/scratch/fji7/transfer_learning_paper/1_saved_LUT/{model}_traits_LUT.csv')
        cols = np.arange(450,2401,10)   ###########
        cols = cols.astype(str)
        LUT_X = LUT_X[cols]
        LUT_y = LUT_y[tr]
        synthetic_data_X = LUT_X.values
        synthetic_data_y = LUT_y.values
        file_name = f'{RTMs[model]}_{tr}'
        p = pool.apply_async(func=fine_tune_synthetic_data,args= ((synthetic_data_X, synthetic_data_y, tr, file_name)))
pool.close()
pool.join()
end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')