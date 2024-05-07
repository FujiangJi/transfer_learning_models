import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Models import GPR_model, PLSR_model, DNN_model
import datetime
import warnings
import multiprocessing as mp
import psutil
warnings.filterwarnings('ignore')

def partial_obs_GPR_PLSR(tr, tr_obs, refl_obs,train_size, n_iterations, data_type):
    X = refl_obs
    y = tr_obs

    n_iterations = n_iterations
    vip_score = pd.DataFrame(np.zeros(shape = (n_iterations, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_iterations, X.shape[1])),columns = X.columns)

    var_start = True
    for iteration in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=iteration)
        print(f'{tr}_{data_type}_iteration {iteration+1}, {str(int(train_size*100))}% train size: {len(X_train)}, test size: {len(X_test)}')
        ### GPR estimation
        pred_gpr = GPR_model(X_train, y_train, X_test, tr)[0]
        pred_gpr = pd.DataFrame(pred_gpr,columns = ['predicted_gpr'])
        
        ### PLSR estimation
        plsr_results = PLSR_model(X_train, y_train, X_test, y_test, tr)
        pred_plsr= plsr_results[0]
        pred_plsr = pd.DataFrame(pred_plsr,columns = ['predicted_plsr'])
        vip_score.iloc[iteration] = plsr_results[2]
        plsr_coef.iloc[iteration] = plsr_results[3]
        
        y_test.reset_index(drop = True, inplace = True)
        pred = pd.concat([y_test,pred_gpr, pred_plsr],axis = 1)
        pred['iteration'] = iteration+1
        
        if var_start:
            iterative_pred = pred
            var_start = False
        else:
            iterative_pred = pd.concat([iterative_pred,pred],axis = 0)

    iterative_pred.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/3_{tr}_{str(int(train_size*100))}% train GPR_PLSR_df_{data_type}.csv',index = False)
    vip_score.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/3_{tr}_{str(int(train_size*100))}% train GPR_PLSR_VIP_{data_type}.csv',index = False)
    plsr_coef.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/3_{tr}_{str(int(train_size*100))}% train GPR_PLSR_coefficients_{data_type}.csv',index = False)
    return


tr = sys.argv[1]         # tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
data_type = sys.argv[2]   # when tr = "Chla+b", "Ccar", "LMA", data_types = ['sites','PFT','temporal'].  when tr = "EWT", data_types = ['sites','PFT']

data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_{data_type}.csv")
tr_obs, refl_obs = data.loc[:,'Dataset ID':], data.loc[:,'450':'2400']

start_t = datetime.datetime.now() 
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))

ts = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

for train_size in ts:
    p = pool.apply_async(func=partial_obs_GPR_PLSR,args= (tr, tr_obs, refl_obs,train_size, 10, data_type))

pool.close()
pool.join()
end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/3600, 'hours')