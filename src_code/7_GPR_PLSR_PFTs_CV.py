import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,train_test_split
from Models import GPR_model, PLSR_model, DNN_model_transferabiliy
import datetime
import warnings
import multiprocessing as mp
import psutil
warnings.filterwarnings('ignore') 

def GPR_PLSR_PFTs_CV(tr, tr_obs, refl_obs,n_splits,n_iterations):
    X = refl_obs
    y = tr_obs
    
    vip_score = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_splits, X.shape[1])),columns = X.columns)
    
    k = 0
    pfts = y['PFT'].unique()
    kf = KFold(n_splits=n_splits)
    var=True
    for train_index, test_index in kf.split(pfts):                                           ### Line 1 chenge
    # for test_index, train_index in kf.split(pfts):
        train_pft = pfts[train_index]
        test_pft = pfts[test_index]
        
        y_train = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
        y_test = pd.DataFrame(np.zeros(shape = (0,len(y.columns))),columns = y.columns)
    
        for i in train_pft:
            temp = y[y['PFT'] == i]
            y_train = pd.concat([y_train,temp])
        for i in test_pft:
            temp = y[y['PFT'] == i]
            y_test = pd.concat([y_test,temp])
            
        X_train = X.iloc[y_train.index]
        X_test =  X.iloc[y_test.index]
        
        n_iterations = n_iterations
        var_start = True
        for iteration in range(n_iterations):
            print(tr, f'Fold{k+1} --- iteration {iteration+1}---train_pfts:{train_pft},train_samples:{len(X_train)} --- test_pfts:{test_pft}, test_samples:{len(X_test)}')
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.2, random_state=iteration)
            
            ### GPR estimation
            gpr = GPR_model(XX_train, yy_train, XX_test, tr, X_test)
            pred_gpr_within, pred_gpr_out = gpr[0], gpr[1]
            pred_gpr_within = pd.DataFrame(pred_gpr_within,columns = ['predicted_gpr'])
            pred_gpr_out = pd.DataFrame(pred_gpr_out,columns = [f'GPR_iteration_{iteration+1}'])
            
            ### PLSR estimation
            plsr = PLSR_model(XX_train, yy_train, XX_test, yy_test, tr, X_test)
            pred_plsr_within, pred_plsr_out = plsr[0], plsr[1]
            pred_plsr_within = pd.DataFrame(pred_plsr_within,columns = ['predicted_plsr'])
            pred_plsr_out = pd.DataFrame(pred_plsr_out,columns = [f'PLSR_iteration_{iteration+1}'])
            vvv, coef = plsr[2], plsr[3]
            
            
            
            yy_test.reset_index(drop = True, inplace = True)
            pred_within = pd.concat([yy_test,pred_gpr_within, pred_plsr_within],axis = 1)
            pred_within['fold'] = k+1
            pred_within['iteration'] = iteration+1
            
            pred_out = pd.concat([pred_gpr_out, pred_plsr_out], axis = 1)
            if var_start:
                iterative_pred_within = pred_within
                iterative_pred_out = pred_out
                coefficients, vip_ = coef, vvv
                var_start = False
            else:
                iterative_pred_within = pd.concat([iterative_pred_within,pred_within],axis = 0)
                iterative_pred_out = pd.concat([iterative_pred_out,pred_out],axis = 1)
                coefficients, vip_ = coefficients+coef, vip_+vvv
        
        idx_gpr_out = [model for model in iterative_pred_out.columns if "GPR" in model]
        idx_plsr_out = [model for model in iterative_pred_out.columns if "PLSR" in model]
        final_pred_gpr_out = pd.DataFrame(iterative_pred_out[idx_gpr_out].mean(axis = 1),columns = ['predicted_gpr'])
        final_pred_plsr_out = pd.DataFrame(iterative_pred_out[idx_plsr_out].mean(axis = 1),columns = ['predicted_plsr'])
       
        y_test.reset_index(drop = True, inplace = True)
        final_pred_out = pd.concat([y_test,iterative_pred_out,final_pred_gpr_out,final_pred_plsr_out],axis = 1)
        final_pred_out['fold'] = f'fold{k+1}'
        
        if var:
            df_out = final_pred_out
            df_within = iterative_pred_within
            var = False
        else:
            df_out = pd.concat([df_out,final_pred_out],axis = 0)
            df_within = pd.concat([df_within,iterative_pred_within],axis = 0)
        
        vip_score.iloc[k] = vip_
        plsr_coef.iloc[k] = coefficients/n_iterations
        
        k = k+1
                                                            
    df_within.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/7_{tr}_PFTs CV train GPR_PLSR_df_within.csv',index = False)   
    df_out.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/7_{tr}_PFTs CV train GPR_PLSR_df_out.csv',index = False)
    vip_score.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/7_{tr}_PFTs CV train GPR_PLSR_VIP.csv',index = False)
    plsr_coef.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/7_{tr}_PFTs CV train GPR_PLSR_coefficients.csv',index = False)
    return

start_t = datetime.datetime.now()
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))

# tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
tr_name = ["Chla+b", 'Ccar']

for tr in tr_name:
    data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_PFT.csv")
    tr_obs, refl_obs = data.loc[:,'Dataset ID':], data.loc[:,'450':'2400']
    if tr =='LMA':
        n_splits = 7
    else:
        n_splits = 3
    
    p = pool.apply_async(func=GPR_PLSR_PFTs_CV,args= (tr, tr_obs, refl_obs, n_splits, 10))


pool.close()
pool.join()
end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/60, 'min')