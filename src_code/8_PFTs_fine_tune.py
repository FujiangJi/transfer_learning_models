import numpy as np
import pandas as pd
import sys
import joblib
import datetime
import warnings
import multiprocessing as mp
import psutil
from sklearn.model_selection import train_test_split,KFold
from Models import transfer_learning_model_transferability
warnings.filterwarnings('ignore')

def PFTs_fine_tune(tr, tr_obs, refl_obs, RTM, n_splits,n_iterations):
    X = refl_obs
    y = tr_obs
    
    k = 0
    pfts = y['PFT'].unique()
    kf = KFold(n_splits=n_splits)
    var=True
    for train_index, test_index in kf.split(pfts):                                                                     
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
        start = True
        for iteration in range(n_iterations):
            print(f'{RTM}_{tr}_Fold{k+1} --- iteration {iteration+1}---train_pfts:{train_pft},train_samples:{len(X_train)} --- test_pfts:{test_pft}, test_samples:{len(X_test)}')
            XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.2, random_state=iteration)
            
            trans_type = f'{RTM} PFT transferability Fold{k+1}_iteration {iteration+1}'                                                                         
            file_name = f'{RTM}_{tr}'
            pred_ann = transfer_learning_model_transferability(XX_train, yy_train, XX_test, yy_test, tr, file_name, trans_type, X_test)
            max_pred_within = pred_ann[0]
            max_pred_out = pred_ann[1]
            
            yy_test.reset_index(drop = True, inplace = True)
            yy_test['pred_fine-tune'] = max_pred_within
            yy_test['fold'] = k+1
            yy_test['iteration'] = iteration+1
            
            max_pred_out = pd.DataFrame(max_pred_out,columns = [f'itration_{iteration+1}'])
                
            if start:
                pred_in,pred_out = yy_test, max_pred_out
                start = False
            else:
                pred_in = pd.concat([pred_in,yy_test],axis = 0)
                pred_out = pd.concat([pred_out,max_pred_out],axis = 1)
        
        y_test.reset_index(drop = True, inplace = True)
        pred_out_final = pd.concat([y_test,pred_out],axis = 1)
        pred_out_final['fold'] = f'fold{k+1}'
        
        if var:
            df_out, df_in = pred_out_final, pred_in
            var = False
        else:
            df_out,df_in = pd.concat([df_out,pred_out_final],axis = 0),pd.concat([df_in,pred_in],axis = 0)
   
        k = k+1

    lr_idx = [lr for lr in df_out.columns if "itration_" in lr]
    df_out['pred_fine-tune'] = df_out[lr_idx].mean(axis = 1)
    
    df_out.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/8_{tr}_{RTM}_PFTs fine-tune_out.csv',index = False)
    df_in.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/8_{tr}_{RTM}_PFTs fine-tune_within.csv',index = False)
    return


start_t = datetime.datetime.now()
print('start:', start_t)
print('cores',psutil.cpu_count(logical=False))
pool = mp.Pool(psutil.cpu_count(logical=False))

tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
RTMs = ['Leaf-SIP','PROSPECT']

for tr in tr_name:
    data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_PFT.csv")
    tr_obs, refl_obs = data.loc[:,'Dataset ID':], data.loc[:,'450':'2400']
    if tr =='LMA':
        n_splits = 7
    else:
        n_splits = 3
    for model in RTMs:
         p = pool.apply_async(func=PFTs_fine_tune,args= (tr, tr_obs, refl_obs, model, n_splits, 10))
         
pool.close()
pool.join()
end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('end:', end_t)
print('total:',elapsed_sec/3600, 'hours')