from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from LeafSIP import Leaf_SIP_model_Calibrated_pq
import warnings
warnings.filterwarnings('ignore')

def calibrated_SIP_cost(para, observed):
    Cab,Ccar,Cbrw,Cw,Cm,Cant = para
    refl_sli = Leaf_SIP_model_Calibrated_pq(Cab, Ccar, Cbrw, Cw, Cm, Cant)[0][50:-100]
    refl_sli = refl_sli[::10]
    MSE = mean_squared_error(observed, refl_sli)
    rmse = np.power(MSE, 0.5)
    return rmse

tr = sys.argv[1]    ##### tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
type = sys.argv[2]  ##### when tr = "Chla+b", "Ccar", "LMA", data_types = ['sites','PFT','temporal'].  when tr = "EWT", data_types = ['sites','PFT']

data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_{type}.csv")
refl = data.loc[:,'450':'2400']
traits = data.loc[:,'Dataset ID':]

SIP = pd.DataFrame(np.zeros(shape=(len(refl),6)))
SIP.columns = ['Chla+b_res','Ccar_res','Cbrw_res','EWT_res','LMA_res','Cant_res']

for i in range(len(refl)):
    observed_refl = refl.iloc[i]
    SIP_bounds = [[0,170],[0, 30],[0,1],[0.002,0.14],[0.00001,0.04],[0,10]]
    
    SIP_optimize = differential_evolution(calibrated_SIP_cost, SIP_bounds, args=(observed_refl,),
                                          updating='deferred', maxiter= 40, popsize= 15, workers = -1)
    
    if SIP_optimize.message!='Optimization terminated successfully.':
        SIP_optimize = differential_evolution(calibrated_SIP_cost, SIP_bounds, args=(observed_refl,),
                                              updating='deferred', maxiter= 80, popsize= 15, workers = -1)
        
    SIP.iloc[i] = SIP_optimize.x.reshape(1,6)
    print(i,'-SIP:',SIP_optimize.message)

SIP_final = pd.concat([traits,SIP], axis=1, join='inner')
SIP_final.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/0_pure_SIP_{tr}_{type}.csv',index = False)