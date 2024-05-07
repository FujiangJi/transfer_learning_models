from scipy.optimize import differential_evolution
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from prospect_d import run_prospect
import warnings
warnings.filterwarnings('ignore')

def prospect_cost(para,observed):
    N,Cab,Ccar,Cbrw,Cw,Cm,Cant = para
    refl_sli = run_prospect(N, Cab, Ccar, Cbrw, Cw, Cm, Cant, 
                            prospect_version="D",  
                            nr=None, kab=None, kcar=None, kbrown=None, kw=None, 
                            km=None, kant=None, alpha=40.)[1][50:-100]
    refl_sli = refl_sli[::10]
    MSE = mean_squared_error(observed, refl_sli)
    rmse = np.power(MSE, 0.5)
    return rmse

tr = sys.argv[1]    ##### tr_name = ["Chla+b", 'Ccar', 'EWT','LMA']
type = sys.argv[2]  ##### when tr = "Chla+b", "Ccar", "LMA", data_types = ['sites','PFT','temporal'].  when tr = "EWT", data_types = ['sites','PFT']

data = pd.read_csv(f"/scratch/fji7/transfer_learning_paper/0_datasets/{tr}_dataset_{type}.csv")
refl = data.loc[:,'450':'2400']
traits = data.loc[:,'Dataset ID':]

prospect = pd.DataFrame(np.zeros(shape=(len(refl),7)))
prospect.columns = ['N','Chla+b_res','Ccar_res','Cbrw_res','EWT_res','LMA_res','Cant_res']

for i in range(len(refl)):
    observed_refl = refl.iloc[i]
    prospect_bounds = [[0.9,2.5],[0,170],[0, 30],[0,1],[0.002,0.14],[0.00001,0.04],[0,10]]
    
    prospect_optimize = differential_evolution(prospect_cost,prospect_bounds, args=(observed_refl,),
                                               updating='deferred', maxiter= 60, popsize= 15, workers = -1)

    if prospect_optimize.message!='Optimization terminated successfully.':
        prospect_optimize = differential_evolution(prospect_cost,prospect_bounds, args=(observed_refl,),
                                                   updating='deferred', maxiter= 80, popsize= 15, workers = -1)
    
    prospect.iloc[i] = prospect_optimize.x.reshape(1,7)
    print(i,'-PROSPECT:', prospect_optimize.message)

prospect_final = pd.concat([traits, prospect], axis=1, join='inner')
prospect_final.to_csv(f'/scratch/fji7/transfer_learning_paper/3_results/{tr}/0_pure_PROSPECT_{tr}_{type}.csv',index = False)