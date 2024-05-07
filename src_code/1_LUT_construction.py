import numpy as np
import pandas as pd
from SALib.sample import fast_sampler
import datetime
from prospect_d import run_prospect
from LeafSIP import Leaf_SIP_model_Calibrated_pq

"""
1. Leaf-SIP LUT
"""

problem = {
    'num_vars': 6,
    'names': ["Chla+b",'Ccar','Cbrw','EWT','LMA','Cant'],
    'bounds': [[0,170],
               [0, 30],
               [0,1],
               [0.002,0.14],
               [0.00001,0.04],
               [0,10]]}

param_values = fast_sampler.sample(problem,2000,M=4) #############

traits = pd.DataFrame(param_values)
traits.columns = ["Chla+b",'Ccar','Cbrw','EWT','LMA','Cant']

LUT_Reflectance_zero = np.zeros(shape=(len(param_values),2101))

start_t = datetime.datetime.now()
print('Leaf_SIP start:', start_t)

for i in range(len(param_values)):
    Cab = param_values[i][0]
    Ccar = param_values[i][1]
    Cbrw = param_values[i][2]
    Cw = param_values[i][3]
    Cm = param_values[i][4]
    Cant = param_values[i][5]

    outputs = Leaf_SIP_model_Calibrated_pq(Cab, Ccar, Cbrw, Cw, Cm, Cant)
    LUT_Reflectance_zero[i] = outputs[0]

LUT_Reflectance_zero = LUT_Reflectance_zero[:,50:-100]

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('Leaf_SIP end:', end_t)
print('Leaf_SIP total:',elapsed_sec/60, 'min')

col = np.arange(450,2401)
reflectance = pd.DataFrame(LUT_Reflectance_zero)
reflectance.columns = col.astype(int).astype(str)

traits = traits.sample(n=10000, replace=True)
reflectance = reflectance.iloc[traits.index]  
traits.reset_index(drop = True, inplace = True)
reflectance.reset_index(drop = True, inplace = True)

traits.to_csv('/scratch/fji7/transfer_learning_paper/1_saved_LUT/1_SIP_traits_LUT.csv',index = False)
reflectance.to_csv('/scratch/fji7/transfer_learning_paper/1_saved_LUT/1_SIP_reflectance_LUT.csv',index = False)
print('Leaf-SIP LUT:','refl_shape',reflectance.shape, 'tr_shape',traits.shape)
print('--------------------------')


"""
2. PROSPECT-D LUT
"""

problem = {
    'num_vars': 7,
    'names': ['N',"Chla+b",'Ccar','Cbrw','EWT','LMA','Cant'],
    'bounds': [[0.8,2.5],
               [0,170],
               [0, 30],
               [0,1],
               [0.002,0.14],
               [0.00001,0.04],
               [0,10]]}

param_values = fast_sampler.sample(problem,2000,M=4) #############

traits_pro = pd.DataFrame(param_values)
traits_pro.columns = ['N',"Chla+b",'Ccar','Cbrw','EWT','LMA','Cant']

LUT_Reflectance_zero = np.zeros(shape=(len(param_values),2101))    
start_t = datetime.datetime.now()
print('PROSPECT start:', start_t)

for i in range(len(param_values)):
    N = param_values[i][0]
    Cab = param_values[i][1]
    Ccar = param_values[i][2]
    Cbrw = param_values[i][3]
    Cw = param_values[i][4]
    Cm = param_values[i][5]
    Cant = param_values[i][6]

    outputs = run_prospect(N,Cab,Ccar,Cbrw,Cw,Cm,Cant, 
                           prospect_version="D",  
                           nr=None, kab=None, kcar=None, kbrown=None, kw=None, 
                           km=None, kant=None, alpha=40.)
    LUT_Reflectance_zero[i] = outputs[1]

LUT_Reflectance_zero = LUT_Reflectance_zero[:,50:-100]

end_t = datetime.datetime.now()
elapsed_sec = (end_t - start_t).total_seconds()
print('PROSPECT end:', end_t)
print('PROSPECT total:',elapsed_sec/60, 'min')

col = np.arange(450,2401)
reflectance_pro = pd.DataFrame(LUT_Reflectance_zero)
reflectance_pro.columns = col.astype(int).astype(str)

traits_pro=traits_pro.sample(len(traits))
reflectance_pro = reflectance_pro.iloc[traits_pro.index]
traits_pro.reset_index(drop = True, inplace = True)
reflectance_pro.reset_index(drop = True, inplace = True)

traits_pro.to_csv('/scratch/fji7/transfer_learning_paper/1_saved_LUT/2_PROSPECT_traits_LUT.csv',index = False)
reflectance_pro.to_csv('/scratch/fji7/transfer_learning_paper/1_saved_LUT/2_PROSPECT_reflectance_LUT.csv',index = False)
print('PROSPECT LUT:','refl_shape',reflectance.shape, 'tr_shape',traits.shape)