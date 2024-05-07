"""
Created on Fri Jan 21 2022

@author: Fujiang Ji
"""
import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy import exp, log, sin, pi
from scipy.special import expn


def Leaf_SIP_model(Cab,Ccar,Cbrw,Cw,Cm,Cant):
    
    Cab = Cab     #Chlorophyll (a+b)(cm-2.microg)
    Ccar = Ccar   #Carotenoids (cm-2.microg)
    Cbrw = Cbrw     #Brown pigments (arbitrary units)[0-1]
    Cw = Cw       #Water  (cm)
    Cm = Cm       #Dry matter (cm-2.g)
    Cant = Cant    #Anthocyanins (cm-2.microg)
    alpha = 600   # constant for the the optimal size of the leaf scattering element  
    
    data = pd.read_csv('dataSpec_PDB.csv')#%Get coefficients (Ks) for leaf bioparameters
    
    lamda = data['lambda']# wavelengths
    nr = data['nr']# refractive index
    Kab = data['Kab']# K for chlorophyll (a+b) content
    Kcar = data['Kcar']# K for carotenoids
    Kant = data['Kant'] # K for anthocyanins
    Kbrw = data['Kbrw']# K for brown pigments
    Kw = data['Kw'] # K for water
    Km = data['Km']# K for dry matter
    
    kall    = (Cab*Kab + Ccar*Kcar + Cant*Kant + Cbrw*Kbrw + Cw*Kw + Cm*Km)/(Cm*alpha)
    w0      = np.exp(-kall)
    
    # spectral invariant parameters
    fLMA = 2765.0*Cm
    gLMA = 102.8 *Cm
    
    p = 1-(1 - np.exp(-fLMA))/fLMA
    q = 1- 2 * np.exp(-gLMA)
    qabs = np.sqrt(q**2)
    
    # leaf single scattering albedo
    w = w0*(1-p)/(1-p*w0)
    
    # leaf reflectance and leaf transmittance
    refl  = w*(1/2+q/2*(1-p*w0)/(1-qabs*p*w0))
    trans = w*(1/2-q/2*(1-p*w0)/(1-qabs*p*w0))
    
    return [refl, trans]

def Leaf_SIP_model_Calibrated_pq(Cab,Ccar,Cbrw,Cw,Cm,Cant):
    
    Cab = Cab     #Chlorophyll (a+b)(cm-2.microg)
    Ccar = Ccar   #Carotenoids (cm-2.microg)
    Cbrw = Cbrw     #Brown pigments (arbitrary units)[0-1]
    Cw = Cw       #Water  (cm)
    Cm = Cm       #Dry matter (cm-2.g)
    Cant = Cant    #Anthocyanins (cm-2.microg)
    alpha = 600   # constant for the the optimal size of the leaf scattering element  
    
    data = pd.read_csv('/scratch/fji7/transfer_learning_paper/4_src_code/dataSpec_PDB.csv')#%Get coefficients (Ks) for leaf bioparameters
    
    lamda = data['lambda']# wavelengths
    nr = data['nr']# refractive index
    Kab = data['Kab']# K for chlorophyll (a+b) content
    Kcar = data['Kcar']# K for carotenoids
    Kant = data['Kant'] # K for anthocyanins
    Kbrw = data['Kbrw']# K for brown pigments
    Kw = data['Kw'] # K for water
    Km = data['Km']# K for dry matter
    
    kall    = (Cab*Kab + Ccar*Kcar + Cant*Kant + Cbrw*Kbrw + Cw*Kw + Cm*Km)/(Cm*alpha)
    w0      = np.exp(-kall)
    
    # spectral invariant parameters
    fLMA = 2519.64832573*Cm
    gLMA = -631.5441990240528*(Cm - 0.006443637536132927)
    
    p = 1-(1 - np.exp(-fLMA))/fLMA
    q = 2/(1+ np.exp(gLMA)) - 1
    qabs = np.sqrt(q**2)
    
    # leaf single scattering albedo
    w = w0*(1-p)/(1-p*w0)
    
    # leaf reflectance and leaf transmittance
    refl  = w*(1/2+q/2*(1-p*w0)/(1-qabs*p*w0))
    trans = w*(1/2-q/2*(1-p*w0)/(1-qabs*p*w0))
    
    return [refl, trans]


