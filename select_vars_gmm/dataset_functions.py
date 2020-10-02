import io
import requests
import os
import sys
import math
import numpy as np  
import pandas as pd
import random
import copy
from sklearn.preprocessing import StandardScaler

def download_dataset(name):
    #Separator#
    if name in ['bank32nh', 'bank8FM','puma8NH','fried_delve', 'delta_ailerons']: sep=" "
    elif name in ['winequality']: sep=";"
    else: sep=","
    ###########
    
    #.data
    url="https://raw.githubusercontent.com/felipemaiapolo/master_thesis/master/open_datasets/"+name+".data"
    s=requests.get(url).content
    d1=pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, sep=sep)
    #.test
    url="https://raw.githubusercontent.com/felipemaiapolo/master_thesis/master/open_datasets/"+name+".test"
    s=requests.get(url).content
    d2=pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, sep=sep)

    #Output
    if d2.iloc[0,0]=='404: Not Found': 
        print("- ***",name,"*** dataset shape=",np.shape(d1)) #Printing shape of dataset
        
        #### exception
        if name in ['bank32nh','bank8FM','puma8NH']:d1=d1.iloc[:,:-1]
        else: pass
        ####
        
        return d1.dropna()
    else: 
        d1=d1.append(d2)
        print("- ***",name,"*** dataset shape=",np.shape(d1)) #Printing shape of dataset
        
        #### exception
        if name in ['bank32nh','bank8FM','puma8NH']:d1=d1.iloc[:,:-1]
        else: pass
        ####
        
        return d1.dropna()
    
def get_X_y(pd_df,scale=True):
    if scale: 
        scaler=StandardScaler()
        scaled=scaler.fit_transform(np.array(pd_df))
        X=scaled[:,:-1]
        y=scaled[:,-1]
    else:
        X=np.array(pd_df)[:,:-1]
        y=np.array(pd_df)[:,-1]
    return [X,y]
