import io
import requests
import os
import sys
import math
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import random
import copy
from scipy.stats import norm as normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from programs.sklearn_models import *

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

def aval_reg(b, X, y): 
    np.random.seed(2*b)
    
    #X=data[name][0]
    #y=data[name][1]

    ### Biasing
    n,d=np.shape(X)
    
    #pca = PCA(n_components=1)
    #X_v=pca.fit_transform(X) 

    v=np.random.uniform(-1,1,d)
    v=v/np.sqrt(v@v)

    X_v=X@v
    med_v=np.median(X_v)
    min_v=np.min(X_v)
    max_v=np.max(X_v)
    std_v=np.std(X_v)

    u_train=np.random.uniform(0,1,n)
    u_test=np.random.uniform(0,1,n)

    #choosing a
    for i in np.linspace(1,100,200):
        
        eps=10**-20
        
        #train
        p_train=normal.cdf(X_v, med_v, std_v/i).reshape(-1)
        s_train=u_train<p_train #Selection variable

        #test
        s_test=u_train>=p_train #u_test<p_test #Selection variable

        #
        w=(1-p_train + eps)/(p_train + eps)
        w_s=w[s_train]
        #w_s=w_s/np.sum(w_s)
        
        ns=np.sum(s_train)
        ess_n=np.sum(w_s)**2/np.sum(w_s**2) #np.sum(s_train)**2
        
        ess_perc=ess_n/np.sum(s_train)

        if ess_perc<=.01: break 

    ### Training Models
    # Datasets
    X_test_train, X_test_test, y_test_train, y_test_test=train_test_split(X[s_test], y[s_test], test_size=0.5, random_state=42)

    #model train
    model_train=DT_reg().fit(X[s_train], y[s_train], sample_weight=None)
    #model test
    model_test=DT_reg().fit(X_test_train, y_test_train, sample_weight=None)
    
    ### Evaluating the models
    y_pred1=model_train.predict(X_test_test)
    y_pred2=model_test.predict(X_test_test)
    
    ### Outputs
    error_ratio=mean_squared_error(y_test_test, y_pred2)/mean_squared_error(y_test_test, y_pred1)
    
    ess_n=np.round(np.sum(w_s)**2/np.sum(w_s**2),2) #np.round(np.sum(w_s)**2/np.sum(w_s**2),2)
    ess_perc=ess_n/np.sum(s_train)

    #print(np.mean(w_s))
    return [error_ratio, ess_perc, ess_n, np.sum(s_train), s_train, w]


def aval_class(b, X, y): 
    np.random.seed(2*b)

    ### Biasing
    n,d=np.shape(X)
    
    v=np.random.uniform(-1,1,d)
    v=v/np.sqrt(v@v)

    X_v=X@v
    med_v=np.median(X_v)
    min_v=np.min(X_v)
    max_v=np.max(X_v)
    std_v=np.std(X_v)

    u_train=np.random.uniform(0,1,n)
    u_test=np.random.uniform(0,1,n)

    #choosing a
    for i in np.linspace(1,100,200):
        
        eps=10**-20
        
        #train
        p_train=normal.cdf(X_v, med_v, std_v/i).reshape(-1)
        s_train=u_train<p_train #Selection variable

        #test
        s_test=u_train>=p_train #u_test<p_test #Selection variable

        #
        w=(1-p_train + eps)/(p_train + eps)
        w_s=w[s_train]
        #w_s=w_s/np.sum(w_s)
        
        ns=np.sum(s_train)
        ess_n=np.sum(w_s)**2/np.sum(w_s**2) #np.sum(s_train)**2
        
        ess_perc=ess_n/np.sum(s_train)

        if ess_perc<=.01: break 

    ### Training Models
    # Datasets
    X_test_train, X_test_test, y_test_train, y_test_test=train_test_split(X[s_test], y[s_test], test_size=0.5, random_state=42)

    #model train
    model_train=DT_class().fit(X[s_train], y[s_train])
    #model test
    model_test=DT_class().fit(X_test_train, y_test_train)
    
    ### Evaluating the models
    y_pred1=model_train.predict(X_test_test)
    y_pred2=model_test.predict(X_test_test)
    
    ### Outputs
    error_ratio=class_error(y_test_test,y_pred2)/class_error(y_test_test,y_pred1)
    
    ess_n=np.round(np.sum(w_s)**2/np.sum(w_s**2),2) #np.round(np.sum(w_s)**2/np.sum(w_s**2),2)
    ess_perc=ess_n/np.sum(s_train)

    #print(np.mean(w_s))
    return [error_ratio, ess_perc, ess_n, np.sum(s_train), s_train, w]