import math
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import random
import copy
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn import mixture



def MI_gmm_reg(X,y,gmm,feat):
    
    '''
    Esta função calcula a informação mútua entre y e X[:,feat] em casos que y é QUANTITATIVA! 
    Ou seja, quando queremos realizar uma tarefa de regressão em uma etapa posterior.
    
    Inputs: X (numpy array de features, y (numpy array de labels), Modelo GMM, índices das features (feat)
    
    Output: dicionário contendo a estimativa para informação mútua entra y e X[:,feat], 
            além do desvio-padrão calculado a partir das amostras.
    '''
    
    eps=10**-50 
    n,d=X.shape
    components=gmm.n_components
    Z=np.hstack((y.reshape((-1,1)),X))
    feat2=[0]+[f+1 for f in feat] #feat2 inclui y também, além de X[:,feat]

    ### Calculando log-likelihood das amostras (x_i,y_i) com base no GMM
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,feat2], gmm.means_[c][feat2], gmm.covariances_[c][feat2][:,feat2])

    log_like_xy=np.log(like + eps)

    
    ### Calculando log-likelihood das amostras (x_i) com base no GMM
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,feat2[1:]], gmm.means_[c][feat2[1:]], gmm.covariances_[c][feat2[1:]][:,feat2[1:]])

    log_like_x=np.log(like + eps)

    
    ### Calculando log-likelihood das amostras (y_i) com base no GMM
    like=np.zeros(n)
    for c in range(components):
        like+=gmm.weights_[c]*multivariate_normal.pdf(Z[:,0], gmm.means_[c][0], gmm.covariances_[c][0][0])

    log_like_y=np.log(like + eps)
     
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}



def MI_gmm_class(X,y,gmm,feat):
    
    '''
    Esta função calcula a informação mútua entre y e X[:,feat] em casos que y é CATEGÓRICA! 
    Ou seja, quando queremos realizar uma tarefa de classificação em uma etapa posterior.
    
    Inputs: X (numpy array de features, y (numpy array de labels), 
            dicionário de modelos GMM (um modelo para cada valor de y), índices das features (feat)
    
    Output: dicionário contendo a estimativa para informação mútua entra y e X[:,feat], 
            além do desvio-padrão calculado a partir das amostras.
    '''
    
    eps=10**-50
    n,d=X.shape
    classes=list(set(y))
    p={}

    ### Calculando log-likelihood das amostras (y_i) com base nas frequências relativas dos labels
    like=np.zeros(n)
    for c in classes:
        p[c]=np.mean(y==c)
        like[y==c]=p[c]

    log_like_y=np.log(like + eps)
    
    
    ### Calculando log-likelihood das amostras (x_i,y_i) com base nos modelos GMM
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux[y==c]+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[y==c][:,feat], gmm[c].means_[comp][feat], gmm[c].covariances_[comp][feat][:,feat])

        #(X,Y)
        like[y==c]=p[c]*like_aux[y==c] 
    log_like_xy=np.log(like + eps)

    
    ### Calculando log-likelihood das amostras (y_i) com base nos GMMs
    like=np.zeros(n)
    for c in classes:
        #X|Y
        like_aux=np.zeros(n)
        for comp in range(gmm[c].n_components):
            like_aux+=gmm[c].weights_[comp]*multivariate_normal.pdf(X[:,feat], gmm[c].means_[comp][feat], gmm[c].covariances_[comp][feat][:,feat])

        #(X,Y)
        like+=p[c]*like_aux

    log_like_x=np.log(like + eps)
    
    
    #Output
    m=np.mean(log_like_xy-log_like_x-log_like_y)
    s=np.std(log_like_xy-log_like_x-log_like_y)
    
    return {'mi':m, 'std':s}



def MI(cand, posic, r, X, y, gmm, include_cand = True):
    
    '''
    Esta função é uma intermediária entre a classe principal e as duas funções que fazem o cálculo 
    das informaçẽos mútuas. Ela basicamente decide qual das duas funções utilizar.
    
    Inputs: X (numpy array de features, y (numpy array de labels), 
            gmm - dicionário de modelos GMM (um modelo para cada valor de y) ou modelo individual GMM, 
            posic - lista com posições das variáveis selecionadas até o momento,
            cand - posição da variável candidata a ser escolhida
            r - contador de round

    Output: cand - posição da variável candidata a ser escolhida
            dic - dicionário contendo a estimativa para informação mútua entra y e X[:,feat], 
            além do desvio-padrão calculado a partir das amostras.    
    '''
        
    n,d=X.shape
    aux = copy.deepcopy(posic)
    if include_cand:
        aux[r] = cand
    else:
        aux.remove(cand)
    
    if type(gmm)==dict:
        dic=MI_gmm_class(X,y,gmm,aux)
    else:
        dic=MI_gmm_reg(X,y,gmm,aux) 
    
    return cand, dic
 
    
    
class SelectVars:
    d_max = None
    selection_mode = None
    gmm = None
    n = None
    '''
    Esta é a classe principal do pacote.
    '''
    
    def __init__(self, gmm, selection_mode = 'include', d_max=10, stop=.01):
        """
        :param gmm: dicionário de modelos GMM (um modelo para cada valor de y) ou modelo individual GMM,
        :param selection_mode: "remove" para comecar com o conjunto completo e remover variáveis ou "include" para comecar
            com conjunto vazio e ir adicionando variáveis
        :param d_max: número máximo de variáveis permitidas
        :param stop: regra de parada
        """
        if not selection_mode in ['remove','include']:
            raise ValueError("Selection model should be either 'remove' or 'include'.")
        self.selection_mode = selection_mode
        self.d_max=d_max
        self.stop=stop
        self.gmm=gmm
    
    def fit(self, X, y, verbose=True):
        
        '''
        Função no estilo "Scikit-Learn" para fazermos a seleção

        Inputs: X (numpy array de features, y (numpy array de labels)   
        '''
        
        n,d=X.shape
        d_max=self.d_max
        gmm=self.gmm
        self.n=n
        include_var = self.selection_mode == 'include'  #True if include or False if remove
        
        if d_max<1 or d_max>=d: raise ValueError("'d_max' must be an integer between 1 and {}.".format(d-1))

        MIs = []  # lista com histórico de info mútuas ao incluirmos as melhores variáveis
        stds = []  # lista com histórico stds e que utilizaremos para p cálculo do erro padrão das MIs
        lista = list(range(d))  # lista com índices de todas as variáveis

        #A lista comeca vazia se for incluindo
        if include_var:
            posic = []  # lista com posições das variáveis selecionadas até o momento
            repetitions = range(d_max)
        else:
            posic = copy.deepcopy(lista)
            repetitions = range(d-1) #range((d-1) - d_max)

            
        ## Começando a seleção ##
        if verbose: print("Let's begin the selection...")
            
        for r in repetitions: # "r" de rounds
            
            if include_var:
                posic.append(None)

            if include_var:
                MI_best=-math.inf
            else:
                MI_best = -math.inf

            #Revisar aqui
            #Calcula MI entre y e X[:,(posic, cand)] >> cand: variável candidata a ser selecionada
            outputs = [MI(cand, posic, r, X, y, gmm, include_var) for cand in lista]
    
            #Escolhendo variável que traz maior retorno
            for out in outputs:
                
                cand, dic = out
                MI_current = dic['mi']
                
                if (MI_current > MI_best and include_var) or (MI_current > MI_best and not include_var):
                    MI_best=MI_current
                    std_best=dic['std']
                    best_index= cand

                    
            #Checking stopping rule
            if self.stop==None: 
                pass
            else:
                if r>=1:
                    if (MI_best/MIs[-1]-1 < self.stop and include_var) or (MI_best/MIs[-1]-1 < -self.stop and len(posic) <= d_max and not include_var):
                        break
            
                   
            #Updating variable list        
            lista.remove(best_index)
            if include_var:
                posic[r] = best_index
            else:
                posic.remove(best_index)
             
            #Updating list of mutual info
            MIs.append(MI_best)
            stds.append(std_best)

            #Verbose
            if verbose: print("- Round={:2d} --- Î={:.2f} --- Selected Features={}".format(r,np.round(MI_best,2),posic))
           
                
        ## Output ##
        self.mi_list=MIs
        self.stds_list=stds
        self.var_list=posic
        self.var_bool=np.zeros(d, dtype=bool)
        
        for var in self.var_list: 
            if var!=None: self.var_bool[var]=True
        
    def transform(self,X): return X[:,self.var_bool]
    
    def get_mi(self): return self.mi_list, self.stds_list
    
    def get_vars(self): return self.var_list
    
    def plot_mi(self): 
        
        l,s=self.get_mi()
        plt.errorbar(list(range(1,len(l)+1)), l, yerr=(s/np.sqrt(self.n)))
        #plt.title("Mutual Information")
        plt.xlabel("Features (added or removed)")
        plt.ylabel("Mutual Information")
        plt.show()