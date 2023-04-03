#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 00:39:52 2023

@author: ric
"""

from kana_tools import *
#from kana_tools import input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class selection_facteur(input):
    """
    class feature selection,
    le but de cette classe est de prendre en entree un csv, ensuite de reduire le nombre de colonne
    donc de variable, de facteur, qui expliquerait une colonne cible en particulier.
    """
    def __init__(self,path="",index="",separateur_csv="",b_swarm_plot=True,target_col="",direct_df=pd.DataFrame()):
        
        if not direct_df.empty:
            self.data = direct_df
        else:
            super().__init__(path,index,separateur_csv)
        self.b_plot_swarm = b_swarm_plot
        self.target = target_col
    def verifications(self):
        if self.b_plot_swarm:
            from seaborn import swarmplot
            df_to_plot = pd.melt(self.data,id_vars=self.target,var_name="facteurs",value_name="valeurs")
            print(">> melt df before swar plot: \n")
            print(df_to_plot)
            plt.rcParams["figure.figsize"] = (12,9)
            swarmplot(data=df_to_plot,x="facteurs",y="valeurs",hue=self.target)
        print(">> FREQUENCE DES VALUERS POSSIBLES DE TARGET:")
        freq_target = self.data['target'].value_counts()
        df_freq_target = freq_target.to_frame(name='count')
        df_freq_target.reset_index(inplace=True)
        df_freq_target["index"] = df_freq_target["index"].astype(str)
        print(df_freq_target)
        sns.barplot(data=df_freq_target,y='count',x="index")
            
    def split_data(self,df):
        from sklearn.model_selection import train_test_split
        X = df.drop([self.target],axis=1)
        Y = df[self.target]
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True,stratify=Y,random_state=42)
        print(f"tailles de X_train:{X_train.shape}")
        print(f"tailles de X_test:{X_test.shape}")
        return X_train,X_test,Y_train,Y_test
    
    def variance_approche(self,X_train):
        
        from sklearn.preprocessing import MinMaxScaler
        df_var = X_train.var(axis=0)
        df_var = df_var.to_frame(name="variance_des_facteurs")
        print(">> df_variance des facteurs:")
        print(df_var)
        df_var.reset_index(inplace=True)
        df_var["index"] = df_var["index"].astype(str)
        sns.barplot(data=df_var,y='variance_des_facteurs',x="index")
        scaler = MinMaxScaler()
        scaled_x_train = scaler.fit_transform(X_train)
        var_x_train_scaled = scaled_x_train.var(axis=0)
        df_scaled_x_train_var = pd.DataFrame(var_x_train_scaled,index=X_train.columns)
        df_scaled_x_train_var.reset_index(inplace=True)
        df_scaled_x_train_var.rename(columns={"index":"facteurs",0:"variances"},inplace=True)
        print(">> df de X_train reduite:")
        print(df_scaled_x_train_var)
        sns.barplot(data=df_scaled_x_train_var,x="facteurs",y="variances")
        plt.show()
        
    def kbest_features_approche(self,b_k_connue=True,k=0):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import f1_score
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import mutual_info_classif
        X_train,X_test,Y_train,Y_test = self.split_data(self.data)
        if not b_k_connue:
            l_f1_score = []
            gbc = GradientBoostingClassifier(max_depth=5,random_state=42)
            
            for k in range(1,len(self.data.columns.values)):
                selector = SelectKBest(mutual_info_classif,k=k)
                selector.fit(X_train,Y_train)
                sel_X_train = selector.transform(X_train)
                sel_X_test = selector.transform(X_test)
                gbc.fit(sel_X_train,Y_train)
                kbest_preds = gbc.predict(sel_X_test)
                f1_score_mod = round(f1_score(Y_test,kbest_preds,average='weighted'),3)
                l_f1_score.append(f1_score_mod)
            print(l_f1_score)
            df_scores = pd.DataFrame(l_f1_score,columns=['f1 scores'])
            sns.barplot(data=df_scores,x=df_scores.index,y='f1 scores')
        else:
            selector = SelectKBest(mutual_info_classif,k=k)
            selector.fit(X_train,Y_train)
            selected_feature_mask = selector.get_support()
            selected_feat = X_train.columns[selected_feature_mask]
            print(">> les facteurs qui devraient être retenus sont:")
            print(selected_feat)
    def model_choisi(self):
        from sklearn.ensemble import GradientBoostingClassifier
        gbc = GradientBoostingClassifier(max_depth=5,random_state=42)
        return gbc
        
    def boruta_approche(self,X_train,X_test,Y_train,Y_test):
        from boruta import BorutaPy
        from sklearn.metrics import f1_score
        gbc = self.model_choisi()
        boruta_selector = BorutaPy(gbc,random_state=42)
        boruta_selector.fit(np.array(X_train),np.array(Y_train).ravel())
        sel_x_train = boruta_selector.transform(X_train.values)#filtrage ici
        sel_x_test = boruta_selector.transform(X_test.values)
        gbc.fit(sel_x_train,Y_train)
        boruta_preds = gbc.predict(sel_x_test)
        boruta_f1_score = round(f1_score(X_train,boruta_preds,average='weighted'),3)
        print(">> les caractérisiques à retenir d'après l'approche boruta:")
        print(boruta_f1_score)
        
    def scoring(self,df=pd.DataFrame(),X_train=pd.DataFrame(),X_test=pd.DataFrame(),Y_train=pd.DataFrame(),Y_test=pd.DataFrame()):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import f1_score
        gbc = GradientBoostingClassifier(max_depth=5,random_state=42)
        if X_train.empty:
            X_train,X_test,Y_train,Y_test = self.split_data(df)
        
        gbc.fit(X_train,Y_train)
        preds = gbc.predict(X_test)
        f1_score_mod = round(f1_score(Y_test,preds,average='weighted'),3)# le poids est pour corriger la difference de frequence entre les labels
        print(f"le score du model avec le dataframe et les facteurs choisi: {f1_score_mod}")
        return f1_score_mod