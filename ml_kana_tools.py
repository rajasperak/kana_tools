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
        gbc = GradientBoostingClassifier(max_depth=5,random_state=42)
        if  b_k_connue == False:
            l_f1_score = []
            for k in range(1,len(self.data.columns.values)):
                selector = SelectKBest(mutual_info_classif,k=k)
                selector.fit(X_train,Y_train)
                #je vais filtrer les facteurs retenu par le selector avant de tester le model
                sel_X_train = selector.transform(X_train)
                sel_X_test = selector.transform(X_test)
                gbc.fit(sel_X_train,Y_train)
                kbest_preds = gbc.predict(sel_X_test)
                f1_score_mod = round(f1_score(Y_test,kbest_preds,average='weighted'),3)
                l_f1_score.append(f1_score_mod)
            print(l_f1_score)
            df_scores = pd.DataFrame(l_f1_score,columns=['f1 scores'])
            sns.set()
            sns.barplot(x=df_scores.index, y='f1 scores', data=df_scores)
            plt.ylim(0, 1)
            plt.xlabel('Nb de facteurs')
        else:
            selector = SelectKBest(mutual_info_classif,k=k)
            selector.fit(X_train,Y_train)
            selected_feature_mask = selector.get_support()
            selected_feat = X_train.columns[selected_feature_mask]
            print(">> le masque de selection des facteurs retenus:")
            print(selected_feature_mask)
            print(">> les facteurs qui devraient être retenus sont:")
            print(selected_feat)
            sel_X_train = selector.transform(X_train)
            sel_X_test = selector.transform(X_test)
            gbc.fit(sel_X_train,Y_train)
            kbest_preds = gbc.predict(sel_X_test)
            f1_score_mod = round(f1_score(Y_test,kbest_preds,average='weighted'),3)
            print(f1_score_mod)
            
    def model_choisi(self):
        from sklearn.ensemble import GradientBoostingClassifier
        gbc = GradientBoostingClassifier(max_depth=5,random_state=42)
        return gbc
    
    def rfe_approche(self,step=1,k=0):
        from sklearn.feature_selection import RFE
        from sklearn.metrics import f1_score
        rfe_f1_score = []
        estimateur = self.model_choisi()
        X_train,X_test,Y_train,Y_test = self.split_data(self.data)
        
        if k==0:
            for k in range(1,len(self.data.columns.values)):
                rfe_selector = RFE(estimator=estimateur,n_features_to_select=k,step=step)
                rfe_selector.fit(X_train,Y_train)
                sel_x_train = rfe_selector.transform(X_train)
                sel_x_test = rfe_selector.transform(X_test)
                estimateur.fit(sel_x_train,Y_train)
                rfe_preds = estimateur.predict(sel_x_test)
                f1_score_rfe = round(f1_score(Y_test,rfe_preds,average="weighted"),3)
                rfe_f1_score.append(f1_score_rfe)
            df_scores = pd.DataFrame(rfe_f1_score,columns=['f1 scores'])
            sns.set()
            sns.barplot(x=df_scores.index, y='f1 scores', data=df_scores)
            plt.ylim(0, 1)
            plt.xlabel('Nb de facteurs')
        
        else:
            rfe_selector = RFE(estimator=estimateur,n_features_to_select=k,step=step)
            rfe_selector.fit(X_train,Y_train)
            selected_features_mask = rfe_selector.get_support()
            selected_feat = X_train.columns[selected_features_mask]
            print(">> les facteurs qui devraient être retenus sont:")
            print(selected_feat)
        
    def boruta_approche(self,X_train,X_test,Y_train,Y_test):
        from boruta import BorutaPy
        from sklearn.metrics import f1_score
        gbc = self.model_choisi()
        boruta_selector = BorutaPy(gbc,random_state=42)
        boruta_selector.fit(np.array(X_train),np.array(Y_train))
        sel_x_train = boruta_selector.transform(np.array(X_train))#filtrage ici
        sel_x_test = boruta_selector.transform(np.array(X_test))
        gbc.fit(sel_x_train,Y_train)
        boruta_preds = gbc.predict(sel_x_test)
        boruta_f1_score = round(f1_score(Y_test,boruta_preds,average='weighted'),3)
        selected_feat_mask = boruta_selector.support_
        selected_feat = X_train.columns[selected_feat_mask]
        print(">> les caractérisiques à retenir d'après l'approche boruta:")
        print(selected_feat)
        print(">> Avec un f1-score de :")
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