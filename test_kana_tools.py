#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 2 mars 2023

@author: karl
'''
#from kaggle.kana_tools import preprocessing,reduction_tools,input,indep_between_two_cat

from kana_tools import preprocessing,reduction_tools,input,indep_between_two_cat,famd_prince,FAMDP,ana_factoriel
from kana_tools import cross_tab,cross_tab_modality
from ml_kana_tools import selection_facteur
import pandas as pd
import numpy as np
from datetime import datetime



#****************************** test algo PCA *************************************************************************************************************
# 
# path = r'C:\Users\karl\Documents\datas\breast_cancer.csv'
# i_input = input(path,index="id_sample",separateur_csv=";")
# i_preprocess = preprocessing(i_input.data,col_label="pam50")
# X,Y = i_preprocess.if_labels()
# X_scaled = i_preprocess.centree_reduite(X,b_plot=False)
# i_reduc = reduction_tools(df_cible=X_scaled,Y=Y)
# X_pca = i_reduc.acp_biplot(biplot=True,out_folder=r'C:\Users\karl\Documents\datas')

#*****************************  test independance 2 variables cat  *********************************************************************************************************

# _____________________test simple de khi 2____________________
# path = r"C:\Users\karl\Documents\datas\data_canal_test\tcont_G11_count.csv"
# i_input = input(path,separateur_csv=";",index="short_err")
# df = i_input.data
# i_test = indep_between_two_cat(df,"error","log_format_hdd")
# i_test.test_khi2(b_t_cont=False)

#______________________test complet ___________________________

# path = r"C:\Users\karl\Documents\datas\data_canal_test\BankChurners.csv"
# i_input = input(path,separateur_csv=",")
# df = i_input.data
# i_test = indep_between_two_cat(df,"Card_Category","Attrition_Flag")
# i_test.test_khi2(b_t_cont=True)
# i_test.heat_map_dependance()
# # v de cramer pour deux variables dans la table:
# i_test.v_cramer()
# ici un p-value qui montre que les variables sont dependantes mais vu la value du v cramer
#cela suggere que l'echantillon n est pas assez representatif.
#_____________________test avec un v cramer entre 0.13 et 0.14
# dataset = np.array([[4, 13, 17, 11], [4, 6, 9, 12],
#                     [2, 7, 4, 2], [5, 13, 10, 12],
#                     [5, 6, 14, 12]])

# df = pd.DataFrame(dataset)
# print(df)
# i_test = indep_between_two_cat(df,"ligne","colonne")
# i_test.test_khi2(b_t_cont=False)
# # v de cramer pour deux variables dans la table:
# i_test.v_cramer()

#************************************** test famd *****************************************************************************
#___________________  exemple 1 _______________________________________________
# path = r'/home/ric/Documents/pycode/py_code/telco_cleaned_renamed.csv'
# i_input = input(path,separateur_csv=",")
# df_cible = i_input.get_df_cible()
# i_famd = famd_prince(nb_composante=2,nb_iter=3,meth="famd",df=df_cible)
# i_famd.get_coord(col_label_name="Churn")
# i_famd_2 = famd_prince(nb_composante=2,nb_iter=3,meth="MFA",df=df_cible)
# i_famd_2.get_coord(col_label_name="Churn")

#__________________  exemple 2 _________________________________________________


# ## Import des bibliothèques nécessaires
# import plotly.express as px
# import pandas as pd
# import prince
# from plotly.offline import plot
# # Chargement des données
# path = r'/home/ric/Documents/pycode/py_code/beers.csv'
# i_input = input(path,separateur_csv=",")
# df = i_input.get_df_cible()
# df.dropna(inplace=True)
# print(df.columns)
# # Création d'un objet FAMD avec les données chargées
# famd = prince.FAMD(n_components=2, random_state=0)
# famd.fit(df)

# # Création d'un dataframe pour les individus avec les axes FAMD

# row_df = famd.row_coordinates(df).rename(columns={0:'PC1',1:'PC2'})
# print("row_df")
# print(famd.row_coordinates(df))
# print(row_df)
# # Création d'un dataframe pour les modalités avec les axes FAMD
# col_df = famd.column_correlations(df).rename(columns={0:'PC1',1:'PC2'})
# print("col_df")
# print(famd.column_correlations(df))
# print(col_df)
# # pourcentage de la variance expliquée:
# print(famd.explained_inertia_)
# # Création d'un graphique scatter plot pour les individus
# fig1 = px.scatter(row_df, x='PC1', y='PC2', hover_name=df.index,color_discrete_sequence=['blue'],
#                   title=f'FAMD: {famd.explained_inertia_[0]:.2%} (PC1) + {famd.explained_inertia_[1]:.2%} (PC2)')

# # Création d'un graphique scatter plot pour les modalités
# fig2 = px.scatter(col_df, x='PC1', y='PC2', hover_name=col_df.index, color_discrete_sequence=['blue'],
#                   title=f'FAMD: {famd.explained_inertia_[0]:.2%} (PC1) + {famd.explained_inertia_[1]:.2%} (PC2)')

# # Affichage des graphiques

# plot(fig1, auto_open=True,filename="graphe des individus.html")
# plot(fig2, auto_open=True,filename="graphe des variables.html")


#__________________________ exemple 3 _________________________________________

# import matplotlib.pyplot as plt
# import prince


# class FADM_visualizer:
#     def __init__(self, data, categorical_cols):
#         self.data = data
#         self.categorical_cols = categorical_cols
#         self.famd = prince.FAMD(n_components=2, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42)
#         self.famd.fit(self.data)
#         self.fig, self.axs = plt.subplots(ncols=2, figsize=(15,5))

#     def plot_dispersion(self):
#         # Select only the first two categorical columns
#         categorical_cols_to_plot = self.data[self.categorical_cols].iloc[:, :2]
        
#         self.famd.plot_row_coordinates(self.data, ax=self.axs[0], color_labels=categorical_cols_to_plot, ellipse_outline=False)
#         self.famd.plot_row_coordinates(self.data, ax=self.axs[1], color_labels=categorical_cols_to_plot, ellipse_outline=True)
#         self.axs[0].set_title('Dispersion des facteurs sans ellipses')
#         self.axs[1].set_title('Dispersion des facteurs avec ellipses')
#         plt.show()


# path = r'/home/ric/Documents/pycode/py_code/telco_cleaned_renamed.csv'
# i_input = input(path,separateur_csv=",")
# df_cible = i_input.get_df_cible()
# print(df_cible.dtypes)
# categ_col = df_cible.select_dtypes(include=['object']).columns
# print("===== colonne avec des donnees categoriel ==============")
# print(categ_col)
# fadm_viz = FADM_visualizer(df_cible, categ_col)
# fadm_viz.plot_dispersion()

#************************************** test class ana_factoriel (AC) *************************************************************

#******************************exemple 1******************************

# path = r'/home/ric/Documents/pycode/py_code/afc_fanalysis/afc_python/FrequentationNov2017.csv'
# i_input = input(path,separateur_csv=";")
# df_cible = i_input.get_df_cible()

# print(df_cible.dtypes)
# i_afc = ana_factoriel(df_cible,l_row_desact=[2,7],col_index="Pays")
# i_afc.afc()

#******************************exemple 2******************************
# path = r"/home/ric/Documents/pycode/py_code/afc_fanalysis/afc_python/afc_exemple2.csv"
# i_input = input(path,separateur_csv=";")
# df_cible = i_input.get_df_cible()

# i_afc = ana_factoriel(df_cible,col_index="Prof")
# i_afc.afc()


#************************************** test class ana_factoriel (MCA) *************************************************************

#******************************exemple 1******************************

# path = r"/home/ric/Documents/pycode/py_code/afc_fanalysis/afc_python/mca_exemple.csv"
# i_input = input(path,separateur_csv=";")
# df_cible = i_input.get_df_cible()
# df_cible["Welcome"] = df_cible["Welcome"].astype(str)
# i_af = ana_factoriel(df_cible,col_index="Customer")
# i_af.mca()

#******************************exemple 1******************************

# path = r"/home/ric/Documents/pycode/py_code/data_canal_test/df_hdd_max_rec.csv"
# i_input = input(path,separateur_csv=",")
# df_cible = i_input.get_df_cible()
# df_cible = df_cible.head(10000)
# df_cible.drop(['Unnamed: 0','size_hdd','plage_pvr','Unnamed: 9'],axis=1,inplace=True)
# df_cible = df_cible.astype(str)
# print(df_cible.columns)
# i_af = ana_factoriel(df_cible,col_index="msd")
# i_af.mca()






#*********************************** table disjonctif complet ***************************************

# a faire!!!!!!!!


# path = r'/home/ric/Documents/pycode/py_code/afc_fanalysis/afc_python/FrequentationNov2017.csv'
# i_input = input(path,separateur_csv=";")
# df_cible = i_input.get_df_cible()
# i_ct = cross_tab_modality(df_cible)
# i_ct.t_contin_margin()



#********************************** features selection ***********************************************


from sklearn.datasets import load_wine
wine_data = load_wine()
df_wine = pd.DataFrame(data=wine_data.data,columns=wine_data.feature_names)
df_wine['target'] = wine_data.target
print(df_wine)
df_wine.drop(['od280/od315_of_diluted_wines'],axis=1,inplace=True)
i_select = selection_facteur(b_swarm_plot=False,target_col="target",direct_df=df_wine)
i_select.verifications()
f1_score_allfact = i_select.scoring(df=df_wine)
X_train,X_test,Y_train,Y_test=i_select.split_data(df_wine)
# i_select.variance_approche(X_train)
# #drop some features after seeing variances
# X_train_var = X_train.drop(['ash','magnesium'],axis=1)
# X_test_var = X_test.drop(['ash','magnesium'],axis=1)
# f1_score_var_app = i_select.scoring(X_train=X_train_var,X_test=X_test_var,Y_train=Y_train,Y_test=Y_test)
# i_select.kbest_features_approche(b_k_connue=True)
# i_select.kbest_features_approche(b_k_connue=False,k=3)
print(X_train)
print(type(X_train))
print(X_train.dtypes)
i_select.boruta_approche(X_train,X_test,Y_train,Y_test)