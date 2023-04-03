# kana_tools
k (karl) ana(analyses): outils les plus utilisés dans un cadre professionnel jusq'ici. le but est d'avoir les outils nécessaire pour un pipeline d'analyse, à disposition rapidement, pour des analyses statistiques de base, des analyses factorielles, des analyses de réduction de dimensions,du clustering et de features selections:

contenus:
- class numeric_corr: destiné à analyser la correlation entre les variable numériques dans un dataframe                                                     
- class indep_between_two: destiné à analyser si dépendance ou no il y a entre deux colonnes d'un dataframe de valeur numériques.                           
- class prepocessing: destiné à préparer le dataframe et à avoir une première idée de ce qu'il y a à l'interieur. il y a quelques fonction qui permet également de faire quelques nettoyage.
- class reduction_tools : destiné à reduire le nombre de variables à analyser, éliminer les variables corrélées, pour eventuellement lancer des modèles dessus, pour de meilleurs résultats.   
- class afc: analyse des correspondances complete, pour l'exploration, l'analyse ou pour un prétraitement en vue d'un clustering
- class fadm: analyse factorielle en donnée mixte, dans le même esprit que précedemment
- class multiv_ana_tools: destiné à creer des tables disjonctifs, des tables de burt, des tests de corrélation entre des variables mixtes. On aura des outils d'analyse factorielle comme l'ACM,l'AFDM.         
- class clustering: destiné à trouver des clusters eventuels. lire la description dans la classe.
- class selection_facteur:destiné à trouverles facteurs expliquant la variable cible. utile dans l'exploration également, ou pour avoir une performance dans les modèles à déployer. 
