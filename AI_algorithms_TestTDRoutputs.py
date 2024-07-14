
#LIBS
import seaborn as sns #for heatmap
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


"""#f1_scores

This function calculates F1 scores, providing insight AI algorithms.
"""

def f1_scores(cm):

  tn = cm[0,0]
  fp = cm[0,1]
  fn = cm[1,0]
  tp = cm[1,1]

  accuracy = round((tn + tp) / (tp + fp + fn + tn),3)
  precision = round(tp / (tp + fp),3)
  recall = round(tp / (tp + fn),3)
  f1_score = round(2 * (precision * recall) / (precision + recall),3)

  return f"Accuracy:{accuracy} Precision :{precision} Recall:{recall} F1 Score :{f1_score}"

"""#algorithms"""
"""Default Machine Learning Algorithms"""

def algorithms(X_train,y_train,X_test,y_test,X_pca):
  cms=[] #all confusion_matrixs and model infos
  cm=[] #confusion matrix of the model trained with the columns given by the TDR/All columns algorithm
  cm_pca=[] #confusion matrix of the model trained with the columns given by the PCA algorithm

  clf= svm.SVC(kernel='rbf', C=1.0)
  clf.fit(X_train,y_train)
  y_predSVC = clf.predict(X_test)
  cm = confusion_matrix(y_test,y_predSVC)

  clf.fit(X_pca,y_train)
  y_pca_predSVC = clf.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_pca_predSVC)

  cms.append(["SVC",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  RFC=RandomForestClassifier()
  RFC.fit(X_train,y_train)
  y_predRF = RFC.predict(X_test)
  cm = confusion_matrix(y_test,y_predRF)

  RFC.fit(X_pca,y_train)
  y_pca_predRF = RFC.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_pca_predRF)

  cms.append(["RFC",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  logreg = LogisticRegression()
  logreg.fit(X_train,y_train)
  y_predLR = logreg.predict(X_test)
  cm = confusion_matrix(y_test,y_predLR)

  logreg.fit(X_pca,y_train)
  y_pca_predLR = logreg.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_pca_predLR)

  cms.append(["LOGREG",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  ###
  scaler = MinMaxScaler()
  X_train_s = scaler.fit_transform(X_train)
  X_test_s = scaler.transform(X_test)
  X_pca_s = scaler.transform(X_pca)
  ###

  gb_clf = GradientBoostingClassifier()
  gb_clf.fit(X_train_s, y_train)
  y_predSGB = gb_clf.predict(X_test_s)
  cm = confusion_matrix(y_test,y_predSGB)

  gb_clf.fit(X_pca_s, y_train)
  y_pca_predSGB = gb_clf.predict(X_test_s)
  cm_pca = confusion_matrix(y_test,y_pca_predSGB)

  cms.append(["GBC",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  model = XGBClassifier()
  model.fit(X_train, y_train)
  y_predEGB = model.predict(X_test)
  cm = confusion_matrix(y_test,y_predEGB)

  model.fit(X_pca, y_train)
  y_pca_predEGB = model.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_pca_predEGB)

  cms.append(["XGBC",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])


  gnb = GaussianNB()
  gnb.fit(X_train, y_train)
  y_predGNB = gnb.predict(X_test)
  cm = confusion_matrix(y_test,y_predGNB)

  gnb.fit(X_pca, y_train)
  y_pca_predGNB = gnb.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_pca_predGNB)

  cms.append(["GNB",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  model = LinearDiscriminantAnalysis()
  model.fit(X_train, y_train)
  y_predLDA = model.predict(X_test)
  cm = confusion_matrix(y_test,y_predLDA)

  model = LinearDiscriminantAnalysis()
  model.fit(X_pca, y_train)
  y_pca_predLDA = model.predict(X_test)
  cm_pca = confusion_matrix(y_test,y_predLDA)

  cms.append(["LDA",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  #In the hybrid model, the label predicted most times by previous models is assigned.
  y_predHBD=[]
  for i in range(len(y_test)):
    val=int(y_predSVC[i])+int(y_predRF[i])+int(y_predLR[i])+int(y_predSGB[i])+int(y_predEGB[i])+int(y_predGNB[i])+int(y_predLDA[i])
    if val > 3:
      y_predHBD.append(1)
    else:
      y_predHBD.append(0)
  cm = confusion_matrix(y_test,y_predHBD)

  y_pca_predHBD=[]
  for i in range(len(y_test)):
    val=y_pca_predRF[i]+y_pca_predRF[i]+y_pca_predLR[i]+y_pca_predSGB[i]+y_pca_predEGB[i]+y_pca_predLDA[i]+y_pca_predGNB[i]
    if val > 3:
      y_pca_predHBD.append(1)
    else:
      y_pca_predHBD.append(0)
  cm_pca = confusion_matrix(y_test,y_pca_predHBD)

  cms.append(["HYBRD",cm,f1_scores(cm),cm_pca,f1_scores(cm_pca)])

  return cms

"""#YZ"""

def YZ(df_train,df_test,ioscs,st=0):

  #For a training with all columns,add all columns to the list.
  ioscs.insert(0,list(range(df_train.shape[1]-1)))

  results=[]
  #After creating the column list according to "importance level", give them one by one for testing with default AI algorithms.
  for i,iosc in enumerate(ioscs):

    #Make the number of columns of PCA the same as the number of columns we recommend as a result of the TDR algorithm.
    pca=PCA(n_components=len(iosc))
    X_pca=pca.fit_transform(df_train.copy().values)

    X_train=df_train.copy().iloc[:,iosc].values
    y_train=df_train.copy().iloc[:,-1].values

    X_test=df_test.copy().iloc[:,iosc].values
    y_test=df_test.copy().iloc[:,-1].values

    results.append([len(iosc),  (X_train,y_train,X_test,y_test,X_pca)])

  for i in results:
    print(i)


dfMain=pd.read_excel("PublicSchoolBuildingsData.xlsx")
dfTrain, dfTest = train_test_split(dfMain, test_size=0.2, random_state=42)

ioscs=[[1,5,7,11,12,14,15,16,17,18],[2,7,10,11,12,14,15,18,19],[2,7,10,11,14,15,18]]
#Use the column ID by decreasing 1. This is the output of the TDR algorithm like that.
ioscs=[[x-1 for x in i] for i in ioscs]
YZ(dfTrain,dfTest,ioscs)



"""#HeatMap"""
# Creat HeatMap for one confisuon matrix output
cf_matrix = np.array([[82, 67], [7, 206]])
# Group Names
group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
# Prepare Labels
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2, 2)
# Create Heatmap
sns.heatmap(data=cf_matrix,
            annot=labels,
            cmap="Blues",
            fmt="s",
            cbar=True)
# Save Heatmap as SVG
plt.savefig("confusion_matrix.svg", format="svg", bbox_inches="tight")
# Show Heatmap
plt.show()
