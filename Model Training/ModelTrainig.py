from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, mean_absolute_error, precision_recall_fscore_support, confusion_matrix, accuracy_score


data = pd.read_csv("final.csv")

x = data.drop('Class', axis = 1)
y = data['Class']

min = [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2000, 0, 0, 0, 0, 0, 0, 0]
max = [120, 360, 1.1, 5.5, 5.5, 1, 1, 1, 1, 700, 500, 100, 200, 100, 50, 100, 30000, 20, 1, 1, 1, 1, 1, 1]

# min-max scaler formula
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

i = 0
for col in x.columns:
    x[col] = np.array(x[col].apply(lambda x:(x - min[i]) / (max[i] - min[i])), dtype=np.float32)
    i += 1



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly', degree = 24)
RF = RandomForestClassifier(n_estimators=1000)
XB = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.7, max_features=24, max_depth=1000)
CNN = MLPClassifier(max_iter=1000, solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(24, 24)) 
NB = GaussianNB()


evc = VotingClassifier(estimators= [('lr',lr),('dt',dt),('svm',svm),('RF',RF),('XB',XB),('CNN',CNN),('NB',NB)],voting = 'hard')


evc.fit(x_train,y_train)
evc_pred = evc.predict(x_test)
# lr.fit(x_train,y_train)
# lr_pred = lr.predict(x_test)
# dt.fit(x_train,y_train)
# dt_pred = dt.predict(x_test)
# svm.fit(x_train,y_train)
# svm_pred = svm.predict(x_test)
# RF.fit(x_train,y_train)
# RF_pred = RF.predict(x_test)
# XB.fit(x_train,y_train)
# XB_pred = XB.predict(x_test)
# CNN.fit(x_train,y_train)
# CNN_pred = CNN.predict(x_test)
# NB.fit(x_train,y_train)
# NB_pred = NB.predict(x_test)

f = open("data.txt", "w")

f.write("LR_f1 : " + str(f1_score(y_test, lr_pred)) + "\n")
f.write("LR_MAE : " + str(mean_absolute_error(y_test, lr_pred)) + "\n")
f.write("LR_RMSE : " + str(mean_squared_error(y_test, lr_pred)) + "\n")
f.write("LR_PRFS : " + str(precision_recall_fscore_support(y_test, lr_pred)) + "\n")
f.write("LR_CM : " + str(confusion_matrix(y_test, lr_pred)) + "\n")
f.write("LR_AS : " + str(accuracy_score(y_test, lr_pred)) + "\n")
f.write("\n\n")

f.write("dt_f1 : " + str(f1_score(y_test, dt_pred)) + "\n")
f.write("dt_MAE : " + str(mean_absolute_error(y_test, dt_pred)) + "\n")
f.write("dt_RMSE : " + str(mean_squared_error(y_test, dt_pred)) + "\n")
f.write("dt_PRFS : " + str(precision_recall_fscore_support(y_test, dt_pred)) + "\n")
f.write("dt_CM : " + str(confusion_matrix(y_test, dt_pred)) + "\n")
f.write("dt_AS : " + str(accuracy_score(y_test, dt_pred)) + "\n")
f.write("\n\n")

f.write("svm_f1 : " + str(f1_score(y_test, svm_pred)) + "\n")
f.write("svm_MAE : " + str(mean_absolute_error(y_test, svm_pred)) + "\n")
f.write("svm_RMSE : " + str(mean_squared_error(y_test, svm_pred)) + "\n")
f.write("svm_PRFS : " + str(precision_recall_fscore_support(y_test, svm_pred)) + "\n")
f.write("svm_CM : " + str(confusion_matrix(y_test, svm_pred)) + "\n")
f.write("svm_AS : " + str(accuracy_score(y_test, svm_pred)) + "\n")
f.write("\n\n")

f.write("RF_f1 : " + str(f1_score(y_test, RF_pred)) + "\n")
f.write("RF_MAE : " + str(mean_absolute_error(y_test, RF_pred)) + "\n")
f.write("RF_RMSE : " + str(mean_squared_error(y_test, RF_pred)) + "\n")
f.write("RF_PRFS : " + str(precision_recall_fscore_support(y_test, RF_pred)) + "\n")
f.write("RF_CM : " + str(confusion_matrix(y_test, RF_pred)) + "\n")
f.write("RF_AS : " + str(accuracy_score(y_test, RF_pred)) + "\n")
f.write("\n\n")

f.write("XB_f1 : " + str(f1_score(y_test, XB_pred)) + "\n")
f.write("XB_MAE : " + str(mean_absolute_error(y_test, XB_pred)) + "\n")
f.write("XB_RMSE : " + str(mean_squared_error(y_test, XB_pred)) + "\n")
f.write("XB_PRFS : " + str(precision_recall_fscore_support(y_test, XB_pred)) + "\n")
f.write("XB_CM : " + str(confusion_matrix(y_test, XB_pred)) + "\n")
f.write("XB_AS : " + str(accuracy_score(y_test, XB_pred)) + "\n")
f.write("\n\n")

f.write("CNN_f1 : " + str(f1_score(y_test, CNN_pred)) + "\n")
f.write("CNN_MAE : " + str(mean_absolute_error(y_test, CNN_pred)) + "\n")
f.write("CNN_RMSE : " + str(mean_squared_error(y_test, CNN_pred)) + "\n")
f.write("CNN_PRFS : " + str(precision_recall_fscore_support(y_test, CNN_pred)) + "\n")
f.write("CNN_CM : " + str(confusion_matrix(y_test, CNN_pred)) + "\n")
f.write("CNN_AS : " + str(accuracy_score(y_test, CNN_pred)) + "\n")
f.write("\n\n")

f.write("NB_f1 : " + str(f1_score(y_test, NB_pred)) + "\n")
f.write("NB_MAE : " + str(mean_absolute_error(y_test, NB_pred)) + "\n")
f.write("NB_RMSE : " + str(mean_squared_error(y_test, NB_pred)) + "\n")
f.write("NB_PRFS : " + str(precision_recall_fscore_support(y_test, NB_pred)) + "\n")
f.write("NB_CM : " + str(confusion_matrix(y_test, NB_pred)) + "\n")
f.write("NB_AS : " + str(accuracy_score(y_test, NB_pred)) + "\n")
f.write("\n\n")

f.write("evc_f1 : " + str(f1_score(y_test, evc_pred)) + "\n")
f.write("evc_MAE : " + str(mean_absolute_error(y_test, evc_pred)) + "\n")
f.write("evc_RMSE : " + str(mean_squared_error(y_test, evc_pred)) + "\n")
f.write("evc_PRFS : " + str(precision_recall_fscore_support(y_test, evc_pred)) + "\n")
f.write("evc_CM : " + str(confusion_matrix(y_test, evc_pred)) + "\n")
f.write("evc_AS : " + str(accuracy_score(y_test, evc_pred)) + "\n")
f.write("\n\n")

acc=evc.score(x_test, y_test)

f.write("accuracy : " + str(acc) + "\n")

with open("evc.pickle","wb") as f:
        pickle.dump(evc,f)