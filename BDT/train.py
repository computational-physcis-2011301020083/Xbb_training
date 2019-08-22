from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt
import os,h5py
from sklearn.metrics import roc_curve, roc_auc_score

#Test git

#Load samples for training
f1=h5py.File("MergedHbb.h5",'r')
f2=h5py.File("MergedTop.h5",'r')
data_hbb=f1.get("data")
data_top=f2.get("data")
data_hbb=np.nan_to_num(data_hbb)
data_top=np.nan_to_num(data_top)
#print data_hbb.shape
train=np.vstack((data_hbb[0:20000,2:19],data_top[0:20000,2:19]))
values=np.hstack((np.full((20000, ), 1),np.full((20000, ), 0)))
#print train.shape,values.shape

#BDT training
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
bdt.fit(train, values)

#Get predictions
score=bdt.decision_function(train)

#Plot ROC curve and save
eff_bkg,eff_signal,thres=roc_curve(values,score)
plt.figure(1)
plt.plot(eff_signal,np.power(eff_bkg,-1.0),label="BDT")
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.yscale("log", nonposy="clip")
plt.xlim(left=0.2)
plt.ylim(top=1e3)
plt.legend(loc='best')
plt.text(0.3,1e3*0.4,r'$\sqrt{s}$=13TeV')
plt.text(0.3,1e3*0.2,r'Hbb vs. Top')
plt.savefig("BDT_roc.pdf")

