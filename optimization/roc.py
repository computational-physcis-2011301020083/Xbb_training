import math,os,glob,h5py
import numpy as np
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import ROOT,math,os,glob,h5py
import argparse
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
parser.add_argument("--name", dest='name',  default="", help="name")
parser.add_argument("--type", dest='type',  default="", help="type")
args = parser.parse_args()

load_file=h5py.File(args.path,'r')
predict=load_file.get("predict")
predict=np.reshape(predict,(predict.shape[0],predict.shape[1]))
predict=predict[(predict[:,6]<=300.) & (predict[:,6]>=50.)]
predict=predict[(predict[:,7]<=2000.) & (predict[:,7]>=200.)]
if args.type=="Dijets":
    predict=predict[predict[:,2]==0]
    name="Dijets"
if args.type=="Top":
    predict=predict[predict[:,0]==0]
    name="Top"
y=predict[:,1]
score=predict[:,3]
Xbb=predict[:,5]
w=predict[:,4]
print y.shape,score.shape,w.shape
eff_bkg,eff_signal,thres=roc_curve(y,score,sample_weight=w)
eff_bkg1,eff_signal1,thres1=roc_curve(y,Xbb,sample_weight=w)
plt.figure(1)
plt.plot(eff_signal,np.power(eff_bkg,-1.0),label="Test")
plt.plot(eff_signal1,np.power(eff_bkg1,-1.0),label="XbbScoreHiggs")
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.yscale("log", nonposy="clip")
plt.xlim(left=0.2)
plt.ylim(top=1e5)
plt.legend(loc='best')
plt.text(0.25,1e5*0.4,r'$\sqrt{s}$=13TeV')
if args.type=="Dijets":
    plt.text(0.25,1e5*0.2,r'Hbb vs. Dijets')
if args.type=="Top":
    plt.text(0.25,1e5*0.2,r'Hbb vs. Top')
plt.text(0.25,1e5*0.1,r'pt:[200,2000]GeV')
plt.text(0.25,1e4*0.5,r'mass:[50,300]GeV')
plt.title("ROC curve")
plt.savefig("Roc_"+name+"_"+args.name+".pdf")



