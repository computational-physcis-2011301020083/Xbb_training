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
if args.type=="Dijets":
    predict=predict[predict[:,0]==1]
    name="Dijets"
if args.type=="Top":
    predict=predict[predict[:,2]==1]
    name="Top"
predict=predict[(predict[:,6]<=300.) & (predict[:,6]>=50.)]
predict1=predict[predict[:,3]>=0.5]
bins = np.linspace(50, 300, 100)
plt.figure(1)
if args.type=="Dijets":
    plt.hist(predict[:,6],weights=predict[:,4]/np.sum(predict[:,4]),bins=bins,label="Dijets mass",histtype="step")
if args.type=="Top":
    plt.hist(predict[:,6],weights=predict[:,4]/np.sum(predict[:,4]),bins=bins,label="Top mass",histtype="step")
plt.hist(predict1[:,6],weights=predict1[:,4]/np.sum(predict1[:,4]),bins=bins,label="Test mass",histtype="step")
plt.legend(loc='upper right', fontsize="x-small")
plt.yscale("log", nonposy="clip")
plt.xlabel("mass [GeV]")
plt.ylabel("Events fraction")
plt.ylim(top=0.1)
plt.text(75,0.1*0.6,r'$\sqrt{s}$=13TeV')
if args.type=="Dijets":
    plt.text(75,0.1*0.4,r'Hbb vs. Dijets')
if args.type=="Top":
    plt.text(75,0.1*0.4,r'Hbb vs. Top')
plt.title("Jet mass")
plt.savefig("Jetmass_"+name+"_"+args.name+".pdf")



