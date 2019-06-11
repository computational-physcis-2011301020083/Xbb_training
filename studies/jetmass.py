import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import ROOT,argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

load_file=h5py.File(args.path)
predict=load_file.get("predict")
predict=np.reshape(predict,(predict.shape[0],predict.shape[1]))
predict=predict[predict[:,2]==1]
#predict=predict[predict[:,0]==1]
#name="Dijets"
name="Top"
predict=predict[(predict[:,7]<=300.) & (predict[:,7]>=50.)]

predict1=predict[predict[:,3]>=0.4]
bins = np.linspace(50, 300, 100)
plt.figure(1)
#plt.hist(predict[:,7],weights=predict[:,4]/np.sum(predict[:,4]),bins=bins,label="Dijets mass",histtype="step")
plt.hist(predict[:,7],weights=predict[:,4]/np.sum(predict[:,4]),bins=bins,label="Top mass",histtype="step")
plt.hist(predict1[:,7],weights=predict1[:,4]/np.sum(predict1[:,4]),bins=bins,label="Test mass",histtype="step")
plt.legend(loc='upper right', fontsize="x-small")
plt.yscale("log", nonposy="clip")
plt.xlabel("mass [GeV]")
plt.ylabel("Events fraction")
plt.ylim(top=1)

plt.text(75,1*0.4,r'$\sqrt{s}$=13TeV')
#plt.text(75,1*0.2,r'Hbb vs. Dijets')
plt.text(75,1*0.2,r'Hbb vs. Top')

plt.title("Jet mass")
plt.savefig("files/jetmass"+name+".pdf")
plt.show()




