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
predict=predict[(predict[:,7]<=300.) & (predict[:,7]>=50.)]
predict=predict[(predict[:,6]<=500.) & (predict[:,6]>=200.)]
#predict=predict[(predict[:,6]<=1000.) & (predict[:,6]>=500.)]
#predict=predict[(predict[:,6]<=2000.) & (predict[:,6]>=1000.)]
predict=predict[predict[:,0]==0]
#predict=predict[predict[:,2]==0]
#name="DijetsPt2000"
name="TopPt500"


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
#plt.text(0.25,1e5*0.2,r'Hbb vs. Dijets')
plt.text(0.25,1e5*0.2,r'Hbb vs. Top')
plt.text(0.25,1e5*0.1,r'pt:[200,500]GeV')
#plt.text(0.25,1e5*0.1,r'pt:[500,1000]GeV')
#plt.text(0.25,1e5*0.1,r'pt:[1000,2000]GeV')
plt.text(0.25,1e4*0.5,r'mass:[50,300]GeV')

plt.title("ROC curve")
plt.savefig("files/roc"+name+".pdf")
plt.show()


