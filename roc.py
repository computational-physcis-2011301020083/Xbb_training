import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import ROOT,argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

load_file=h5py.File(args.path)
y=load_file.get("y")
score=load_file.get("score")
w=load_file.get("w")
print y.shape,score.shape,w.shape
eff_bkg,eff_signal,thres=roc_curve(y,score,sample_weight=w)
plt.figure(1)
plt.plot(eff_signal,np.power(eff_bkg,-1.0),label="test")
plt.xlabel('Signal Efficiency')
plt.ylabel('background Rejection')
plt.yscale("log", nonposy="clip")
plt.legend(loc='best')
plt.title("ROC curve")
plt.savefig("roc.pdf")
plt.show()


