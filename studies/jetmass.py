import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import ROOT,argparse,math,os,glob,h5py
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

load_file=h5py.File(args.path)
pre=load_file.get("pre")[0:865593,:]
pre=np.reshape(pre,(pre.shape[0],4))
print pre.shape,pre[:,3]>50.
print pre[pre[:,3]>50.]
pre=pre[(pre[:,3]<=300.) & (pre[:,3]>=50.)]
pre1=pre[pre[:,1]>=0.25]
bins = np.linspace(50, 300, 100)
plt.figure(1)
plt.hist(pre[:,3],weights=pre[:,2],bins=bins,label="Dijets",histtype="step")
plt.hist(pre1[:,3],weights=pre1[:,2],bins=bins,label="Tagged",histtype="step")
plt.legend(loc='upper right', fontsize="x-small")
plt.yscale("log", nonposy="clip")
plt.xlabel("mass")
plt.ylabel("Events")
plt.title("Jet mass")
plt.savefig("jet_mass.pdf")
plt.show()




