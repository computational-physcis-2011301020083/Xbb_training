import pandas as pd
import numpy as np
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
parser.add_argument("--style", dest='style',  default="", help="style")
parser.add_argument("--dsid", dest='dsid',  default="", help="dsid")
args = parser.parse_args()

Files=sorted(glob.glob(args.path+"/*"+args.dsid+"*.h5"))
j=0
for i in Files:
        j=j+1
	f=h5py.File(i)
	Data=f.get("data")
	if j==1:
		Merged=Data
	else:
		Merged=np.vstack((Merged,Data))



SaveFile= h5py.File("../DataVRGhost/FlattenData3a/MergedDijetsDSID/MergedDijets."+args.dsid+".h5", 'a')
SaveFile.create_dataset("data",data=Merged)



