import numpy as np
import pandas as pd
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

feature_names = ['fat_jet', 'subjet_VR_1', 'subjet_VR_2', 'subjet_VR_3'] #, 'weight']
files=sorted(glob.glob(args.path+"/*.h5"))
save_f = h5py.File("MeanStd/mean_std.h5", 'w')
for i in feature_names:
  Bg_data=h5py.File(files[0],"r")
  bg_data=Bg_data.get(i+"/train")
  Signal_data=h5py.File(files[1],"r")
  signal_data=Signal_data.get(i+"/train")
  Top_data=h5py.File(files[2],"r")
  top_data=Top_data.get(i+"/train")
  print i,bg_data, signal_data, top_data
  cal_data = np.vstack((bg_data,signal_data,top_data))
  mean_vector = np.nanmean(cal_data, axis=0)
  std_vector = np.nanstd(cal_data, axis=0)
  save_f.create_dataset(i+"/mean",data=mean_vector)
  save_f.create_dataset(i+"/std",data=std_vector)




