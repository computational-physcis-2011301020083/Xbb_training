import argparse
import os,glob,h5py,ROOT,shutil
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
parser.add_argument("--model", dest='model',  default="", help="model")
args = parser.parse_args()
import tensorflow as tf
from keras import backend as K
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

features1=['XbbScoreQCD', 'XbbScoreHiggs', 'XbbScoreTop','mcEventWeight','mass','pt', 'eta']
features2=['JetFitter_N2Tpair', 'JetFitter_dRFlightDir', 'JetFitter_deltaeta', 'JetFitter_deltaphi', 'JetFitter_energyFraction', 'JetFitter_mass', 'JetFitter_massUncorr', 'JetFitter_nSingleTracks', 'JetFitter_nTracksAtVtx', 'JetFitter_nVTX', 'JetFitter_significance3d','SV1_L3d', 'SV1_Lxy', 'SV1_N2Tpair', 'SV1_NGTinSvx', 'SV1_deltaR', 'SV1_dstToMatLay', 'SV1_efracsvx', 'SV1_masssvx', 'SV1_significance3d','rnnip_pb', 'rnnip_pc', 'rnnip_ptau', 'rnnip_pu']
paths=args.path
h0=pd.read_hdf(paths,"fat_jet")[features1]
h1=pd.read_hdf(paths,"subjet_VRGhostTag_1")[features2]
h2=pd.read_hdf(paths,"subjet_VRGhostTag_2")[features2]
h3=pd.read_hdf(paths,"subjet_VRGhostTag_3")[features2]
h=pd.concat([h0,h1,h2,h3], axis=1)
h["pt"] = (h["pt"]/1000.0).astype("float64")
h["mass"] = (h["mass"]/1000.0).astype("float64")
data=h.values[:,5:79]
XbbScore=h.values[:,0:3]

meanFile=h5py.File("meanstd1.h5","r")
mean_vector=meanFile.get("mean")
std_vector=meanFile.get("std")
data=(data-mean_vector)/std_vector
data=np.nan_to_num(data)
XbbScore=np.nan_to_num(XbbScore)

model_file="WeiAdm3bStd1Opt1.h5"
model_pre = keras.models.load_model(model_file)
predictions = model_pre.predict(data)
Data=np.hstack((predictions,XbbScore,h.values[:,3:6]))

new_file_name="./Prediction_"+args.path.split("/")[-1]
new_hdf5 = h5py.File(new_file_name, 'w')
new_hdf5.create_dataset("data",data=Data)


print "In the output file, the information of these columns are in order :"
print "PredictionScoreQCD,PredictionScoreHiggs,PredictionScoreTop,XbbScoreQCD,XbbScoreHiggs,XbbScoreTop,mcEventWeight,mass[GeV],pt[GeV]"
#print "Predictions: "
#print predictions[110:140,:]
#print "XbbScore: "
#print XbbScore[110:140,:]








