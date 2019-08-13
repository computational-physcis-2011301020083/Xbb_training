import numpy as np
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

new_file_name=args.path.split("/")[-1]
new_hdf5 = h5py.File("../DataVRGhost/FlattenData3a/ReducedDijetsDSID/"+new_file_name, 'w')

f=h5py.File(args.path)
Data=f.get("data")
N=Data.shape[0]
select_index=np.random.choice(N,int(0.035*N),replace=False)
select_data=np.take(Data,select_index,axis=0)

new_hdf5.create_dataset("data",data=select_data)


