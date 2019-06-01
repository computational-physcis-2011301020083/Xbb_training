import pandas as pd
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()

feature_names = [u'fat_jet', u'subjet_VR_1', u'subjet_VR_2', u'subjet_VR_3', 'weight']
fat_list=['pt',  'eta', 'mass','Angularity', 'Aplanarity', 'C2', 'D2', 'FoxWolfram20','KtDR', 'Qw', 'PlanarFlow', 'Split12', 'Split23','Tau21_wta', 'Tau32_wta','ZCut12', 'e3']
VR1_list=['IP2D_pb_1', 'IP2D_pc_1', 'IP2D_pu_1', 'IP3D_pb_1', 'IP3D_pc_1', 'IP3D_pu_1', 'JetFitter_N2Tpair_1', 'JetFitter_dRFlightDir_1', 'JetFitter_deltaeta_1', 'JetFitter_deltaphi_1', 'JetFitter_energyFraction_1', 'JetFitter_mass_1', 'JetFitter_massUncorr_1', 'JetFitter_nSingleTracks_1', 'JetFitter_nTracksAtVtx_1', 'JetFitter_nVTX_1', 'JetFitter_significance3d_1', 'SV1_L3d_1', 'SV1_Lxy_1', 'SV1_N2Tpair_1', 'SV1_NGTinSvx_1', 'SV1_deltaR_1', 'SV1_dstToMatLay_1', 'SV1_efracsvx_1', 'SV1_masssvx_1', 'SV1_pb_1', 'SV1_pc_1', 'SV1_pu_1', 'SV1_significance3d_1', 'deta_1', 'dphi_1', 'dr_1', 'eta_1', 'pt_1', 'rnnip_pb_1', 'rnnip_pc_1', 'rnnip_ptau_1', 'rnnip_pu_1']
VR2_list=['IP2D_pb_2', 'IP2D_pc_2', 'IP2D_pu_2', 'IP3D_pb_2', 'IP3D_pc_2', 'IP3D_pu_2', 'JetFitter_N2Tpair_2', 'JetFitter_dRFlightDir_2', 'JetFitter_deltaeta_2', 'JetFitter_deltaphi_2', 'JetFitter_energyFraction_2', 'JetFitter_mass_2', 'JetFitter_massUncorr_2', 'JetFitter_nSingleTracks_2', 'JetFitter_nTracksAtVtx_2', 'JetFitter_nVTX_2', 'JetFitter_significance3d_2', 'SV1_L3d_2', 'SV1_Lxy_2', 'SV1_N2Tpair_2', 'SV1_NGTinSvx_2', 'SV1_deltaR_2', 'SV1_dstToMatLay_2', 'SV1_efracsvx_2', 'SV1_masssvx_2', 'SV1_pb_2', 'SV1_pc_2', 'SV1_pu_2', 'SV1_significance3d_2', 'deta_2', 'dphi_2', 'dr_2', 'eta_2', 'pt_2', 'rnnip_pb_2', 'rnnip_pc_2', 'rnnip_ptau_2', 'rnnip_pu_2']
VR3_list=['IP2D_pb_3', 'IP2D_pc_3', 'IP2D_pu_3', 'IP3D_pb_3', 'IP3D_pc_3', 'IP3D_pu_3', 'JetFitter_N2Tpair_3', 'JetFitter_dRFlightDir_3', 'JetFitter_deltaeta_3', 'JetFitter_deltaphi_3', 'JetFitter_energyFraction_3', 'JetFitter_mass_3', 'JetFitter_massUncorr_3', 'JetFitter_nSingleTracks_3', 'JetFitter_nTracksAtVtx_3', 'JetFitter_nVTX_3', 'JetFitter_significance3d_3', 'SV1_L3d_3', 'SV1_Lxy_3', 'SV1_N2Tpair_3', 'SV1_NGTinSvx_3', 'SV1_deltaR_3', 'SV1_dstToMatLay_3', 'SV1_efracsvx_3', 'SV1_masssvx_3', 'SV1_pb_3', 'SV1_pc_3', 'SV1_pu_3', 'SV1_significance3d_3', 'deta_3', 'dphi_3', 'dr_3', 'eta_3', 'pt_3', 'rnnip_pb_3', 'rnnip_pc_3', 'rnnip_ptau_3', 'rnnip_pu_3']
feature_name={u'fat_jet':fat_list,u'subjet_VR_1':VR1_list,u'subjet_VR_2':VR2_list,u'subjet_VR_3':VR3_list}

new_file_name=args.path.split("/")[-1]
new_hdf5 = h5py.File("FlattenData/"+new_file_name, 'w')
df = pd.read_hdf(args.path)
for i in feature_names:
  if i != "weight":	
    df1=df[feature_name[i]]
    start=0
    end=df1.values.shape[0]
    new_hdf5.create_dataset(i, data=df1.values)
    #save_data = new_hdf5.get(i)
    #data=df1.values[:,:]
    #print data
    #save_data[start:end, :] = data
    #print df1.values[0:10,:]
    #print df1
  if i=="weight":
    df1=df[i]
    new_hdf5.create_dataset(i,data=df1.values)




