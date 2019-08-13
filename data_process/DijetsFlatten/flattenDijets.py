import pandas as pd
import glob,h5py
import argparse,math,os
parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", dest='path',  default="", help="path")
args = parser.parse_args()


features_group1=['weight','XbbScoreQCD','XbbScoreHiggs','XbbScoreTop','mass','pt','eta'] #0-7
features_group2=['JetFitter_N2Tpair_1', 'JetFitter_dRFlightDir_1', 'JetFitter_deltaeta_1', 'JetFitter_deltaphi_1', 'JetFitter_energyFraction_1', 'JetFitter_mass_1', 'JetFitter_massUncorr_1', 'JetFitter_nSingleTracks_1', 'JetFitter_nTracksAtVtx_1', 'JetFitter_nVTX_1', 'JetFitter_significance3d_1','SV1_L3d_1', 'SV1_Lxy_1', 'SV1_N2Tpair_1', 'SV1_NGTinSvx_1', 'SV1_deltaR_1', 'SV1_dstToMatLay_1', 'SV1_efracsvx_1', 'SV1_masssvx_1', 'SV1_significance3d_1','rnnip_pb_1', 'rnnip_pc_1', 'rnnip_ptau_1', 'rnnip_pu_1','JetFitter_N2Tpair_2', 'JetFitter_dRFlightDir_2', 'JetFitter_deltaeta_2', 'JetFitter_deltaphi_2', 'JetFitter_energyFraction_2', 'JetFitter_mass_2', 'JetFitter_massUncorr_2', 'JetFitter_nSingleTracks_2', 'JetFitter_nTracksAtVtx_2', 'JetFitter_nVTX_2', 'JetFitter_significance3d_2', 'SV1_L3d_2', 'SV1_Lxy_2', 'SV1_N2Tpair_2', 'SV1_NGTinSvx_2', 'SV1_deltaR_2', 'SV1_dstToMatLay_2', 'SV1_efracsvx_2', 'SV1_masssvx_2', 'SV1_significance3d_2', 'rnnip_pb_2', 'rnnip_pc_2', 'rnnip_ptau_2', 'rnnip_pu_2','JetFitter_N2Tpair_3', 'JetFitter_dRFlightDir_3', 'JetFitter_deltaeta_3', 'JetFitter_deltaphi_3', 'JetFitter_energyFraction_3', 'JetFitter_mass_3', 'JetFitter_massUncorr_3', 'JetFitter_nSingleTracks_3', 'JetFitter_nTracksAtVtx_3', 'JetFitter_nVTX_3', 'JetFitter_significance3d_3', 'SV1_L3d_3', 'SV1_Lxy_3', 'SV1_N2Tpair_3', 'SV1_NGTinSvx_3', 'SV1_deltaR_3', 'SV1_dstToMatLay_3', 'SV1_efracsvx_3', 'SV1_masssvx_3', 'SV1_significance3d_3', 'rnnip_pb_3', 'rnnip_pc_3', 'rnnip_ptau_3', 'rnnip_pu_3'] #7-79
features_group3=['Aplanarity','ZCut12','PlanarFlow','KtDR','Angularity','FoxWolfram20','Tau21_wta', 'Tau32_wta','C2','D2','Qw','Split12', 'Split23','e3'] #79-93

new_file_name=args.path.split("/")[-1]
new_hdf5 = h5py.File("../DataVRGhost/FlattenData3a/MergedDijets/"+new_file_name, 'w')
df = pd.read_hdf(args.path)[features_group1+features_group2+features_group3]
Data=df.values

new_hdf5.create_dataset("data",data=Data)




