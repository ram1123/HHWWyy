# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for HH->WWyy analysis.
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
from array import array
import time
import pandas
import optparse, json, argparse, math
import uproot
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
#from sklearn.utils import class_weight
from sklearn.metrics import log_loss
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from plotting.plotter import plotter
from numpy.testing import assert_allclose
from keras.callbacks import ModelCheckpoint

seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    #out = commands.getoutput(eos_cmd)
    return

def load_data(inputPath,variables,criteria):
    # Load dataset to .csv format file
    my_cols_list=variables
    keys=['HH','bckg']
    data = pandas.DataFrame(columns=my_cols_list)
    for key in keys :
        print('key: ', key)
        if 'HH' in key:
            sampleNames=key
            subdir_name = 'Signal'
            fileNames = ['ggF_SM_WWgg_qqqq_Hadded']
            target=1
        else:
            sampleNames= key
            subdir_name = 'Backgrounds'
            fileNames = ['GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8',
                         'GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8',
                         'QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8',
                         'QCD_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8',
                         'TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8'
                        ]
            target=0

        for filen in fileNames:
            if 'ggF_SM_WWgg_qqqq_Hadded' in filen:
                treename=['tagsDumper/trees/GluGluToHHTo_WWgg_qqqq_nodeSM_13TeV_HHWWggTag_2']
                process_ID = 'HH'
            elif 'QCD_Pt-30to40' in filen:
                treename=['tagsDumper/trees/QCD_Pt_30to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                process_ID = 'QCD'
            elif 'QCD_Pt-40toInf' in filen:
                treename=['tagsDumper/trees/QCD_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                process_ID = 'QCD'
            elif 'DiPhotonJetsBox_MGG' in filen:
                treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa',
                ]
                process_ID = 'DiPhoton'
            elif 'GJet_Pt-20to40' in filen:
                treename=['tagsDumper/trees/GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                process_ID = 'GJet'
            elif 'GJet_Pt-40toInf' in filen:
                # treename=['GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8']
                treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                process_ID = 'GJet'
            elif 'DYJetsToLL_M-50_TuneCP5' in filen:
                treename=['DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8',
                ]
                process_ID = 'DY'
            elif 'TTGG' in filen:
                treename=['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8',
                ]
                process_ID = 'TTGG'
            elif 'TTGJets' in filen:
                treename=['tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_2']
                process_ID = 'TTGJets'
            elif 'TTJets_HT-600to800' in filen:
                treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-800to1200' in filen:
                treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-1200to2500' in filen:
                treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-2500toInf' in filen:
                treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'W1JetsToLNu_LHEWpT_0-50' in filen:
                treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_50-150' in filen:
                treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_150-250' in filen:
                treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_250-400' in filen:
                treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_400-inf' in filen:
                treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_0-50' in filen:
                treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_50-150' in filen:
                treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_150-250' in filen:
                treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_250-400' in filen:
                treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_400-inf' in filen:
                treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W3JetsToLNu' in filen:
                treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W4JetsToLNu' in filen:
                treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'WJets'
            elif 'ttHJetToGG' in filen:
                treename=['ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8',
                ]
                process_ID = 'ttH'

            fileName = os.path.join(subdir_name,filen)
            filename_fullpath = inputPath+"/"+fileName+".root"
            print("Input file: ", filename_fullpath)
            print("treename: ", treename[0])

            tree = uproot.open(filename_fullpath)[treename[0]]
            chunk_df_all = tree.arrays(cut=criteria,library='pd')
            chunk_df     = tree.arrays(filter_name=my_cols_list,cut=criteria,library='pd')
            # print(chunk_df)

            # my_cols_list=set(my_cols_list)
            # for arrs in tree.iterate(my_cols_list,library='pd'):
                # print(arrs)

            chunk_df['weight'] = chunk_df_all['weight']
            chunk_df['unweighted'] = 1.0
            chunk_df['target'] = target
            chunk_df['key'] = key
            chunk_df['classweight']=1.0
            chunk_df['process_ID']=process_ID
            # print("Shape (Before duplicate removal):")
            # print("\tchunk_df: ",chunk_df.shape)
            # print("\tdata: ",data.shape)
            # Duplicate removal
            chunk_df = chunk_df.loc[:,~chunk_df.columns.duplicated()]
            data     = data.loc[:,~data.columns.duplicated()]
            # print("Shape (After duplicate removal):")
            # print("\tchunk_df: ",chunk_df.shape)
            # print("\tdata: ",data.shape)
            # chunk_df.reset_index()
            # data.reset_index()
            # print("Shape (After reset index):")
            # print("\tchunk_df: ",chunk_df.shape)
            # print("\tdata: ",data.shape)
            data = data.append(chunk_df, ignore_index=True)
            # print(data)
            if len(data) == 0 : continue
            # print("Shape (before return)")
            # print("\tchunk_df: ",chunk_df.shape)
            # print("\tdata: ",data.shape)

    return data

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model

def baseline_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(100,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(80,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(40,activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def gscv_model(learn_rate=0.001):
    model = Sequential()
    model.add(Dense(32,input_dim=29,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

def main():
    print('Using Keras version: ', keras.__version__)
    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=1, type=int)
    parser.add_argument('-l', '--load_dataset', dest='load_dataset', help='Option to load dataset from root file (0=False, 1=True)', default=0, type=int)
    parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
    parser.add_argument('-p', '--para', dest='hyp_param_scan', help='Option to run hyper-parameter scan', default=0, type=int)
    parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing directories \'Bkgs\' and \'Signal\' which contain background and signal ntuples respectively.', default='', type=str)
    args = parser.parse_args()
    do_model_fit = args.train_model
    suffix = args.suffix

    # Create instance of the input files directory
    inputs_file_path = '/Users/ramkrishna/cernbox/post_doc_ihep/Machine-Learning/HHWWyy/MVANtuples/'

    hyp_param_scan=args.hyp_param_scan
    # Set model hyper-parameters
    weights='BalanceYields'# 'BalanceYields' or 'BalanceNonWeighted'
    optimizer = 'Nadam'
    validation_split=0.1
    # hyper-parameter scan results
    if weights == 'BalanceNonWeighted':
        learn_rate = 0.0005
        epochs = 200
        batch_size=200
    if weights == 'BalanceYields':
        learn_rate = 0.0001
        epochs = 200
        batch_size=400

    # Create instance of output directory where all results are saved.
    output_directory = 'HHWWyyDNN_binary_%s_%s/' % (suffix,weights)
    check_dir(output_directory)
    hyperparam_file = os.path.join(output_directory,'additional_model_hyper_params.txt')
    additional_hyperparams = open(hyperparam_file,'w')
    additional_hyperparams.write("optimizer: "+optimizer+"\n")
    additional_hyperparams.write("learn_rate: "+str(learn_rate)+"\n")
    additional_hyperparams.write("epochs: "+str(epochs)+"\n")
    additional_hyperparams.write("validation_split: "+str(validation_split)+"\n")
    additional_hyperparams.write("weights: "+weights+"\n")
    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')
    input_var_jsonFile = open('input_variables.json','r')
    selection_criteria = '( ((Leading_Photon_pt/CMS_hgg_mass) > 0.35) & ((Subleading_Photon_pt/CMS_hgg_mass) > 0.25) & (passbVeto==1) & (ExOneLep==0) & (N_goodJets>=4))'
    #selection_criteria = '( ((Leading_Photon_pt/CMS_hgg_mass) > 0.35) && ((Subleading_Photon_pt/CMS_hgg_mass) > 0.25) && passbVeto==1 && ExOneLep==1 && N_goodJets>=1)'

    # Load Variables from .json
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Create list of headers for dataset .csv
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('weight')
    column_headers.append('unweighted')
    column_headers.append('target')
    column_headers.append('key')
    column_headers.append('classweight')
    column_headers.append('process_ID')

    # Load ttree into .csv including all variables listed in column_headers
    print('<train-DNN> Input file path: ', inputs_file_path)
    outputdataframe_name = '%s/output_dataframe.csv' %(output_directory)
    print('outputdataframe_name: ',outputdataframe_name)
    if os.path.isfile(outputdataframe_name) and (args.load_dataset == 0):
        """Load dataset or not

        If one changes the input training variables then we have to reload dataset.
        Don't use the previous .csv file if you update the list of input variables.
        """
        data = pandas.read_csv(outputdataframe_name)
        print('<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name))
    else:
        print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
        data = load_data(inputs_file_path,column_headers,selection_criteria)
        # Change sentinal value to speed up training.
        data = data.replace(to_replace=-999.000000,value=-9.0)
        data.to_csv(outputdataframe_name, index=False)
        data = pandas.read_csv(outputdataframe_name)

    print('<main> data columns: ', (data.columns.values.tolist()))
    n = len(data)
    nHH = len(data.iloc[data.target.values == 1])
    nbckg = len(data.iloc[data.target.values == 0])
    print("Total (train+validation) length of HH = %i, bckg = %i" % (nHH, nbckg))

    # Make instance of plotter tool
    Plotter = plotter()
    # Create statistically independant training/testing data
    traindataset, valdataset = train_test_split(data, test_size=0.1)
    valdataset.to_csv((output_directory+'valid_dataset.csv'), index=False)

    print('<train-DNN> Training dataset shape: ', traindataset.shape)
    print('<train-DNN> Validation dataset shape: ', valdataset.shape)

    # Event weights
    weights_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'weight']
    weights_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'weight']
    weights_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'weight']
    weights_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'weight']
    weights_for_TTGG = traindataset.loc[traindataset['process_ID']=='TTGG', 'weight']
    weights_for_TTGJets = traindataset.loc[traindataset['process_ID']=='TTGJets', 'weight']
    weights_for_TTJets = traindataset.loc[traindataset['process_ID']=='TTJets', 'weight']
    weights_for_WJets = traindataset.loc[traindataset['process_ID']=='WJets', 'weight']
    weights_for_ttH = traindataset.loc[traindataset['process_ID']=='ttH', 'weight']

    HHsum_weighted= sum(weights_for_HH)
    GJetsum_weighted= sum(weights_for_GJet)
    DiPhotonsum_weighted= sum(weights_for_DiPhoton)
    TTGGsum_weighted= sum(weights_for_TTGG)
    TTGJetssum_weighted= sum(weights_for_TTGJets)
    TTJetssum_weighted= sum(weights_for_TTJets)
    WJetssum_weighted= sum(weights_for_WJets)
    ttHsum_weighted= sum(weights_for_ttH)
    DYsum_weighted= sum(weights_for_DY)
    #bckgsum_weighted = DiPhotonsum_weighted+WJetssum_weighted+ttHsum_weighted
    bckgsum_weighted = GJetsum_weighted + DiPhotonsum_weighted+TTGGsum_weighted+TTGJetssum_weighted+TTJetssum_weighted+WJetssum_weighted+ttHsum_weighted+DYsum_weighted

    nevents_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'unweighted']
    nevents_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'unweighted']
    nevents_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'unweighted']
    nevents_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'unweighted']
    nevents_for_TTGG = traindataset.loc[traindataset['process_ID']=='TTGG', 'unweighted']
    nevents_for_TTGJets = traindataset.loc[traindataset['process_ID']=='TTGJets', 'unweighted']
    nevents_for_TTJets = traindataset.loc[traindataset['process_ID']=='TTJets', 'unweighted']
    nevents_for_WJets = traindataset.loc[traindataset['process_ID']=='WJets', 'unweighted']
    nevents_for_ttH = traindataset.loc[traindataset['process_ID']=='ttH', 'unweighted']

    HHsum_unweighted= sum(nevents_for_HH)
    GJetsum_unweighted= sum(nevents_for_GJet)
    DiPhotonsum_unweighted= sum(nevents_for_DiPhoton)
    TTGGsum_unweighted= sum(nevents_for_TTGG)
    TTGJetssum_unweighted= sum(nevents_for_TTGJets)
    TTJetssum_unweighted= sum(nevents_for_TTJets)
    WJetssum_unweighted= sum(nevents_for_WJets)
    ttHsum_unweighted= sum(nevents_for_ttH)
    DYsum_unweighted= sum(nevents_for_DY)

    #bckgsum_unweighted = DiPhotonsum_unweighted+WJetssum_unweighted+ttHsum_unweighted
    bckgsum_unweighted = GJetsum_unweighted + DiPhotonsum_unweighted + TTGGsum_unweighted + TTGJetssum_unweighted + TTJetssum_unweighted + WJetssum_unweighted + ttHsum_unweighted + DYsum_unweighted

    if weights=='BalanceYields':
        print('HHsum_weighted= ' , HHsum_weighted)
        print('ttHsum_weighted= ' , ttHsum_weighted)
        print('DiPhotonsum_weighted= ', DiPhotonsum_weighted)
        print('WJetssum_weighted= ', WJetssum_weighted)
        print('DYsum_weighted= ', DYsum_weighted)
        print('GJetsum_weighted= ', GJetsum_weighted)
        print('bckgsum_weighted= ', bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = 1.
        traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='WJets', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='TTGG', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='TTGJets', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='TTJets', ['classweight']] = (HHsum_weighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='ttH', ['classweight']] = (HHsum_weighted/bckgsum_weighted)

    if weights=='BalanceNonWeighted':
        print('HHsum_unweighted= ' , HHsum_unweighted)
        print('ttHsum_unweighted= ' , ttHsum_unweighted)
        print('DiPhotonsum_unweighted= ', DiPhotonsum_unweighted)
        print('WJetssum_unweighted= ', WJetssum_unweighted)
        print('DYsum_unweighted= ', DYsum_unweighted)
        print('GJetsum_unweighted= ', GJetsum_unweighted)
        print('bckgsum_unweighted= ', bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = 1.
        traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='WJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='TTGG', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='TTGJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='TTJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='ttH', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)

    # Remove column headers that aren't input variables
    training_columns = column_headers[:-6]
    print('<train-DNN> Training features: ', training_columns)

    column_order_txt = '%s/column_order.txt' %(output_directory)
    column_order_file = open(column_order_txt, "wb")
    for tc_i in training_columns:
        line = tc_i+"\n"
        pickle.dump(str(line), column_order_file)

    num_variables = len(training_columns)

    # Extract training and testing data
    X_train = traindataset[training_columns].values
    X_test = valdataset[training_columns].values

    # Extract labels data
    Y_train = traindataset['target'].values
    Y_test = valdataset['target'].values

    # Create dataframe containing input features only (for correlation matrix)
    train_df = data.iloc[:traindataset.shape[0]]

    # Event weights if wanted
    train_weights = traindataset['weight'].values
    test_weights = valdataset['weight'].values

    # Weights applied during training.
    if weights=='BalanceYields':
        trainingweights = traindataset.loc[:,'classweight']*traindataset.loc[:,'weight']
    if weights=='BalanceNonWeighted':
        trainingweights = traindataset.loc[:,'classweight']
    trainingweights = np.array(trainingweights)

    ## Input Variable Correlation plot
    correlation_plot_file_name = 'correlation_plot.png'
    Plotter.correlation_matrix(train_df)
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name)

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)

    if do_model_fit == 1:
        print('<train-BinaryDNN> Training new model . . . . ')
        histories = []
        labels = []

        if hyp_param_scan == 1:
            print('Begin at local time: ', time.localtime())
            hyp_param_scan_name = 'hyp_param_scan_results.txt'
            hyp_param_scan_results = open(hyp_param_scan_name,'a')
            time_str = str(time.localtime())+'\n'
            hyp_param_scan_results.write(time_str)
            hyp_param_scan_results.write(weights)
            learn_rates=[0.00001, 0.0001]
            epochs = [150,200]
            batch_size = [400,500]
            param_grid = dict(learn_rate=learn_rates,epochs=epochs,batch_size=batch_size)
            model = KerasClassifier(build_fn=gscv_model,verbose=0)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
            grid_result = grid.fit(X_train,Y_train,shuffle=True,sample_weight=trainingweights)
            print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
            hyp_param_scan_results.write("Best score: %f , best params: %s\n" %(grid_result.best_score_,grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("Mean (stdev) test score: %f (%f) with parameters: %r" % (mean,stdev,param))
                hyp_param_scan_results.write("Mean (stdev) test score: %f (%f) with parameters: %r\n" % (mean,stdev,param))
            exit()
        else:
            # Define model for analysis
            early_stopping_monitor = EarlyStopping(patience=30, monitor='val_loss', verbose=1)
            model = baseline_model(num_variables, learn_rate=learn_rate)

            # Fit the model
            # Batch size = examples before updating weights (larger = faster training)
            # Epoch = One pass over data (useful for periodic logging and evaluation)
            #class_weights = np.array(class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train))
            history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor])
            histories.append(history)
            labels.append(optimizer)
            # Make plot of loss function evolution
            Plotter.plot_training_progress_acc(histories, labels)
            acc_progress_filename = 'DNN_acc_wrt_epoch.png'
            Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
    else:
        model_name = os.path.join(output_directory,'model.h5')
        model = load_trained_model(model_name)

    # Node probabilities for training sample events
    result_probs = model.predict(np.array(X_train))
    result_classes = model.predict_classes(np.array(X_train))

    # Node probabilities for testing sample events
    result_probs_test = model.predict(np.array(X_test))
    result_classes_test = model.predict_classes(np.array(X_test))

    # Store model in file
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weights(weights_output_name)
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')
    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)
    model.summary()
    model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    '''
    print('================')
    print('Training event labels: ', len(Y_train))
    print('Training event probs', len(result_probs))
    print('Training event weights: ', len(train_weights))
    print('Testing events: ', len(Y_test))
    print('Testing event probs', len(result_probs_test))
    print('Testing event weights: ', len(test_weights))
    print('================')
    '''

    # Make overfitting plots of output nodes
    Plotter.binary_overfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)
    e = shap.DeepExplainer(model, X_train[:400, ])
    shap_values = e.shap_values(X_test[:400, ])
    Plotter.plot_dot(title="DeepExplainer_sigmoid_y0", x=X_test[:400, ], shap_values=shap_values, column_headers=column_headers)
    Plotter.plot_dot_bar(title="DeepExplainer_Bar_sigmoid_y0", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)
    #e = shap.GradientExplainer(model, X_train[:100, ])
    #shap_values = e.shap_values(X_test[:100, ])
    #Plotter.plot_dot(title="GradientExplainer_sigmoid_y0", x=X_test[:100, ], shap_values=shap_values, column_headers=column_headers)
    #e = shap.KernelExplainer(model.predict, X_train[:100, ])
    #shap_values = e.shap_values(X_test[:100, ])
    #Plotter.plot_dot(title="KernelExplainer_sigmoid_y0", x=X_test[:100, ],shap_values=shap_values, column_headers=column_headers)
    #Plotter.plot_dot_bar(title="KernelExplainer_Bar_sigmoid_y0", x=X_test[:100,], shap_values=shap_values, column_headers=column_headers)
    #Plotter.plot_dot_bar_all(title="KernelExplainer_bar_All_Var_sigmoid_y0", x=X_test[:100,], shap_values=shap_values, column_headers=column_headers)

    # Create confusion matrices for training and testing performance
    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'index')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'index')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_confusion_matrix_TEST.png')

    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'columns')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_columns_confusion_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'columns')
    # Plotter.save_plots(dir=plots_dir, filename='yields_norm_columns_confusion_matrix_TEST.png')

    # Plotter.conf_matrix(original_encoded_train_Y,result_classes_train,train_weights,'')
    # Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TRAIN.png')
    # Plotter.conf_matrix(original_encoded_test_Y,result_classes_test,test_weights,'')
    # Plotter.save_plots(dir=plots_dir, filename='yields_matrix_TEST.png')

    Plotter.ROC_sklearn(Y_train, result_probs, Y_test, result_probs_test, 1 , 'BinaryClassifierROC',train_weights, test_weights)

main()
