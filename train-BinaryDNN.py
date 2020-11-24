# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for HH->WWyy analysis.
# import shap
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
from array import array
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
from ROOT import TTree
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
from root_numpy import root2array, tree2array

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
    data = pd.DataFrame(columns=my_cols_list)
    keys=['HH','bckg']
    #keys=['HH']
    data = pd.DataFrame(columns=my_cols_list)
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
            fileNames = ['GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8']#,'TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_Hadded']#,'TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_Hadded']#,'TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_Hadded','TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_Hadded']
            #fileNames = ['DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa_Hadded']
            target=0

        for filen in fileNames:
            if 'ggF_SM_WWgg_qqqq_Hadded' in filen:
                #treename=['tagsDumper/trees/ggF_SM_WWgg_qqlnugg_13TeV_HHWWggTag_2','tagsDumper/trees/ggF_SM_WWgg_qqlnugg_13TeV_HHWWggTag_1']
                #treename=['tagsDumper/trees/ggF_SM_WWgg_qqlnugg_13TeV_HHWWggTag_1']
                treename=['tagsDumper/trees/GluGluToHHTo_WWgg_qqqq_nodeSM_13TeV_HHWWggTag_2']
                process_ID = 'HH'
            elif 'DiPhotonJetsBox_MGG' in filen:
                treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_0',
                'DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_1',
                'DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_2'
                #'DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_3',
                #'DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_4'
                ]
                process_ID = 'DiPhoton'
            elif 'GJet_Pt-40toInf' in filen:
                treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                # treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0','tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_1']
                #treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_1']
                process_ID = 'GJet'
            elif 'DYJetsToLL_M-50_TuneCP5' in filen:
                treename=['tagsDumper/trees/DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_2']
                #treename=['tagsDumper/trees/DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_1']
                process_ID = 'DY'
            elif 'TTGG' in filen:
                treename=['tagsDumper/trees/TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_2']
                #,'tagsDumper/trees/TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_1']
                process_ID = 'TTGG'
            elif 'TTGJets' in filen:
                treename=['tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_2']
                #,'tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_1']
                process_ID = 'TTGJets'
            #elif 'TTJets_HT-600to800' in filen:
            #    treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'TTJets'
            #elif 'TTJets_HT-800to1200' in filen:
            #    treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'TTJets'
            #elif 'TTJets_HT-1200to2500' in filen:
            #    treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'TTJets'
            #elif 'TTJets_HT-2500toInf' in filen:
            #    treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'TTJets'
            #elif 'W1JetsToLNu_LHEWpT_0-50' in filen:
            #    treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W1JetsToLNu_LHEWpT_50-150' in filen:
            #    treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W1JetsToLNu_LHEWpT_150-250' in filen:
            #    treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W1JetsToLNu_LHEWpT_250-400' in filen:
            #    treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W1JetsToLNu_LHEWpT_400-inf' in filen:
            #    treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W2JetsToLNu_LHEWpT_0-50' in filen:
            #    treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W2JetsToLNu_LHEWpT_50-150' in filen:
            #    treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W2JetsToLNu_LHEWpT_150-250' in filen:
            #    treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W2JetsToLNu_LHEWpT_250-400' in filen:
            #    treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W2JetsToLNu_LHEWpT_400-inf' in filen:
            #    treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
            #    'W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_1',
            #    'W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_2'
            #    #'W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_3',
            #    #'W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W3JetsToLNu' in filen:
            #    treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'W4JetsToLNu' in filen:
            #    treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
            #    'W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_1',
            #    'W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_2'
            #    #'W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_3',
            #    #'W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'WJets'
            #elif 'ttHJetToGG' in filen:
            #    treename=['ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_0',
            #    'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_1',
            #    'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_2'
            #    #'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_3',
            #    #'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_4'
            #    ]
            #    process_ID = 'ttH'

            fileName = os.path.join(subdir_name,filen)
            filename_fullpath = inputPath+"/"+fileName+".root"
            print("Input file: ", filename_fullpath)
            tfile = ROOT.TFile(filename_fullpath)
            for tname in treename:
                ch_0 = tfile.Get(tname)
                if ch_0 is not None :
                    # Create dataframe for ttree
                    #chunk_arr = tree2array(tree=ch_0, selection=criteria)  # Check this
                    chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria)
                    #chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria, start=0, stop=500)
                    # This dataframe will be a chunk of the final total dataframe used in training
                    chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
                    # Add values for the process defined columns.
                    # (i.e. the values that do not change for a given process).
                    chunk_df['key']=key
                    chunk_df['target']=target
                    chunk_df['weight']=chunk_df["weight"]
                    chunk_df['process_ID']=process_ID
                    chunk_df['classweight']=1.0
                    chunk_df['unweighted'] = 1.0
                    # Append this chunk to the 'total' dataframe
                    data = data.append(chunk_df, ignore_index=True)
                else:
                    print("TTree == None")
                ch_0.Delete()
            tfile.Close()
        if len(data) == 0 : continue

    return data

def load_trained_model(weights_path, num_variables, optimizer,nClasses):
    model = baseline_model(num_variables, optimizer,nClasses)
    model.load_weights(weights_path)
    return model

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
def binary_focal_loss_fixed(y_true, y_pred):
    """
    :param y_true: A tensor of the same shape as `y_pred`
    :param y_pred:  A tensor resulting from a sigmoid
    :return: Output tensor.
    """
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
           -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def normalise(x_train, x_test):
    mu = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_normalised = (x_train - mu) / std
    x_test_normalised = (x_test - mu) / std
    return x_train_normalised, x_test_normalised

def baseline_model(num_variables,optimizer='Nadam',learn_rate=0.001,nClasses=1):
    model = Sequential()
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
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
    parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
    args = parser.parse_args()
    do_model_fit = args.train_model
    suffix = args.suffix
    # Create instance of output directory where all results are saved.
    output_directory = 'HHWWyyDNN_binary_%s/' % (suffix)
    check_dir(output_directory)
    # Set model hyper-parameters
    optimizer = 'Nadam'
    learn_rate = 0.001
    epochs = 150
    validation_split=0.1
    batch_size=100
    weights='BalanceNonWeighted'#'BalanceYields'
    hyperparam_file = os.path.join(output_directory,'additional_model_hyper_params.txt')
    additional_hyperparams = open(hyperparam_file,'w')
    additional_hyperparams.write("optimizer: "+optimizer+"\n")
    additional_hyperparams.write("learn_rate: "+str(learn_rate)+"\n")
    additional_hyperparams.write("epochs: "+str(epochs)+"\n")
    additional_hyperparams.write("validation_split: "+str(validation_split)+"\n")
    additional_hyperparams.write("weights: "+weights+"\n")
    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')

    #input_var_jsonFile = open('input_variables_new.json','r')
    input_var_jsonFile = open('input_variables.json','r')
    selection_criteria = '(AtLeast4GoodJets0Lep==1)'
    # selection_criteria = '(passPhotonSels==1 && passbVeto==1 && ExOneLep==1 && goodJets==1)'
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

    # Create instance of the input files directory
    #inputs_file_path = '/afs/cern.ch/work/a/atishelm/public/ForJosh/2017_DataMC_ntuples_moreVars'
    inputs_file_path = '/eos/user/r/rasharma/post_doc_ihep/double-higgs/ntuples/September29/MVANtuples'
    #inputs_file_path = '/eos/user/a/atishelm/ntuples/HHWWgg_DataSignalMCnTuples/PromptPromptApplied/'
    #inputs_file_path = 'PromptPromptApplied/'

    # Load ttree into .csv including all variables listed in column_headers
    print('<train-DNN> Input file path: ', inputs_file_path)
    outputdataframe_name = '%s/output_dataframe.csv' %(output_directory)
    if os.path.isfile(outputdataframe_name):
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
    bckgsum_weighted = DiPhotonsum_weighted+WJetssum_weighted
    print('HHsum_weighted= ' , HHsum_weighted)
    print('bckgsum_weighted= ', bckgsum_weighted)

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
    bckgsum_unweighted = DiPhotonsum_unweighted+WJetssum_unweighted
    print('HHsum_unweighted= ' , HHsum_unweighted)
    print('DiPhotonsum_unweighted= ', DiPhotonsum_unweighted)
    print('WJetssum_unweighted= ', WJetssum_unweighted)
    print('DYsum_unweighted= ', DYsum_unweighted)
    print('GJetsum_unweighted= ', GJetsum_unweighted)
    print('bckgsum_unweighted= ', bckgsum_unweighted)

    if weights=='BalanceYields':
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

    ## Input Variable Correlation plot
    correlation_plot_file_name = 'correlation_plot.png'
    Plotter.correlation_matrix(train_df)
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name)

    ####################################################################################
    # Weights applied during training. You will also need to update the class weights if
    # you are going to change the event weights applied. Introduce class weights and any
    # event weight you want to use here.
    #trainingweights = traindataset.loc[:,'classbalance']#*traindataset.loc[:,'weight']
    #trainingweights = np.array(trainingweights)

    # Temp hack to be able to change class weights without remaking dataframe
    #for inde in xrange(len(trainingweights)):
    #    newweight = 13243.0/6306.0
    #    trainingweights[inde]= newweight
    #print 'training event weight = ', trainingweights[0]

    # Event weights calculation so we can correctly apply event weights to diagnostic plots.
    # use seperate list because we don't want to apply class weights in plots.
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
    # correlation_plot_file_name = 'correlation_plot.pdf'
    # Plotter.correlation_matrix(train_df)
    # Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name)

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
        # Define model and early stopping
        early_stopping_monitor = EarlyStopping(patience=30, monitor='val_loss', verbose=1)
        model = baseline_model(num_variables, optimizer, learn_rate=learn_rate)

        '''epochs = [50,100,200]
        batchsize = [500,1000,2000,3000]
        # init = ['glorot_normal','uniform','normal','glorot_uniform']
        learning_rates = [0.001,0.005,0.01]
        # lossfns=['binary_crossentropy']
        optimizers = ['Adamax','Adam','Nadam','Adadelta','Adagrad']
        # Parameter grid to scan on. Can take comma seperated list of parameters.
        param_grid = dict(learn_rate=learning_rates,batch_size=batchsize,epochs=epochs)
        #param_grid = dict(optimizer=optimizers)
        # Wrap keras model using KerasClassifier so it can be used in sklearn
        # 'baseline_model' builds and returns keras sequential model that's passed to build_fn to construct KerasClassifier class.
        model = KerasClassifier(build_fn=baseline_model,num_variables=num_variables,optimizer='Nadam')
        #model = KerasClassifier(build_fn=baseline_model,num_variables=num_variables,epochs=100,batch_size=1000)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_train,Y_train,validation_split=0.2,shuffle=True,sample_weight=trainingweights)
        print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("Mean (stdev) test score: %f (%f) with parameters: %r" % (mean,stdev,param))
        exit()'''

        # Fit the model
        # Batch size = examples before updating weights (larger = faster training)
        # Epoch = One pass over data (useful for periodic logging and evaluation)
        #class_weights = np.array(class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train))
        history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor])
        histories.append(history)
        labels.append(optimizer)

        print "DEBUG: labels: ",labels

        # Make plot of loss function evolution
        Plotter.plot_training_progress_acc(histories, labels)
        acc_progress_filename = 'DNN_acc_wrt_epoch.png'
        Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
    else:
        # Which model do you want to load?
        model_name = os.path.join(output_directory,'model.h5')
        print('<train-DNN> Loaded Model: %s' % (model_name))
        model = load_trained_model(model_name,num_variables,optimizer,1)

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
    model_schematic_name = os.path.join(output_directory,'model_schematic.eps')
    print "DEBUG: ",model_schematic_name
    plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)
    # plot_model(model, to_file='model_schematic.eps', show_shapes=True, show_layer_names=True)

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    # Make overfitting plots of output nodes
    Plotter.binary_overfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)
    print "DEBUG: Y_train shape: ",Y_train.shape

    # # Get true process integers for training dataset
    # original_encoded_train_Y = []
    # for i in xrange(len(result_probs)):
    #     if Y_train[i][0] == 1:
    #         original_encoded_train_Y.append(0)
    #     if Y_train[i][1] == 1:
    #         original_encoded_train_Y.append(1)
    #     if Y_train[i][2] == 1:
    #         original_encoded_train_Y.append(2)
    #     if Y_train[i][3] == 1:
    #         original_encoded_train_Y.append(3)

    # Get true class values for testing dataset
    # result_classes_test = newencoder.inverse_transform(result_classes_test)
    # result_classes_train = newencoder.inverse_transform(result_classes)
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

    Plotter.ROC_sklearn(Y_train, result_probs, Y_test, result_probs_test, 1 , 'BinaryClassifierROC')

main()
