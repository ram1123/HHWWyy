import matplotlib.pyplot as plt
import numpy as np
import numpy
import pandas
import pandas as pd
import optparse, json, argparse
import ROOT
import sys
from array import array
from ROOT import TFile, TTree, gDirectory
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, LabelEncoder
from sklearn.utils import class_weight
import os
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['EOS_MGM_URL'] = 'root://eosuser.cern.ch/'
import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam, Adadelta, Adagrad
from keras.optimizers import Nadam
from keras.layers import Dropout, Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from root_numpy import root2array, tree2array

# python looks here for its packages: $PYTHONPATH. Need to add path to $PYTHONPATH so python can find the required packages.
sys.path.insert(0, '/Users/joshuhathomas-wilsker/Documents/work/lxplus_remote/work/private/IHEP/HH/HHWWyy/')
from plotting.plotter import plotter

class apply_DNN(object):

    def __init__(self):
        pass

    def getEOSlslist(self, directory, mask='', prepend='root://eosuser.cern.ch'):
        eos_dir = '/eos/user/%s ' % (directory)
        eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
        #out = commands.getoutput(eos_cmd)
        out = subprocess.run(['ls', '-l'], capture_output=True, text=True).stdout
        print('out: ', out)
        full_list = []
        ## if input file was single root file:
        if directory.endswith('.root'):
            if len(out.split('\n')[0]) > 0:
                return [os.path.join(prepend,eos_dir).replace(" ","")]
        ## instead of only the file name append the string to open the file in ROOT
        for line in out.split('\n'):
            print('line: ', line)
            if len(line.split()) == 0: continue
            full_list.append(os.path.join(prepend,eos_dir,line).replace(" ",""))
        ## strip the list of files if required
        if mask != '':
            stripped_list = [x for x in full_list if mask in x]
            return stripped_list
        print('full files list from eos: ', full_list)
        ## return
        return full_list

    def load_data(self, inputPath, variables, criteria, process):
        my_cols_list=variables
        data = pd.DataFrame(columns=my_cols_list)
        if 'HHWWgg-SL-SM-NLO-2017' in process:
            sampleNames=process
            fileNames = [process]
            target=1
        else:
            sampleNames=process
            fileNames = [process]
            target=0

        print('<apply_DNN> Process: ' , process)
        if 'HHWWgg-SL-SM-NLO-2017' in process:
            treename=[
            'GluGluToHHTo2G2Qlnu_node_cHHH1_TuneCP5_PSWeights_13TeV_powheg_pythia8alesauva_2017_1_10_6_4_v0_RunIIFall17MiniAODv2_PU2017_12Apr2018_94X_mc2017_realistic_v14_v1_1c4bfc6d0b8215cc31448570160b99fdUSER',
            ]
        elif 'DiPhotonJetsBox_MGG' in process:
            treename=[
            'DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa',
            ]
        elif 'GJet_Pt-20toInf' in process:
            treename = [
            'GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8',
            ]
        elif 'GJet_Pt-20to40' in process:
            treename = [
            'GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8',
            ]
        elif 'GJet_Pt-40toInf' in process:
            treename=[
            'GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8',
            ]
        elif 'DYJetsToLL_M-50_TuneCP5' in process:
            treename=[
            'DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8',
            ]
        elif 'TTGG' in process:
            treename=[
            'TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8',
            ]
        elif 'TTGJets' in process:
            treename=[
            'TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8',
            ]
        elif 'TTJets_HT-600to800' in process:
            treename=[
            'TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'TTJets_HT-800to1200' in process:
            treename=[
            'TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'TTJets_HT-1200to2500' in process:
            treename=[
            'TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'TTJets_HT-2500toInf' in process:
            treename=[
            'TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'W1JetsToLNu_LHEWpT_0-50' in process:
            treename=[
            'W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W1JetsToLNu_LHEWpT_50-150' in process:
            treename=[
            'W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W1JetsToLNu_LHEWpT_150-250' in process:
            treename=[
            'W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W1JetsToLNu_LHEWpT_250-400' in process:
            treename=[
            'W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W1JetsToLNu_LHEWpT_400-inf' in process:
            treename=[
            'W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W2JetsToLNu_LHEWpT_0-50' in process:
            treename=[
            'W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W2JetsToLNu_LHEWpT_50-150' in process:
            treename=[
            'W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W2JetsToLNu_LHEWpT_150-250' in process:
            treename=[
            'W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W2JetsToLNu_LHEWpT_250-400' in process:
            treename=[
            'W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W2JetsToLNu_LHEWpT_400-inf' in process:
            treename=[
            'W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
            ]
        elif 'W3JetsToLNu' in process:
            treename=[
            'W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'W4JetsToLNu' in process:
            treename=[
            'W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
            ]
        elif 'ttHJetToGG' in process:
            treename=[
            'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8',
            ]

        filename_fullpath = inputPath+"/"+process+".root"
        print("<apply_DNN> Input file: ", filename_fullpath)
        tfile = ROOT.TFile(filename_fullpath)
        for tname in treename:
            print('<apply_DNN> TTree: ', tname)
            ch_0 = tfile.Get(tname)
            if ch_0 is not None :
                #chunk_arr = tree2array(tree=ch_0, selection=criteria)
                chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-1], selection=criteria)
                chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
                chunk_df['key']=process
                chunk_df['target']=target
                chunk_df['weight']=chunk_df["weight"]
                # =============== Weights ==================
                # WARNING! 'sample_weight' will overide 'class_weight'
                # ==========================================
                if sampleNames=='HH':
                    # Reweight classes
                    chunk_df['classbalance'] = 1.0
                if sampleNames=='bckg':
                    chunk_df['classbalance'] = 1.0
                data = data.append(chunk_df, ignore_index=True)
            else:
                print("<apply_DNN> TTree == None")
            ch_0.Delete()
        tfile.Close()
        return data

    def baseline_model(self,num_variables,learn_rate=0.001):
        model = Sequential()
        model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
        #model.add(Dense(16,activation='relu'))
        model.add(Dense(8,activation='relu'))
        model.add(Dense(4,activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
        return model

    def load_trained_model(self, weights_path, num_variables, learn_rate):
        print('<load_trained_model> weights_path: ', model_path)
        model = keras.models.load_model(model_path, compile=False)
        return model

    def check_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate_model(self, eventnum_resultsprob_dict, Eventnum_):
        output_node_value = -1.
        output_node_value = eventnum_resultsprob_dict.get(Eventnum_)[0]
        return output_node_value

    def event_categorised_max_value(self, event_classification, evaluated_node_values):

        event_node_values_max_only = [-2,-2,-2,-2]
        if event_classification == 0:
            event_node_values_max_only[0] = evaluated_node_values[0]
        elif event_classification == 1:
            event_node_values_max_only[1] = evaluated_node_values[1]
        elif event_classification == 2:
            event_node_values_max_only[2] = evaluated_node_values[2]
        elif event_classification == 3:
            event_node_values_max_only[3] = evaluated_node_values[3]

        return event_node_values_max_only
