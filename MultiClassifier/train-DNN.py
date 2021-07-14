# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for HH->WWyy analysis.

######################################################################################################################################################################
# Example Commands:
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc7-opt/setup.sh
####-- Newer software source: source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh
#
##-- Testing
# python train-DNN.py -t 1 -s DryRunCorrelation_lessSamples -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 3 --useKinWeight --VHToGGClassWeightFactor 2.0 --ttHJetToGGClassWeightFactor 2.0 --BkgClassWeightFactor 1.0 --LessSamples
# python train-DNN.py -t 1 -s DryRunCorrelation -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 3 --useKinWeight --VHToGGClassWeightFactor 2.0 --ttHJetToGGClassWeightFactor 2.0 --BkgClassWeightFactor 1.0
# python train-DNN.py -t 0 -s MultiClassDryRunForClassWeights -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 3 --useKinWeight --VHToGGClassWeightFactor 2.0 --ttHJetToGGClassWeightFactor 2.0 --BkgClassWeightFactor 1.0
# python train-DNN.py -t 1 -s MultiClass_EvenSingleH_2Hgg_withKinWeight_HggClassScale_1_BkgClassScale_1_HHNormed -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 200 --useKinWeight --HggClassWeightFactor 1.0 --BkgClassWeightFactor 1.0
# python train-DNN.py -t 1 -s MultiClass_EvenSingleH_2Hgg_withKinWeight_HggClassScale_4_BkgClassScale_1 -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 200 --useKinWeight --HggClassWeightFactor 4.0 --BkgClassWeightFactor 1.0
# python train-DNN.py -t 1 -s WithHggFactor2-3Epochs-3ClassMulticlass_EvenSingleH_2Hgg_withKinWeightCut10 -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 3 --useKinWeight

# ##-- Run Multiclassifier and save all outputs to website
# python train-DNN.py -t 1 -s 200Epochs-3ClassMulticlass_EvenSingleH_4Hgg_withKinWeightCut10 -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 3 --kinWeight
# python train-DNN.py -t 1 -s 5Epochs-3ClassMulticlass_EvenSingleH_Run2 -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 5
# python train-DNN.py -t 1 -s 200Epochs-3ClassMulticlass_EvenSingleH_Run2 -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput -e 200
#
# ##-- Run Binary
# python train-DNN.py -t 1 -s Binary-10Epochs -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ -e 10 --LessSamples
#
# ##-- Run Binary Classifier, output to website
# python train-DNN.py -t 1 -s Test-Train-DNN-Binary -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --FastCheck --Website /eos/user/a/atishelm/www/HHWWgg/DNN/
#
# ##-- Output Files Locally (leave --Website flag empty)
# python train-DNN.py -t 1 -s Test-Train-DNN-Binary -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --FastCheck
#
# ##-- Run multiclassifier and output to website
# python train-DNN.py -t 1 -s FastCheck-Multiclass -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --FastCheck --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass
#
# python train-DNN.py -t 1 -s 100Epochs-MultiClass-2Classes -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --FastCheck --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass --SaveOutput
######################################################################################################################################################################
# Next two files are to get rid of warning while traning on IHEP GPU
import os
import sys
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
# import shap
from array import array
import time
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
from sklearn.metrics import roc_curve
from os import environ
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.callbacks import EarlyStopping
from plotting.plotter import plotter
from numpy.testing import assert_allclose
from tensorflow.keras.callbacks import ModelCheckpoint
from root_numpy import root2array, tree2array
from tensorflow.keras.callbacks import CSVLogger
# from tensorflow.keras.utils.vis_utils import model_to_dot
from tensorflow.keras import utils

##-- External Modules
from python.trainDNNTools import *

seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    #out = commands.getoutput(eos_cmd)
    return

def load_data(channel, inputPath, variables, criteria, LessSamples, useKinWeight):
    # Load dataset to .csv format file
    my_cols_list=variables
    data = pd.DataFrame(columns=my_cols_list)
    # keys=['HH','H','bkg']
    keys=['HH','bbgg','bkg']
    # keys=['HH','DiPhoton','QCD']
    data = pd.DataFrame(columns=my_cols_list)
    for key in keys :
        print('key: ', key)
        if 'HH' in key:
            sampleNames=key
            subdir_name = 'Signal'
            fileNames = []

            fileNames_SL = [
            ##-- 2016
            # 'GluGluToHHTo2G2Qlnu_node_10_2016_LO_withNLO_HHWWggTag_0_MoreVars_kinWeight_noHgg',
            # 'GluGluToHHTo2G2Qlnu_node_11_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_1_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_12_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_2_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_3_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_4_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_5_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_6_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_7_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_8_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_9_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_SM_2016_LO_withNLOweights_HHWWggTag_0_MoreVars',

            ##-- 2017
            'GluGluToHHTo2G2Qlnu_node_10_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_11_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_1_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_12_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_2_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_3_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_4_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_5_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_6_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_7_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_8_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_9_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',
            'GluGluToHHTo2G2Qlnu_node_SM_2017_LO_withNLOweights_HHWWggTag_0_MoreVars',

            ##-- 2018
            # 'GluGluToHHTo2G2Qlnu_node_10_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_11_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_1_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_12_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_2_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_3_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_4_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_5_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_6_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_7_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_8_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_9_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',
            # 'GluGluToHHTo2G2Qlnu_node_SM_2018_LO_withNLOweights_HHWWggTag_0_MoreVars',

            #'GluGluToHHTo2G2Qlnu_node_cHHH1_2017_HHWWggTag_0_odd_MoreVars_kinWeight_noHgg',
            #'GluGluToHHTo2G2Qlnu_node_cHHH1_2017_HHWWggTag_0_even_MoreVars_kinWeight_noHgg'
            ]

            fileNames_FH = [
                # 'GluGluToHHTo2G4Q_node_cHHH1_2017'
                'GluGluToHHTo2G4Q_node_1_2017',
                'GluGluToHHTo2G4Q_node_2_2017'
                # 'GluGluToHHTo2G4Q_node_3_2017',
                # 'GluGluToHHTo2G4Q_node_4_2017',
                # 'GluGluToHHTo2G4Q_node_5_2017',
                # 'GluGluToHHTo2G4Q_node_6_2017',
                # 'GluGluToHHTo2G4Q_node_7_2017',
                # 'GluGluToHHTo2G4Q_node_8_2017',
                # 'GluGluToHHTo2G4Q_node_9_2017',
                # 'GluGluToHHTo2G4Q_node_10_2017',
                # 'GluGluToHHTo2G4Q_node_11_2017',
                # 'GluGluToHHTo2G4Q_node_12_2017',
                # 'GluGluToHHTo2G4Q_node_SM_2017',
            ]

            if(LessSamples):
                fileNames_SL = ['GluGluToHHTo2G2Qlnu_node_SM_2017_LO_withNLOweights_HHWWggTag_0_MoreVars']

            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 0

        elif key == 'H':
            sampleNames = key
            subdir_name = 'Backgrounds'
            fileNames = []
            fileNames_SL = [
                        ##-- 2016
                        # 'GluGluHToGG_M125_2016_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_M125_2016_HHWWggTag_0_MoreVars_even',
                        'VHToGG_M125_2016_HHWWggTag_0_MoreVars_even',
                        'ttHJetToGG_M125_2016_HHWWggTag_0_MoreVars_even',

                        ##-- 2017
                        # 'GluGluHToGG_2017_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_2017_HHWWggTag_0_MoreVars_even',
                        'VHToGG_2017_HHWWggTag_0_MoreVars_even',
                        'ttHJetToGG_2017_HHWWggTag_0_MoreVars_even',

                        ##-- 2018
                        # 'GluGluHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        'VHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        'ttHJetToGG_2018_M125_HHWWggTag_0_MoreVars_even',
                        ]
            fileNames_FH = [
                # 'ttHJetToGG_M125_13TeV',
                # 'VBFHToGG_M125_13TeV',
                # 'GluGluHToGG_M125_TuneCP5_13TeV',
                # 'VHToGG_M125_13TeV',
                'GluGluToHHTo2B2G_node_cHHH1_2017'
            ]
            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 1

        elif key == 'bbgg':
            sampleNames = key
            subdir_name = 'Backgrounds'
            fileNames = []
            fileNames_SL = [
                        ##-- 2016
                        # 'GluGluHToGG_M125_2016_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_M125_2016_HHWWggTag_0_MoreVars_even',
                        ##-- 2017
                        # 'GluGluHToGG_2017_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_2017_HHWWggTag_0_MoreVars_even',
                        'VHToGG_2017_HHWWggTag_0_MoreVars_even',
                        'ttHJetToGG_2017_HHWWggTag_0_MoreVars_even',

                        ##-- 2018
                        # 'GluGluHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        # 'VBFHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        'VHToGG_M125_2018_HHWWggTag_0_MoreVars_even',
                        'ttHJetToGG_2018_M125_HHWWggTag_0_MoreVars_even',
                        ]
            fileNames_FH = [
                # 'ttHJetToGG_M125_13TeV',
                # 'VBFHToGG_M125_13TeV',
                # 'GluGluHToGG_M125_TuneCP5_13TeV',
                # 'VHToGG_M125_13TeV',
                'GluGluToHHTo2B2G_node_cHHH1_2017'
            ]
            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 1

        elif key == 'DiPhoton':
            sampleNames = key
            subdir_name = 'Backgrounds'
            fileNames = []
            fileNames_SL = []
            fileNames_FH = [
                'DiPhotonJetsBox_MGG-80toInf_13TeV',
            ]
            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 1

        elif key == 'QCD':
            sampleNames = key
            subdir_name = 'Backgrounds'
            fileNames = []
            fileNames_SL = []
            fileNames_FH = [
                'datadrivenQCD_v2'
            ]
            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 2

        elif key == 'bkg':
            sampleNames = key
            subdir_name = 'Backgrounds'
            fileNames = []
            fileNames_SL = [
                #'DiPhotonJetsBox_M40_80_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'DiPhotonJetsBox_MGG-80toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'GJet_Pt-20to40_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'GJet_Pt-20toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'GJet_Pt-40toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'QCD_Pt-30to40_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'QCD_Pt-30toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'QCD_Pt-40toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'DYJetsToLL_M-50_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'TTGG_0Jets_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'TTGJets_TuneCP5_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'TTJets_HT-600to800_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'TTJets_HT-800to1200_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'TTJets_HT-1200to2500_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'TTJets_HT-2500toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'ttWJets_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'TTJets_TuneCP5_extra_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'W1JetsToLNu_LHEWpT_0-50_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'W1JetsToLNu_LHEWpT_50-150_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W1JetsToLNu_LHEWpT_150-250_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W1JetsToLNu_LHEWpT_250-400_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W1JetsToLNu_LHEWpT_400-inf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'W2JetsToLNu_LHEWpT_0-50_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'W2JetsToLNu_LHEWpT_50-150_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W2JetsToLNu_LHEWpT_150-250_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W2JetsToLNu_LHEWpT_250-400_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'W2JetsToLNu_LHEWpT_400-inf_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'W3JetsToLNu_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'W4JetsToLNu_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                'WGGJets_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                # 'WGJJToLNuGJJ_EWK_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                'WGJJToLNu_EWK_QCD_HHWWggTag_0_MoreVars_kinWeight_noHgg',

                #'WWTo1L1Nu2Q_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                #'WW_TuneCP5_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                # 'GluGluHToGG_2017_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                # 'VBFHToGG_2017_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                # 'VHToGG_2017_HHWWggTag_0_MoreVars_kinWeight_noHgg',
                # 'ttHJetToGG_2017_HHWWggTag_0_MoreVars_kinWeight_noHgg'
            ]


            if(LessSamples):
                fileNames = ['DiPhotonJetsBox_MGG-80toInf_HHWWggTag_0_MoreVars_kinWeight_noHgg']

            fileNames_FH = [
                'DiPhotonJetsBox_MGG-80toInf_13TeV',
                'TTGG_0Jets_TuneCP5_13TeV',
                'TTGJets_TuneCP5_13TeV',
                'datadrivenQCD_v2'
            ]
            if (channel == "SL"):
                fileNames = fileNames_SL
            if (channel == "FH"):
                fileNames = fileNames_FH
            target = 2
        for filen in fileNames:
            treename, process_ID = GetFileInfo(filen,channel)
            print("filen:",filen)

            ##-- if 2016 or 2018 HH or H, replace 2017 in hardcoded path with 2016 or 2018
            if(target == 1):
                inputPath_ = "/hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v5_BScoreBugFix/"
                if("2016" in filen):
                    subdir_name_ = "2016"
                elif("2017" in filen):
                    subdir_name_ = "2017"
                    # inputPath_ = "/eos/user/a/atishelm/ntuples/HHWWgg_DNN/EvenOddNtuples/"
                    # inputPath_ = "%s"%(inputPath)
                    # inputPath_ = inputPath_.replace("2017", "2016")
                    # subdir_name_ = "Single_H_hadded"
                    # inputPath_ = inputPath[:].replace("2017", "2016")
                    # subdir_name_ = "Single_H_hadded" subdir_name.replace("Backgrounds", "Single_H_hadded")
                elif("2018" in filen):
                    subdir_name_ = "2018"
                    # inputPath_ = "%s"%(inputPath)
                    # inputPath_ = inputPath_.replace("2017", "2018")
                    # inputPath_ = inputPath[:].replace("2017", "2018")
                    # subdir_name_ = "Single_H_2018_hadded"
                    # subdir_name_ = subdir_name.replace("Backgrounds", "Single_H_hadded")
                else:
                    subdir_name_ = "%s"%(subdir_name)
                    inputPath_ = "%s"%(inputPath)
            else:
                subdir_name_ = "%s"%(subdir_name)
                inputPath_ = "%s"%(inputPath)

            fileName = os.path.join(subdir_name_,filen)

            # filename_fullpath = inputPath+"/"+fileName+".root"
            filename_fullpath = "%s/%s.root"%(inputPath_, fileName)
            # filename_fullpath = inputPath_ + "/" + fileName + ".root"
            print("Input file: ", filename_fullpath)
            print("treename:",treename)
            tfile = ROOT.TFile(filename_fullpath)
            for tname in treename:
                print("tname:",tname)
                ch_0 = tfile.Get(tname)

                ##-- Remove events with large weight * kinWeight values
                if(useKinWeight):
                    print("APPLYING fiducial selection fabs(weight * kinWeight) < 10 to background events")
                    CutDict = {
                        0 : "(1.)", ##-- HH
                        1 : "(1.)", ##-- H
                        2 : '( ( fabs(weight * kinWeight) < 10 ) )' ##-- Continuum Background, apply kinWeight fiducial cut to remove very large weight*kinWeight events
                    }

                elif(not useKinWeight):
                    print("NOT APPLYING fiducial selection fabs(weight * kinWeight) < 10 to background events")
                    # CutDict = {
                    #     0 : "(1.)", ##-- HH
                    #     1 : "(1.)", ##-- H
                    #     2 : '(1.)', ##-- Diphoton
                    #     3 : '(1.)', ##-- QCD
                    #     4 : '(1.)' ##-- Continuum Background
                    # }

                    CutDict = {
                        0 : "(1.)", ##-- HH
                        1 : "(1.)", ##-- bbgg
                        2 : '(1.)' ##-- bkg
                    }

                if ch_0 is not None :
                    criteria_tmp = criteria
                    #if process_ID == "HH": criteria_tmp = criteria + " && (event%2!=0)"
                    # selection_criteria = '( ( fabs(weight * kinWeight) < 10 ) )'

                    # criteria_tmp = CutDict[target]
                    print("Sample Event Selection:",criteria_tmp)

                    # Create dataframe for ttree
                    print("my_cols_list:",my_cols_list)
                    print("my_cols_list[:-5]:",my_cols_list[:-5])
                    chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria_tmp)
                    #print "chunk_arr:",chunk_arr
                    #chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria, start=0, stop=500)
                    # This dataframe will be a chunk of the final total dataframe used in training
                    chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
                    # Add values for the process defined columns.
                    # (i.e. the values that do not change for a given process).
                    chunk_df['weight']=chunk_df['weight']
                    if 'HH' in key:
                        chunk_df['weight_NLO_SM']=chunk_df['weight_NLO_SM']
                    else:
                        chunk_df['weight_NLO_SM']=1.0
                    chunk_df['unweighted'] = 1.0
                    chunk_df['target']=target
                    chunk_df['key']=key
                    chunk_df['classweight']=1.0
                    chunk_df['process_ID']=process_ID
                    #print "chunk_df:",chunk_df['weight'],chunk_df["weight_NLO_SM"]
                    # Append this chunk to the 'total' dataframe
                    data = data.append(chunk_df, ignore_index=True)
                else:
                    print("TTree == None")
                ch_0.Delete()
            tfile.Close()
        if len(data) == 0 : continue

    return data

def check_dir(dir, Website):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)
        if(Website != ""): os.system("cp %s/../index.php %s"%(dir,dir))

def main():
    print('Using Keras version: ', keras.__version__)

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=1, type=int)
    parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
    parser.add_argument('-p', '--para', dest='hyp_param_scan', help='Option to run hyper-parameter scan', default=0, type=int)
    parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing directories \'Bkgs\' and \'Signal\' which contain background and signal ntuples respectively.', default='', type=str)
    parser.add_argument("--LessSamples", action="store_true", help = "Run with minimal backgrounds, signals in order to quickly test network configuration")
    parser.add_argument("--MultiClass", action="store_true", help = "Train a multiclassifier network")
    parser.add_argument("--Website", type = str, default = "", help = "Output files to website path")
    parser.add_argument("--SaveOutput", action="store_true", help = "Save X and Y train and test arrays as pickle files")
    parser.add_argument("-e", "--epochs", type = int, default = 200, help = "Number of epochs to train")
    parser.add_argument("--useKinWeight", action="store_true", help = "Use kinematic MC weights, derived to improve Data / MC agreement, in training")
    parser.add_argument("--VHToGGClassWeightFactor", type = float, default = 1., help = "Factor to adjust Hgg class weights")
    parser.add_argument("--ttHJetToGGClassWeightFactor", type = float, default = 1., help = "Factor to adjust Hgg class weights")
    parser.add_argument("--BkgClassWeightFactor", type = float, default = 1., help = "Factor to adjust bkg class weights")
    parser.add_argument("--channel", type=str, default = "FH", help = "Which channel we are running on FH or SL?")
    parser.add_argument("--IHEPFix", type=bool, default = False, help = "On IHEP machine ")
    args = parser.parse_args()

    print(sys.argv)

    print('#---------------------------------------')
    print('#    Print all input arguments         #')
    print('#---------------------------------------')
    print('train_model                   : %s'%(args.train_model))
    print('suffix                        : %s'%(args.suffix))
    print('Do hyper-parameter scan       : %s'%(args.hyp_param_scan))
    print('inputs_file_path              : %s'%(args.inputs_file_path))
    print("if LessSamples                : %s"%(args.LessSamples))
    print("if MultiClass                 : %s"%(args.MultiClass))
    print("Website path                  : %s"%(args.Website))
    print("if SaveOutput                 : %s"%(args.SaveOutput))
    print("epochs                        : %s"%(args.epochs))
    print("useKinWeight                  : %s"%(args.useKinWeight))
    print("--VHToGGClassWeightFactor     : %s"%(args.VHToGGClassWeightFactor))
    print("--ttHJetToGGClassWeightFactor : %s"%(args.ttHJetToGGClassWeightFactor))
    print("--BkgClassWeightFactor        : %s"%(args.BkgClassWeightFactor))
    print("--channel                     : %s"%(args.channel))
    print("--IHEPFix                     : %s"%(args.IHEPFix))
    print('#---------------------------------------')

    do_model_fit = args.train_model
    suffix = args.suffix
    useKinWeight = args.useKinWeight
    VHToGGClassWeightFactor = args.VHToGGClassWeightFactor
    ttHJetToGGClassWeightFactor = args.ttHJetToGGClassWeightFactor
    BkgClassWeightFactor = args.BkgClassWeightFactor

    # Create instance of the input files directory
    inputs_file_path = args.inputs_file_path
    hyp_param_scan=args.hyp_param_scan

    # Set model hyper-parameters
    weights='BalanceYields'# 'BalanceYields' or 'BalanceNonWeighted'
    optimizer = 'Nadam'
    validation_split=0.1
    # hyper-parameter scan results
    if weights == 'BalanceNonWeighted':
        learn_rate = 0.0005
        epochs = args.epochs
        batch_size=200
    if weights == 'BalanceYields':
        learn_rate = 0.0001
        epochs = args.epochs
        # batch_size=100
        batch_size=500

    # Create instance of output directory where all results are saved.
    output_directory = '%sHHWWyyDNN_%s_%s/' % (args.Website,suffix,weights)
    check_dir(output_directory, args.Website)
    hyperparam_file = os.path.join(output_directory,'additional_model_hyper_params.txt')
    additional_hyperparams = open(hyperparam_file,'w')
    additional_hyperparams.write("optimizer: "+optimizer+"\n")
    additional_hyperparams.write("learn_rate: "+str(learn_rate)+"\n")
    additional_hyperparams.write("epochs: "+str(epochs)+"\n")
    additional_hyperparams.write("validation_split: "+str(validation_split)+"\n")
    additional_hyperparams.write("weights: "+weights+"\n")

    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')
    # print("INPUT VARIABLES JSON INCLUDES CMS_hgg_mass ONLY FOR CHECKING CORRELATION")
    # print("SHOULD NOT TRAIN ON THIS VARIABLE")
    input_var_jsonFile = open('input_variables.json','r')

    # input_var_jsonFile = open('input_variables_withHggMass.json','r')


    # selection_criteria = '( ( fabs(weight * kinWeight) < 10 ) )'
    # selection_criteria = '( 1.>0. )'
    selection_criteria = '( (Leading_Photon_pt/CMS_hgg_mass) > 1/3. && (Subleading_Photon_pt/CMS_hgg_mass) > 1/4. && Leading_Photon_MVA>-0.7 && Subleading_Photon_MVA>-0.7)'

    # Load Variables from .json
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Create list of headers for dataset .csv
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('weight')
    column_headers.append('weight_NLO_SM')
    # column_headers.append('kinWeight')
    column_headers.append('unweighted')
    column_headers.append('target')
    column_headers.append('key')
    column_headers.append('classweight')
    column_headers.append('process_ID')

    # Load ttree into .csv including all variables listed in column_headers
    print('<train-DNN> Input file path: ', inputs_file_path)
    outputdataframe_name = '%s/output_dataframe.csv' %(output_directory)
    if os.path.isfile(outputdataframe_name):
        data = pandas.read_csv(outputdataframe_name)
        print('<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name))
    else:
        print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
        data = load_data(args.channel,inputs_file_path,column_headers,selection_criteria,args.LessSamples, useKinWeight)
        # Change sentinal value to speed up training.
        # data = data.mask(data<-25., -9.)
        data = data.mask(data==np.inf, -9.)
        data = data.mask(data==-np.inf, -9.)
        data = data.mask(data==np.nan, -9.)
        data_inf = data.isin([np.inf, -np.inf])
        data_nan = data.isin([np.nan])
        count_inf = np.isinf(data_inf).values.sum()
        count_nan = np.isinf(data_nan).values.sum()
        if count_inf>0: print("WARNING ---> It contained " + str(count_inf) + " infinite values")
        if count_nan>0: print("WARNING ---> It contained " + str(count_nan) + " NaN values")
        data.to_csv(outputdataframe_name, index=False)
        data = pandas.read_csv(outputdataframe_name)

    print('<main> data columns: ', (data.columns.values.tolist()))
    n = len(data)

    nHH = len(data.iloc[data.target.values == 0])
    # nH = len(data.iloc[data.target.values == 1])
    # nDiPhoton = len(data.iloc[data.target.values == 1])
    nbbgg = len(data.iloc[data.target.values == 1])
    nbkg = len(data.iloc[data.target.values == 2])
    # nQCD = len(data.iloc[data.target.values == 2])

    # print("Total (train+validation) length of HH = %i, Diphoton = %i, QCD = %i" % (nHH, nDiPhoton, nQCD))
    print("Total (train+validation) length of HH = %i, bbgg = %i, bkg = %i" % (nHH, nbbgg, nbkg))

    # Make instance of plotter tool
    Plotter = plotter(args.Website)

    # Create statistically independant training/testing data
    traindataset, valdataset = train_test_split(data, test_size=validation_split)
    valdataset.to_csv((output_directory+'valid_dataset.csv'), index=False)

    print('<train-DNN> Training dataset shape: ', traindataset.shape)
    print('<train-DNN> Validation dataset shape: ', valdataset.shape)

    # Event weights
    weights_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'weight']
    weights_for_HH_NLO = traindataset.loc[traindataset['process_ID']=='HH', 'weight_NLO_SM']
    weights_for_bbgg = traindataset.loc[traindataset['process_ID']=='bbgg', 'weight']
    weights_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'weight']
    weights_for_QCD = traindataset.loc[traindataset['process_ID']=='QCD', 'weight']
    weights_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'weight']
    # weights_for_VHToGG = traindataset.loc[traindataset['process_ID']=='VHToGG', 'weight']
    # weights_for_ttHJetToGG = traindataset.loc[traindataset['process_ID']=='ttHJetToGG', 'weight']
    # weights_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'weight']
    # weights_for_QCD = traindataset.loc[traindataset['process_ID']=='QCD', 'weight']
    # weights_for_bckg = traindataset.loc[traindataset['process_ID']=='bkg', 'weight']
    # weights_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'weight']
    # weights_for_WGsJets = traindataset.loc[traindataset['process_ID']=='WGsJets', 'weight']
    # weights_for_WW = traindataset.loc[traindataset['process_ID']=='WW', 'weight']

    ##-- Compute weight sums
    # BkgProcs = ["DiPhoton", "GJet", "QCD", "DY", "TTGsJets", "WGsJets", "WW"]
    # BkgProcs = ["DiPhoton", "QCD"]
    # BkgProcs = ["DiPhoton","QCD", "TTGsJets"]
    BkgProcs = ["bbgg", "DiPhoton","QCD", "TTGsJets"]

    ##-- Computed Weighted sums for HH and Hgg, no kinweights
    # XS_HH = 31.049
    # BR_HH_WWgg = 0.000970198
    # BR_WWgg_qqlnu = 0.441 ##-- Semileptonic WW decay

    HHsum_weighted = sum(weights_for_HH*weights_for_HH_NLO)
    # HHsum_weighted = sum(weights_for_HH)

    ##-- To apply HH XS and BR normalization to HHsum weighted
    # HHsum_weighted = HHsum_weighted * XS_HH * BR_HH_WWgg * BR_WWgg_qqlnu
    BBgg_weighted = sum(weights_for_bbgg)

    DiPhotonsum_weighted = sum(weights_for_DiPhoton)
    QCDsum_weighted = sum(weights_for_QCD)
    TTGsJetssum_weighted = sum(weights_for_TTGsJets)

    # GJetsum_weighted = sum(weights_for_GJet)
    # Bckg_weighted = sum(weights_for_bckg)
    # DYsum_weighted = sum(weights_for_DY)
    # WGsJetssum_weighted = sum(weights_for_WGsJets)
    # WWsum_weighted = sum(weights_for_WW)
    # VHToGGsum_weighted = sum(weights_for_VHToGG)
    # ttHJetToGGsum_weighted = sum(weights_for_ttHJetToGG)

    ##-- If using kinematic weights, extract values from files and multiply with MC weights ('weight' branch) per event for MC weighted sums
    # if(useKinWeight):
    #     print("Using kinematic weights in MC weighted sum computation")
    #     for BkgProc in BkgProcs:
    #         exec("kinWeights_for_%s = traindataset.loc[traindataset['process_ID'] == '%s', 'kinWeight']"%(BkgProc, BkgProc))
    #         exec("%ssum_weighted = sum(weights_for_%s * kinWeights_for_%s)"%(BkgProc, BkgProc, BkgProc)) ##-- weighted sum computed from sum of products of MC weight and kinweight

    # ##-- If not using kinematic weights, compute weighted event sum only from MC weights
    # elif(not useKinWeight):
    #     print("NOT using kinematic weights in MC weighted sum computation")
    #     for BkgProc in BkgProcs:
    #         exec("%ssum_weighted = sum(weights_for_%s)"%(BkgProc, BkgProc)) ##-- weighted sum only computed from MC weight ('weight' branch)


    ##-- Already know we want to focus more on VH than ttH
    # VHToGGsum_weighted = (VHToGGsum_weighted * VHToGGClassWeightFactor) / 2.
    # VHToGGsum_weighted = (VHToGGsum_weighted * VHToGGClassWeightFactor)
    # ttHJetToGGsum_weighted = ttHJetToGGsum_weighted * ttHJetToGGClassWeightFactor

    # bckgsum_weighted = bckgsum_weighted * BkgClassWeightFactor

    # Hggsum_weighted = VHToGGsum_weighted + ttHJetToGGsum_weighted

    ##-- If running a multiclass neural network, do not include Hgg event sum in background sum, as this constitutes a different class
    # if(args.MultiClass): bckgsum_weighted = DiPhotonsum_weighted + GJetsum_weighted + QCDsum_weighted + DYsum_weighted + TTGsJetssum_weighted + WGsJetssum_weighted + WWsum_weighted
    # if(args.MultiClass): bckgsum_weighted = DiPhotonsum_weighted + QCDsum_weighted
    # if(args.MultiClass): bckgsum_weighted = weights_for_bbgg + weights_for_DiPhoton + weights_for_QCD + weights_for_TTGsJets
    if(args.MultiClass): bckgsum_weighted =  DiPhotonsum_weighted + QCDsum_weighted + TTGsJetssum_weighted
    # else: bckgsum_weighted = Hggsum_weighted + DiPhotonsum_weighted + GJetsum_weighted + QCDsum_weighted + DYsum_weighted + TTGsJetssum_weighted + WGsJetssum_weighted + WWsum_weighted
    # else: bckgsum_weighted = VHToGGsum_weighted + ttHJetToGGsum_weighted + DiPhotonsum_weighted + GJetsum_weighted + QCDsum_weighted + DYsum_weighted + TTGsJetssum_weighted + WGsJetssum_weighted + WWsum_weighted
    else: bckgsum_weighted = BBgg_weighted + DiPhotonsum_weighted + QCDsum_weighted + TTGsJetssum_weighted

    bckgsum_weighted = bckgsum_weighted * BkgClassWeightFactor

    nevents_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'unweighted']

    nevents_for_bbgg = traindataset.loc[traindataset['process_ID']=='bbgg', 'unweighted']

    nevents_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'unweighted']
    nevents_for_QCD = traindataset.loc[traindataset['process_ID']=='QCD', 'unweighted']
    nevents_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'unweighted']

    # nevents_for_VHToGG = traindataset.loc[traindataset['process_ID']=='VHToGG', 'unweighted']
    # nevents_for_ttHJetToGG = traindataset.loc[traindataset['process_ID']=='ttHJetToGG', 'unweighted']
    # nevents_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'unweighted']
    # nevents_for_bckg = traindataset.loc[traindataset['process_ID']=='bkg', 'unweighted']
    # nevents_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'unweighted']
    # nevents_for_WGsJets = traindataset.loc[traindataset['process_ID']=='WGsJets', 'unweighted']
    # nevents_for_WW = traindataset.loc[traindataset['process_ID']=='WW', 'unweighted']

    HHsum_unweighted= sum(nevents_for_HH)

    BBggsum_unweighted= sum(nevents_for_bbgg)

    DiPhotonsum_unweighted = sum(nevents_for_DiPhoton)
    QCDsum_unweighted = sum(nevents_for_QCD)
    TTGsJetssum_unweighted = sum(nevents_for_TTGsJets)
    # VHToGGsum_unweighted= sum(nevents_for_VHToGG)
    # ttHJetToGGsum_unweighted= sum(nevents_for_ttHJetToGG)
    # GJetsum_unweighted= sum(nevents_for_GJet)
    # Bckgsum_unweighted= sum(nevents_for_bckg)
    # DYsum_unweighted= sum(nevents_for_DY)
    # WGsJetssum_unweighted= sum(nevents_for_WGsJets)
    # WWsum_unweighted= sum(nevents_for_WW)
    # if(args.MultiClass): bckgsum_unweighted = DiPhotonsum_unweighted + GJetsum_unweighted + QCDsum_unweighted + DYsum_unweighted + TTGsJetssum_unweighted + WGsJetssum_unweighted + WWsum_unweighted
    # if(args.MultiClass): bckgsum_unweighted = DiPhotonsum_unweighted + QCDsum_unweighted
    # if(args.MultiClass): bckgsum_unweighted = BBggsum_unweighted + DiPhotonsum_unweighted + QCDsum_unweighted + TTGsJetssum_unweighted
    if(args.MultiClass): bckgsum_unweighted =  DiPhotonsum_unweighted + QCDsum_unweighted + TTGsJetssum_unweighted
    else: bckgsum_unweighted = BBggsum_unweighted + DiPhotonsum_unweighted + QCDsum_unweighted + TTGsJetssum_unweighted
    # else: bckgsum_unweighted = VHToGGsum_unweighted + ttHJetToGGsum_unweighted + DiPhotonsum_unweighted + GJetsum_unweighted + QCDsum_unweighted + DYsum_unweighted + TTGsJetssum_unweighted + WGsJetssum_unweighted + WWsum_unweighted

    ##-- Adjust class weights if desired
    if(ttHJetToGGClassWeightFactor != 1.):
        print("Adjusting Hgg class weights: weighted / unweighted sums scaled by: ", ttHJetToGGClassWeightFactor)
    if(VHToGGClassWeightFactor != 1.):
        print("Adjusting Hgg class weights: weighted / unweighted sums scaled by: ", VHToGGClassWeightFactor)
    if(BkgClassWeightFactor != 1.):
        print("Adjusting class weights: weighted / unweighted sums scaled by: ", BkgClassWeightFactor)

    ##-- Already know we want to focus more on VH than ttH
    # VHToGGsum_unweighted = VHToGGsum_unweighted * VHToGGClassWeightFactor
    # ttHJetToGGsum_unweighted = ttHJetToGGsum_unweighted * ttHJetToGGClassWeightFactor
    bckgsum_unweighted = bckgsum_unweighted * BkgClassWeightFactor

    # Hggsum_unweighted = VHToGGsum_unweighted + ttHJetToGGsum_unweighted

    ##-- Define class weights
    if weights=='BalanceYields':
        print('HHsum_weighted= ' , HHsum_weighted)

        print('BBgg_weighted= ', BBgg_weighted)

        print('DiPhotonsum_weighted= ', DiPhotonsum_weighted)
        print('QCDsum_weighted= ', QCDsum_weighted)
        print('TTGsJetssum_weighted= ', TTGsJetssum_weighted)
        # print('Hggsum_weighted= ' , Hggsum_weighted)
        # print('GJetsum_weighted= ', GJetsum_weighted)
        # print('QCDsum_weighted= ', QCDsum_weighted)
        # print('Bckg_weighted= ', Bckg_weighted)
        # print('DYsum_weighted= ', DYsum_weighted)
        # print('WGsJetssum_weighted= ', WGsJetssum_weighted)
        # print('WWsum_weighted= ', WWsum_weighted)

        print('bckgsum_weighted = ', bckgsum_weighted)

        ##-- Choose class weight scale target
        classweight_Target = HHsum_unweighted
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = classweight_Target/HHsum_weighted

        if(args.MultiClass):

            ##-- Output unweighted, weighted yields and class weights to tex style table

            ##-- TeX file table
            fileName = "DNN_YieldsAndWeights.tex"
            file = open(fileName,"w")
            file.write("\\begin{table}[H]\n")
            file.write("\t\\begin{center}\n")
            file.write("\t\t\\begin{tabular}{c|c|c|c}\n")
            file.write("\t\t\tClass & Unweighted Yield & Weighted Yield & Class Weight \\\ \\hline \n")

            ##-- Scale ttHJet and VH separately based on their weighted yields
            # traindataset.loc[traindataset['process_ID']=='VHToGG', ['classweight']] = (classweight_Target/VHToGGsum_weighted) ##-- Scale all Hgg events to HH unweighted
            # traindataset.loc[traindataset['process_ID']=='ttHJetToGG', ['classweight']] = (classweight_Target/ttHJetToGGsum_weighted) ##-- Scale all Hgg events to HH unweighted

            ##-- Scale ttHJet and VH in the same way - based on weighted sum of Hgg yields
            # traindataset.loc[traindataset['process_ID']=='VHToGG', ['classweight']] = (classweight_Target/Hggsum_weighted) ##-- Scale all Hgg events to HH unweighted
            # traindataset.loc[traindataset['process_ID']=='ttHJetToGG', ['classweight']] = (classweight_Target/Hggsum_weighted) ##-- Scale all Hgg events to HH unweighted

            ##-- Printout class weights
            print('----[HH class]----')
            print('HH unweighted:',int(HHsum_unweighted))
            print('HH weighted:' , round(HHsum_weighted, 6))
            print('HH Class Weight:',round(classweight_Target/HHsum_weighted, 6))
            print(' ')
            print('----[bbgg class]----')
            print('bbgg unweighted:' , round(BBggsum_unweighted, 6))
            print('bbgg weighted:' , round(BBgg_weighted, 6))
            print('bbgg Class Weight:',round(classweight_Target/BBgg_weighted, 6))

            # print('----[H class]----')
            # print('VHToGG unweighted:' , int(VHToGGsum_unweighted))
            # print('VHToGG weighted:' , round(VHToGGsum_weighted, 6))
            # print('VHToGG Class Weight:',(round(classweight_Target/VHToGGsum_weighted, 6)))
            # print('ttHJetToGG unweighted:' , int(ttHJetToGGsum_unweighted))
            # print('ttHJetToGG weighted:' , round(ttHJetToGGsum_weighted, 6))
            # print('ttHJetToGG Class Weight:',(round(classweight_Target/ttHJetToGGsum_weighted, 6)))
            # print('Hgg unweighted:' , int(Hggsum_unweighted))
            # print('Hgg weighted:' , round(Hggsum_weighted, 6))
            # print('Hgg Class Weight:',(round(classweight_Target/Hggsum_weighted, 6)))
            # print(' ')
            # print('----[QCD class]----')
            # print('QCD unweighted:', int(QCDsum_unweighted))
            # print('QCD weighted:', int(QCDsum_weighted))
            print(' ')
            # print('----[DiPhoton class]----')
            # print('DiPhoton unweighted:', int(DiPhotonsum_unweighted))
            # print('DiPhotonsum_weighted:', round(DiPhotonsum_weighted, 6))
            print(' ')
            # print('GJet unweighted:', int(GJetsum_unweighted))
            # print('GJetsum_weighted:', round(GJetsum_weighted, 6))
            # print(' ')
            # print('TTGsJets unweighted:', int(TTGsJetssum_unweighted))
            # print('TTGsJetssum_weighted:', round(TTGsJetssum_weighted, 6))
            # print(' ')
            # print('WGsJets unweighted:', int(WGsJetssum_unweighted))
            # print('WGsJetssum_weighted:', round(WGsJetssum_weighted, 6))
            # print(' ')
            print('----[Bkg class]----')
            print('Bkg Unweighted:', int(bckgsum_unweighted))
            print('Bkg Weighted:', round(bckgsum_weighted, 6))
            print(' ')
            print('Bkg Class Weight:',round((classweight_Target/bckgsum_weighted), 6))

            file.write("\t\t\t HH & %s & %s & %s \\\ \n"%(int(HHsum_unweighted), round(HHsum_weighted, 4), round(classweight_Target/HHsum_weighted, 4)))
            # file.write("\t\t\t H & %s & %s & %s \\\ \n"%(int(Hggsum_unweighted), round(Hggsum_weighted, 4), round(classweight_Target/Hggsum_weighted, 4)))
            file.write("\t\t\t Continuum Background & %s & %s & %s \\\ \n"%(int(bckgsum_unweighted), round(bckgsum_weighted, 4), round(classweight_Target/bckgsum_weighted, 4)))

            file.write("\t\t\end{tabular}\n")
            file.write("\t\caption{Unweighted and weighted yields, and class weights applied in the Semi-Leptonic DNN training}\n")
            file.write("\t\\end{center}\n")
            file.write("\end{table}\n")

            file.close()


        else: traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (classweight_Target/bckgsum_weighted)

        traindataset.loc[traindataset['process_ID']=='bbgg', ['classweight']] = (classweight_Target/bckgsum_weighted)

        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (classweight_Target/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='QCD', ['classweight']] = (classweight_Target/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (classweight_Target/bckgsum_weighted)

        # traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (classweight_Target/bckgsum_weighted)
        # traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (classweight_Target/bckgsum_weighted)
        # traindataset.loc[traindataset['process_ID']=='WGsJets', ['classweight']] = (classweight_Target/bckgsum_weighted)
        # traindataset.loc[traindataset['process_ID']=='WW', ['classweight']] = (classweight_Target/bckgsum_weighted)

    if weights=='BalanceNonWeighted':
        print('HHsum_unweighted= ' , HHsum_unweighted)

        print('bbggsum_unweighted= ' , bbggsum_unweighted)

        print('DiPhotonsum_unweighted= ', DiPhotonsum_unweighted)
        print('QCDsum_unweighted= ', QCDsum_unweighted)
        print('TTGsJetssum_unweighted= ', TTGsJetssum_unweighted)

        # print('Hggsum_unweighted= ' , Hggsum_unweighted)
        # print('GJetsum_unweighted= ', GJetsum_unweighted)
        # print('DYsum_unweighted= ', DYsum_unweighted)
        # print('WGsJetssum_unweighted= ', WGsJetssum_unweighted)
        # print('WWsum_unweighted= ', WWsum_unweighted)
        print('bckgsum_unweighted= ', bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = 1.

        traindataset.loc[traindataset['process_ID']=='bbgg', ['classweight']] = (classweight_Target/bckgsum_unweighted)

        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='QCD', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (classweight_Target/bckgsum_unweighted)

        # traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='WGsJets', ['classweight']] = (classweight_Target/bckgsum_unweighted)
        # traindataset.loc[traindataset['process_ID']=='WW', ['classweight']] = (classweight_Target/bckgsum_unweighted)

    # Remove column headers that aren't input variables
    # nonTrainingVariables = ['weight', 'weight_NLO_SM', 'kinWeight', 'unweighted', 'target', 'key', 'classweight', 'process_ID']
    nonTrainingVariables = ['weight', 'weight_NLO_SM', 'unweighted', 'target', 'key', 'classweight', 'process_ID']

    # column_headers.append('weight')
    # column_headers.append('weight_NLO_SM')
    # column_headers.append('kinWeight')
    # column_headers.append('unweighted')
    # column_headers.append('target')
    # column_headers.append('key')
    # column_headers.append('classweight')
    # column_headers.append('process_ID')

    training_columns = [h for h in column_headers if h not in nonTrainingVariables]

    # training_columns = column_headers[:-5]
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
    # train_weights = traindataset['weight'].values*traindataset['weight_NLO_SM'].values
    # test_weights = valdataset['weight'].values*valdataset['weight_NLO_SM'].values
    # train_weights = abs(traindataset['weight'].values)*abs(traindataset['weight_NLO_SM'].values)
    # test_weights = abs(valdataset['weight'].values)*abs(valdataset['weight_NLO_SM'].values)
    # train_weights = abs(traindataset['weight'].values)*abs(traindataset['weight_NLO_SM'].values)*abs(traindataset['kinWeight'].values)
    # test_weights = abs(valdataset['weight'].values)*abs(valdataset['weight_NLO_SM'].values)*abs(traindataset['kinWeight'].values)
    train_weights = abs(traindataset['weight'].values)*abs(traindataset['weight_NLO_SM'].values)
    test_weights  = abs(valdataset['weight'].values)*abs(valdataset['weight_NLO_SM'].values)
    # test_weights = abs(valdataset['weight'].values)
    # train_weights = abs(traindataset['weight'].values)

    if(useKinWeight):
        print("INCLUDING kinematic weights in training and test weights")
        train_weights = abs(traindataset['weight'].values)*abs(traindataset['weight_NLO_SM'].values)*abs(traindataset['kinWeight'].values)
        test_weights = abs(valdataset['weight'].values)*abs(valdataset['weight_NLO_SM'].values)*abs(valdataset['kinWeight'].values)

    # Weights applied during training.
    if weights=='BalanceYields':
        if(useKinWeight):
            print("INCLUDING kinWeight in training weights")
            trainingweights = traindataset.loc[:,'classweight'].abs()*traindataset.loc[:,'weight'].abs()*traindataset.loc[:,'weight_NLO_SM'].abs()*traindataset.loc[:,'kinWeight'].abs()
        elif(not useKinWeight):
            print("NOT INCLUDING kinWeight in training weights")
            trainingweights = traindataset.loc[:,'classweight']*traindataset.loc[:,'weight']*traindataset.loc[:,'weight_NLO_SM']

    if weights=='BalanceNonWeighted':
        trainingweights = traindataset.loc[:,'classweight']*traindataset.loc[:,'weight_NLO_SM']
    trainingweights = np.array(trainingweights)

    ## Input Variable Correlation plot
    correlation_plot_file_name = 'correlation_plot'
    Plotter.correlation_matrix(train_df, args.MultiClass)
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.png')
    Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.pdf')

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)

    # Transform to one hot encoded arrays for MultiClassifier
    if(args.MultiClass):
        Y_train = utils.to_categorical(encoded_Y)
        Y_test = utils.to_categorical(encoded_Y_test)

    if do_model_fit == 1:
        print('<train-DNN> Training new model . . . . ')
        histories = []
        labels = []

        if hyp_param_scan == 1:
            print('Begin at local time: ', time.localtime())
            hyp_param_scan_name = output_directory+'/hyp_param_scan_results.txt'
            hyp_param_scan_results = open(hyp_param_scan_name,'a')
            time_str = str(time.localtime())+'\n'
            hyp_param_scan_results.write(time_str)
            hyp_param_scan_results.write(weights)
            learn_rates=[0.00001, 0.0001]
            epochs = [150,200]
            batch_size = [200, 400]
            param_grid = dict(learn_rate=learn_rates,epochs=epochs,batch_size=batch_size)
            model = KerasClassifier(build_fn=new_model5,verbose=1)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
            grid_result = grid.fit(X_train,Y_train,shuffle=True,verbose=1,sample_weight=trainingweights)
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
            early_stopping_monitor = EarlyStopping(patience=100, monitor='val_loss', min_delta=0.01, verbose=1)

            if(args.MultiClass):
                # nClasses = 2 ##-- HH, H or maybe HH, (H + continuum)
                nClasses = 3 ##-- HH, H, Bkg (continuum)
                # model = MultiClassifier_Model(num_variables, nClasses, learn_rate=learn_rate)
                model = new_model5(num_variables, nClasses, learn_rate=learn_rate)
            else:
                model = new_model(num_variables, learn_rate=learn_rate)

            # Fit the model
            # Batch size = examples before updating weights (larger = faster training)
            # Epoch = One pass over data (useful for periodic logging and evaluation)
            #class_weights = np.array(class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train))

            ##-- Note: Removes early_stopping_monitor. Should only be used for a stable training
            if(args.SaveOutput):
                csv_logger = CSVLogger('%s/training.log'%(output_directory), separator=',', append=False)
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[csv_logger,early_stopping_monitor])

            else:
                history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor])
            histories.append(history)
            labels.append(optimizer)

            from_log = 0 ##-- Not plotting from a log file, but from a callback
            Plotter.history_plot(history, from_log, label='loss')
            Plotter.save_plots(dir=plots_dir, filename='history_loss.png')
            Plotter.save_plots(dir=plots_dir, filename='history_loss.pdf')

            Plotter.history_plot(history, from_log, label='acc')
            Plotter.save_plots(dir=plots_dir, filename='history_acc.png')
            Plotter.save_plots(dir=plots_dir, filename='history_acc.pdf')

    else:
        model_name = os.path.join(output_directory,'model.h5')
        model = load_trained_model(model_name)

    # Node probabilities for training sample events
    result_probs = model.predict(np.array(X_train))
    # result_classes = model.predict_classes(np.array(X_train))
    # model.predict_classes is going to be deprecated.. so one should use np.argmax
    result_classes = np.argmax(model.predict(np.array(X_train)), axis=-1)

    # Node probabilities for testing sample events
    result_probs_test = model.predict(np.array(X_test))
    #result_classes_test = model.predict_classes(np.array(X_test))

    # Store model in file
    model_output_name = os.path.join(output_directory,'model.h5')
    model.save(model_output_name)
    weights_output_name = os.path.join(output_directory,'model_weights.h5')
    model.save_weights(weights_output_name)
    model_json = model.to_json()
    model_json_name = os.path.join(output_directory,'model_serialised.json')

    ##-- Convert model to pb
    CONVERT_COMMAND = "python ../convert_hdf5_2_pb.py --input %s/model.h5 --output %s/model.pb"%(output_directory, output_directory)
    print("Converting model.h5 to model.pb...")
    print(CONVERT_COMMAND)
    os.system(CONVERT_COMMAND)

    with open(model_json_name,'w') as json_file:
        json_file.write(model_json)
    model.summary()
    model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)
    model_schematic_name = os.path.join(output_directory,'model_schematic.pdf')
    plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

    print('================')
    print('Training event labels: ', len(Y_train))
    print('Training event probs', len(result_probs))
    print('Training event weights: ', len(train_weights))
    print('Testing events: ', len(Y_test))
    print('Testing event probs', len(result_probs_test))
    print('Testing event weights: ', len(test_weights))
    print('================')

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    ##-- Save test and training data to study a configuration's output without retraining. NOTE: These files may be very large depending on the number of training events
    if(args.SaveOutput):
        print("[train-DNN.py] - Saving outputs as pickle files")
        objectsToSave = ["X_test","Y_test","X_train","Y_train","train_df", "train_weights", "test_weights"]
        # objectsToSave = ["X_test","Y_test","X_train","Y_train","train_df"]
        for objToSave in objectsToSave:
            print("Saving %s..."%(objToSave))
            executeLine = "pickle.dump( %s , open( '%s/%s.p', 'wb' ) )"%(objToSave, output_directory, objToSave)
            exec(executeLine)

    if(args.MultiClass):
        print("[train-DNN.py#983]....")
        Plotter.ROC_MultiClassifier(model, X_test, Y_test, X_train, Y_train)
    else:
        Plotter.ROC(model, X_test, Y_test, X_train, Y_train)
        Plotter.save_plots(dir=plots_dir, filename='ROC.png')
        Plotter.save_plots(dir=plots_dir, filename='ROC.pdf')

main()
