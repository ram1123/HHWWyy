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
from sklearn.utils import class_weight
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

def load_data(inputPath,variables,criteria):
    # Load dataset to .csv format file
    my_cols_list=variables#+['key', 'target']
    data = pd.DataFrame(columns=my_cols_list)
    keys=['HH','bckg']
    #keys=['HH']
    data = pd.DataFrame(columns=my_cols_list)
    for key in keys :
        print 'key: ', key
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
            elif 'DiPhotonJetsBox_MGG' in filen:
                treename=['tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_0','tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_1']
                #treename=['tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_1']
            elif 'GJet_Pt-40toInf' in filen:
                treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_2']
                # treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0','tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_1']
                #treename=['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_1']
            elif 'DYJetsToLL_M-50_TuneCP5' in filen:
                treename=['tagsDumper/trees/DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_2']
                #treename=['tagsDumper/trees/DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_1']
            elif 'TTGG' in filen:
                treename=['tagsDumper/trees/TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_2']
                #,'tagsDumper/trees/TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_1']
            elif 'TTGJets' in filen:
                treename=['tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_2']
                #,'tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_1']

            fileName = os.path.join(subdir_name,filen)
            filename_fullpath = inputPath+"/"+fileName+".root"
            print "Input file: ", filename_fullpath
            tfile = ROOT.TFile(filename_fullpath)
            for tname in treename:
                print 'tname: ', tname
                ch_0 = tfile.Get(tname)
                if ch_0 is not None :
                    '''
                    #============= Create new lepton collections =====================
                    nentries = ch_0.GetEntries()
                    ch_0.SetBranchStatus("*",1)
                    skimmed_file = ROOT.TFile("skimmed_file.root","RECREATE")
                    skimmed_tree = TTree("skimmed_tree", "skimmed tree")
                    N_goodJets = array('d',[0])
                    goodJets_0_pt = array('d',[0])
                    goodJets_0_eta = array('d',[0])
                    goodJets_0_phi = array('d',[0])
                    goodJets_0_E = array('d',[0])
                    goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb = array('d',[0])
                    goodJets_1_pt = array('d',[0])
                    goodJets_1_eta = array('d',[0])
                    goodJets_1_phi = array('d',[0])
                    goodJets_1_E = array('d',[0])
                    goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb = array('d',[0])
                    Leading_Photon_E = array('d',[0])
                    Leading_Photon_pt = array('d',[0])
                    Leading_Photon_eta = array('d',[0])
                    Leading_Photon_phi = array('d',[0])
                    Leading_Photon_MVA = array('d',[0])
                    Subleading_Photon_E = array('d',[0])
                    Subleading_Photon_pt = array('d',[0])
                    Subleading_Photon_eta = array('d',[0])
                    Subleading_Photon_phi = array('d',[0])
                    Subleading_Photon_MVA = array('d',[0])
                    weight = array('d',[0])
                    looser_lep0_pt = array('d',[0])
                    looser_lep0_eta = array('d',[0])
                    looser_lep0_phi = array('d',[0])
                    looser_lep0_E = array('d',[0])
                    looser_lep0_flav = array('d',[0])
                    skimmed_tree.Branch("weight", weight, "weight/D")
                    skimmed_tree.Branch("N_goodJets", N_goodJets, "N_goodJets/D")
                    skimmed_tree.Branch("goodJets_0_pt", goodJets_0_pt, "goodJets_0_pt/D")
                    skimmed_tree.Branch("goodJets_0_eta", goodJets_0_eta, "goodJets_0_eta/D")
                    skimmed_tree.Branch("goodJets_0_phi", goodJets_0_phi, "goodJets_0_phi/D")
                    skimmed_tree.Branch("goodJets_0_E", goodJets_0_E, "goodJets_0_E/D")
                    skimmed_tree.Branch("goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb", goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb, "goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb/D")
                    skimmed_tree.Branch("goodJets_1_pt", goodJets_1_pt, "goodJets_1_pt/D")
                    skimmed_tree.Branch("goodJets_1_eta", goodJets_1_eta, "goodJets_1_eta/D")
                    skimmed_tree.Branch("goodJets_1_phi", goodJets_1_phi, "goodJets_1_phi/D")
                    skimmed_tree.Branch("goodJets_1_E", goodJets_1_E, "goodJets_1_E/D")
                    skimmed_tree.Branch("goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb", goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb, "goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb/D")
                    skimmed_tree.Branch("Leading_Photon_E", Leading_Photon_E, "Leading_Photon_E/D")
                    skimmed_tree.Branch("Leading_Photon_pt", Leading_Photon_pt, "Leading_Photon_pt/D")
                    skimmed_tree.Branch("Leading_Photon_eta", Leading_Photon_eta, "Leading_Photon_eta/D")
                    skimmed_tree.Branch("Leading_Photon_phi", Leading_Photon_phi, "Leading_Photon_phi/D")
                    skimmed_tree.Branch("Leading_Photon_MVA", Leading_Photon_MVA, "Leading_Photon_MVA/D")
                    skimmed_tree.Branch("Subleading_Photon_E", Subleading_Photon_E, "Subleading_Photon_E/D")
                    skimmed_tree.Branch("Subleading_Photon_pt", Subleading_Photon_pt, "Subleading_Photon_pt/D")
                    skimmed_tree.Branch("Subleading_Photon_eta", Subleading_Photon_eta, "Subleading_Photon_eta/D")
                    skimmed_tree.Branch("Subleading_Photon_phi", Subleading_Photon_phi, "Subleading_Photon_phi/D")
                    skimmed_tree.Branch("Subleading_Photon_MVA", Subleading_Photon_MVA, "Subleading_Photon_MVA/D")
                    skimmed_tree.Branch("looser_lep0_pt", looser_lep0_pt, "looser_lep0_pt/D")
                    skimmed_tree.Branch("looser_lep0_eta", looser_lep0_eta, "looser_lep0_eta/D")
                    skimmed_tree.Branch("looser_lep0_phi", looser_lep0_phi, "looser_lep0_phi/D")
                    skimmed_tree.Branch("looser_lep0_E", looser_lep0_E, "looser_lep0_E/D")
                    skimmed_tree.Branch("looser_lep0_flav", looser_lep0_flav, "looser_lep0_flav/D")
                    if 'DiPhoton' in fileName and nentries > 100000:
                        nentries = 100000
                    print 'nentries: ', nentries
                    for event in range(nentries):
                        ch_0.GetEntry(event)
                        percent_done = event/(nentries/100.)
                        if percent_done%10.==0.:
                            print "Percentage done: ", event/(nentries/100.)

                        if ch_0.goodJets!=1 or ch_0.passbVeto!=1 or ch_0.passPhotonSels!=1:
                            continue

                        ele_id_array = [ch_0.allElectrons_0_passLooseId, ch_0.allElectrons_1_passLooseId, ch_0.allElectrons_2_passLooseId, ch_0.allElectrons_3_passLooseId, ch_0.allElectrons_4_passLooseId]
                        ele_pt_array = [ch_0.allElectrons_0_pt, ch_0.allElectrons_1_pt, ch_0.allElectrons_2_pt, ch_0.allElectrons_3_pt, ch_0.allElectrons_4_pt]
                        ele_eta_array = [ch_0.allElectrons_0_eta, ch_0.allElectrons_1_eta, ch_0.allElectrons_2_eta, ch_0.allElectrons_3_eta, ch_0.allElectrons_4_eta]
                        ele_phi_array = [ch_0.allElectrons_0_phi, ch_0.allElectrons_1_phi, ch_0.allElectrons_2_phi, ch_0.allElectrons_3_phi, ch_0.allElectrons_4_phi]
                        ele_E_array = [ch_0.allElectrons_0_E, ch_0.allElectrons_1_E, ch_0.allElectrons_2_E, ch_0.allElectrons_3_E, ch_0.allElectrons_4_E]
                        count_loose_ele=0
                        loose_ele_pt = -1
                        loose_ele_eta = -1
                        loose_ele_phi = -1
                        loose_ele_E = -1
                        for ele in xrange(len(ele_id_array)):
                            if ele_id_array[ele]>0 and ele_pt_array[ele]>=10. and ele_eta_array[ele]<2.5:
                                count_loose_ele+=1
                                loose_ele_pt = ele_pt_array[ele]
                                loose_ele_eta = ele_eta_array[ele]
                                loose_ele_phi = ele_phi_array[ele]
                                loose_ele_E = ele_E_array[ele]
                                looser_lep0_flav = 0
                        mu_id_array = [ch_0.allMuons_0_isLooseMuon, ch_0.allMuons_1_isLooseMuon, ch_0.allMuons_2_isLooseMuon, ch_0.allMuons_3_isLooseMuon, ch_0.allMuons_4_isLooseMuon]
                        mu_pt_array = [ch_0.allMuons_0_pt, ch_0.allMuons_1_pt, ch_0.allMuons_2_pt, ch_0.allMuons_3_pt, ch_0.allMuons_4_pt]
                        mu_eta_array = [ch_0.allMuons_0_eta, ch_0.allMuons_1_eta, ch_0.allMuons_2_eta, ch_0.allMuons_3_eta, ch_0.allMuons_4_eta]
                        mu_phi_array = [ch_0.allMuons_0_phi, ch_0.allMuons_1_phi, ch_0.allMuons_2_phi, ch_0.allMuons_3_phi, ch_0.allMuons_4_phi]
                        mu_E_array = [ch_0.allMuons_0_E, ch_0.allMuons_1_E, ch_0.allMuons_2_E, ch_0.allMuons_3_E, ch_0.allMuons_4_E]
                        count_loose_mu=0
                        loose_mu_pt = -1
                        loose_mu_eta = -1
                        loose_mu_phi = -1
                        loose_mu_E = -1
                        for mu in xrange(len(mu_id_array)):
                            if mu_id_array[mu]>0 and mu_pt_array[mu]>=10. and mu_eta_array[mu]<2.5:
                                count_loose_mu+=1
                                loose_mu_pt = mu_pt_array[mu]
                                loose_mu_eta = mu_eta_array[mu]
                                loose_mu_phi = mu_phi_array[mu]
                                loose_mu_E = mu_E_array[mu]
                        if (count_loose_mu+count_loose_ele)==1:
                            if count_loose_mu==1:
                                looser_lep0_pt[0] = loose_mu_pt
                                looser_lep0_eta[0] = loose_mu_eta
                                looser_lep0_phi[0] = loose_mu_phi
                                looser_lep0_E[0] = loose_mu_E
                                looser_lep0_flav = 0
                            if count_loose_ele==1:
                                looser_lep0_pt[0] = loose_ele_pt
                                looser_lep0_eta[0] = loose_ele_eta
                                looser_lep0_phi[0] = loose_ele_phi
                                looser_lep0_E[0] = loose_ele_E
                                looser_lep0_flav = 1
                            N_goodJets[0] = ch_0.N_goodJets
                            weight[0] = ch_0.weight
                            goodJets_0_pt[0] = ch_0.goodJets_0_pt
                            goodJets_0_eta[0] = ch_0.goodJets_0_eta
                            goodJets_0_phi[0] = ch_0.goodJets_0_phi
                            goodJets_0_E[0] = ch_0.goodJets_0_E
                            goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb[0] = ch_0.goodJets_0_bDiscriminator_mini_pfDeepFlavourJetTags_probb
                            goodJets_1_pt[0] = ch_0.goodJets_1_pt
                            goodJets_1_eta[0] = ch_0.goodJets_1_eta
                            goodJets_1_phi[0] = ch_0.goodJets_1_phi
                            goodJets_1_E[0] = ch_0.goodJets_1_E
                            goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb[0] = ch_0.goodJets_1_bDiscriminator_mini_pfDeepFlavourJetTags_probb
                            Leading_Photon_E[0] = ch_0.Leading_Photon_E
                            Leading_Photon_pt[0] = ch_0.Leading_Photon_pt
                            Leading_Photon_eta[0] = ch_0.Leading_Photon_eta
                            Leading_Photon_phi[0] = ch_0.Leading_Photon_phi
                            Leading_Photon_MVA[0] = ch_0.Leading_Photon_MVA
                            Subleading_Photon_E[0] = ch_0.Subleading_Photon_E
                            Subleading_Photon_pt[0] = ch_0.Subleading_Photon_pt
                            Subleading_Photon_eta[0] = ch_0.Subleading_Photon_eta
                            Subleading_Photon_phi[0] = ch_0.Subleading_Photon_phi
                            Subleading_Photon_MVA[0] = ch_0.Subleading_Photon_MVA
                            #looser_lep0_pt[0] = ch_0.allElectrons_0_pt
                            skimmed_tree.Fill()
                        else: continue
                    #==================================
                    nentries = skimmed_tree.GetEntries()
                    # Check ttree is loaded properly and values in ttree are good
                    print "====== Check input values from tree ======"
                    print "Skimmed tree entries: ", nentries
                    for tentry in xrange(1,3):
                        skimmed_tree.GetEntry(tentry)
                        print "Test entry: ", tentry
                        print "njets: ", skimmed_tree.N_goodJets
                        print "weight: ", skimmed_tree.weight
                    print "=========================================="
                    print 'convert tree2array'
                    chunk_arr = tree2array(tree=skimmed_tree)
                    '''
                    chunk_arr = tree2array(tree=ch_0, selection=criteria)
                    chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
                    chunk_df['key']=key
                    chunk_df['target']=target
                    chunk_df['weight']=chunk_df["weight"]
                    # =============== Weights ==================
                    # WARNING! 'sample_weight' will overide 'class_weight'
                    # ==========================================
                    if sampleNames=='HH':
                        # Reweight classes
                        chunk_df['classbalance'] = 1.0
                    if sampleNames=='bckg':
                        #chunk_df['classbalance'] = 13243.0/6306.0
                        chunk_df['classbalance'] = 13243.0/194.0
                    data = data.append(chunk_df, ignore_index=True)
                else:
                    print "TTree == None"
                ch_0.Delete()
            print 'Closing File'
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

def baseline_model(num_variables,optimizer='Adam',learn_rate=0.001,nClasses=1):
    model = Sequential()
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    '''model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(0.01))'''
    model.add(Dense(16,activation='relu'))
    '''model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dropout(0.01))'''
    model.add(Dense(8,activation='relu'))
    '''model.add(Dense(8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dropout(0.01))'''
    model.add(Dense(4,activation='relu'))
    '''model.add(Dense(4,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(4,activation='relu'))'''
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss=[binary_focal_loss(alpha=0.25, gamma=2)],optimizer=Adam(lr=0.005),metrics=['acc'])
    if optimizer=='Adam':
        model.compile(loss='binary_crossentropy',optimizer=Adam(lr=learn_rate),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adamax':
        model.compile(loss='binary_crossentropy',optimizer=Adamax(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adadelta':
        model.compile(loss='binary_crossentropy',optimizer=Adadelta(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adagrad':
        model.compile(loss='binary_crossentropy',optimizer=Adagrad(lr=learn_rate),metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print 'mkdir: ', dir
        os.makedirs(dir)

def main():
    print 'Using Keras version: ', keras.__version__

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=0, type=int)
    parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
    args = parser.parse_args()
    do_model_fit = args.train_model
    suffix = args.suffix

    # Create instance of output directory where all results are saved.
    output_directory = 'HHWWyyDNN_binary%s/' % (suffix)

    check_dir(output_directory)

    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')

    #input_var_jsonFile = open('input_variables_new.json','r')
    input_var_jsonFile = open('input_variables.json','r')
    selection_criteria = '(AtLeast4GoodJets0Lep==1)'
    # selection_criteria = '(passPhotonSels==1 && passbVeto==1 && ExOneLep==1 && goodJets==1)'

    # Load Variables from .json
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Create list of headers for dataset .csv
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('weight')
    column_headers.append('target')
    column_headers.append('key')
    column_headers.append('classbalance')

    # Create instance of the input files directory
    #inputs_file_path = '/afs/cern.ch/work/a/atishelm/public/ForJosh/2017_DataMC_ntuples_moreVars'
    inputs_file_path = '/eos/user/r/rasharma/post_doc_ihep/double-higgs/ntuples/September29/MVANtuples'

    # Load ttree into .csv including all variables listed in column_headers
    print '<train-DNN> Input file path: ', inputs_file_path
    outputdataframe_name = '%s/output_dataframe.csv' %(output_directory)
    if os.path.isfile(outputdataframe_name):
        data = pandas.read_csv(outputdataframe_name)
        print '<train-DNN> Loading data .csv from: %s . . . . ' % (outputdataframe_name)
    else:
        print '<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path)
        data = load_data(inputs_file_path,column_headers,selection_criteria)
        data.to_csv(outputdataframe_name, index=False)
        data = pandas.read_csv(outputdataframe_name)

    # Change sentinal value to speed up training.
    data = data.replace(to_replace=-999.000000,value=-9.0)

    '''
    for col in column_headers:
        print 'data col ', col
        print data[col]
    '''
    print '<main> data columns: ', (data.columns.values.tolist())
    n = len(data)
    nHH = len(data.ix[data.target.values == 1])
    nbckg = len(data.ix[data.target.values == 0])
    print "Total length of HH = %i, bckg = %i" % (nHH, nbckg)

    # Make instance of plotter tool
    Plotter = plotter()

    # Create statistically independant lists train/test data (used to train/evaluate the network)
    traindataset, valdataset = train_test_split(data, test_size=0.1)
    valdataset.to_csv((output_directory+'valid_dataset.csv'), index=False)

    print '<train-DNN> Training dataset shape: ', traindataset.shape
    print '<train-DNN> Validation dataset shape: ', valdataset.shape

    # Remove last two columns (Event weight and xsrw) from column headers
    print '<train-DNN> Data frame column headers: ', column_headers
    training_columns = column_headers[:-4]
    print '<train-DNN> Training features: ', training_columns
    num_variables = len(training_columns)

    # Select data from columns under the remaining column headers in traindataset
    X_train = traindataset[training_columns].values
    X_test = valdataset[training_columns].values
    # Select data from 'target' as target for MVA
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
    trainingweights = traindataset.loc[:,'classbalance']#*traindataset.loc[:,'weight']
    trainingweights = np.array(trainingweights)

    # Temp hack to be able to change class weights without remaking dataframe
    #for inde in xrange(len(trainingweights)):
    #    newweight = 13243.0/6306.0
    #    trainingweights[inde]= newweight
    #print 'training event weight = ', trainingweights[0]

    # Event weights calculation so we can correctly apply event weights to diagnostic plots.
    # use seperate list because we don't want to apply class weights in plots.
    train_weights = traindataset['weight'].values
    test_weights = valdataset['weight'].values

    # Fit label encoder to Y_train
    newencoder = LabelEncoder()
    newencoder.fit(Y_train)
    # Transform to encoded array
    encoded_Y = newencoder.transform(Y_train)
    encoded_Y_test = newencoder.transform(Y_test)

    optimizer = 'Adam'#'Nadam'
    if do_model_fit == 1:
        histories = []
        labels = []
        # Define model and early stopping
        early_stopping_monitor = EarlyStopping(patience=51,monitor='val_loss',verbose=1)
        model3 = baseline_model(num_variables,optimizer='Nadam',learn_rate=0.005)

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
        class_weights = np.array(class_weight.compute_class_weight('balanced',np.unique(Y_train),Y_train))
        history3 = model3.fit(X_train,Y_train,validation_split=0.1,epochs=500,batch_size=100,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor])
        histories.append(history3)
        labels.append(optimizer)

        # Make plot of loss function evolution
        Plotter.plot_training_progress_acc(histories, labels)
        acc_progress_filename = 'DNN_acc_wrt_epoch.png'
        Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
        # Which model do you want the rest of the plots for?
        model = model3
    else:
        # Which model do you want to load?
        model_name = os.path.join(output_directory,'model.h5')
        print '<train-DNN> Loaded Model: %s' % (model_name)
        model = load_trained_model(model_name,num_variables,optimizer,1)

        '''continuetraining=1
        if continuetraining == 1:
            new_model = load_model(model_name)
            assert_allclose(new_model.predict(X_train),new_model.predict(X_train),1e-5)
            checkpoint = ModelCheckpoint(model_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
            callbacks_list = [checkpoint]
            history3 = new_model.fit(X_train,Y_train,validation_split=0.2,epochs=50,batch_size=1500,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=callbacks_list)'''


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
    print 'result_probs: '
    print result_probs
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
