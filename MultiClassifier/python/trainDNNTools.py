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
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model

def baseline_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    #model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
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

def new_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(32, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation="sigmoid")) 
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def MultiClassifier_Model(num_variables, nClasses, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(32,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Dense(16,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(8,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification 
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy 
    
    return model

def GetFileInfo(filen):

    if('GluGluToHHTo2G2Qlnu_node' in filen):
        node = filen.split('_')[2]
        treename = ['GluGluToHHTo2G2Qlnu_node_%s_13TeV_HHWWggTag_0_v1'%(node)]
        process_ID = 'HH'

    # if 'GluGluToHHTo2G2Qlnu_node_cHHH1_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_cHHH1_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'
    # elif 'GluGluToHHTo2G2Qlnu_node_10_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_10_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH' 
    # elif 'GluGluToHHTo2G2Qlnu_node_11_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_11_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_1_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_1_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_12_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_12_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_2_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_2_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_3_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_3_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH' 
    # elif 'GluGluToHHTo2G2Qlnu_node_4_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_4_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH' 
    # elif 'GluGluToHHTo2G2Qlnu_node_5_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_5_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'      
    # elif 'GluGluToHHTo2G2Qlnu_node_6_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_6_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH' 
    # elif 'GluGluToHHTo2G2Qlnu_node_7_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_7_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'      
    # elif 'GluGluToHHTo2G2Qlnu_node_8_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_8_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_9_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_9_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'  
    # elif 'GluGluToHHTo2G2Qlnu_node_SM_2017' in filen:
    #     treename=['GluGluToHHTo2G2Qlnu_node_SM_13TeV_HHWWggTag_0_v1']
    #     process_ID = 'HH'     

    elif 'GluGluHToGG' in filen:
        treename=['ggh_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'GluGluHToGG'
    elif 'VBFHToGG' in filen:
        treename=['vbf_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'VBFHToGG'
    elif 'VHToGG' in filen:
        treename=['wzh_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'VHToGG'  
    elif 'ttHJetToGG' in filen:
        treename=['tth_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'ttHJetToGG'
    elif 'DiPhotonJetsBox_M40_80' in filen:
        treename=['DiPhotonJetsBox_M40_80_Sherpa_13TeV_HHWWggTag_0',
        ]
        process_ID = 'DiPhoton'
    elif 'DiPhotonJetsBox_MGG-80toInf' in filen:
        treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_0',
        ]
        process_ID = 'DiPhoton'
    elif 'GJet_Pt-20to40' in filen:
        treename=['GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'GJet_Pt-20toInf' in filen:
        treename=['GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'GJet_Pt-40toInf' in filen:
        treename=['GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'QCD_Pt-30to40' in filen:
        treename=['QCD_Pt_30to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'QCD'
    elif 'QCD_Pt-30toInf' in filen:
        treename=['QCD_Pt_30toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'QCD'
    elif 'QCD_Pt-40toInf' in filen:
        treename=['QCD_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'DY'
    elif 'DYJetsToLL_M-50' in filen:
        treename=['DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTGG_0Jets' in filen:
        treename=['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTGJets_TuneCP5' in filen:
        treename=['TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-600to800' in filen:
        treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-800to1200' in filen:
        treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-1200to2500' in filen:
        treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-2500toInf' in filen:
        treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'ttWJets' in filen:
        treename=['ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_TuneCP5_extra' in filen:
        treename=['TTJets_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_0-50' in filen:
        treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_50-150' in filen:
        treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_150-250' in filen:
        treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_250-400' in filen:
        treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_400-inf' in filen:
        treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_0-50' in filen:
        treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_50-150' in filen:
        treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_150-250' in filen:
        treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_250-400' in filen:
        treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_400-inf' in filen:
        treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W3JetsToLNu' in filen:
        treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W4JetsToLNu' in filen:
        treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGGJets' in filen:
        treename=['WGGJets_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGJJToLNuGJJ_EWK' in filen:
        treename=['WGJJToLNuGJJ_EWK_aQGC_FS_FM_TuneCP5_13TeV_madgraph_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGJJToLNu_EWK_QCD' in filen:
        treename=['WGJJToLNu_EWK_QCD_TuneCP5_13TeV_madgraph_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets' 
    elif 'WWTo1L1Nu2Q' in filen:
        treename=['WWTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WW'
    elif 'WW_TuneCP5' in filen:
        treename=['WW_TuneCP5_13TeV_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WW'

    return treename, process_ID 