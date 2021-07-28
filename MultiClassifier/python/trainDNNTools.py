from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def load_trained_model(model_path):
    print('<load_trained_model> weights_path: ', model_path)
    model = load_model(model_path, compile=False)
    return model


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

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

def ANN_model(
                   num_variables,
                   nClasses,
                   optimizer='Nadam',
                   activation='relu',
                   loss='categorical_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    model.add(Dense(num_variables,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    # model.add(Dense(num_variables,kernel_initializer=init_mode,activation=activation))
    model.add(Dense(nClasses, activation='softmax'))
    if optimizer=='Adam':
        model.compile(loss=loss,optimizer=Adam(lr=learn_rate),metrics=['acc'])
    if optimizer=='Nadam':
        model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adamax':
        model.compile(loss=loss,optimizer=Adamax(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adadelta':
        model.compile(loss=loss,optimizer=Adadelta(lr=learn_rate),metrics=['acc'])
    if optimizer=='Adagrad':
        model.compile(loss=loss,optimizer=Adagrad(lr=learn_rate),metrics=['acc'])
    return model

def ANN_model2(
                   num_variables,
                   nClasses,
                   optimizer='Nadam',
                   activation='relu',
                   loss='categorical_crossentropy',
                   dropout_rate=0.2,
                   init_mode='glorot_normal',
                   learn_rate=0.001,
                   metrics=METRICS
                   ):
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    model = Sequential()
    model.add(Dense(num_variables,input_dim=num_variables,kernel_initializer=init_mode,activation=activation))
    model.add(Dense(30,activation='relu'))
    model.add(Dense(15,activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    model.compile(loss=loss,optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    return model

def MultiClassifier_Modelv1(num_variables, nClasses, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def MultiClassifier_Modelv2(num_variables, nClasses, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def MultiClassifier_Modelv3(num_variables, nClasses, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def MultiClassifier_Modelv4(num_variables, nClasses, learn_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model


def MultiClassifier_ModelVarLayerV1(num_variables, nClasses, learn_rate=0.001, nlayers = 1):
    model = Sequential()
    neuronsHiddenLayer = []
    neuronsInputLayer = num_variables
    for x in range(0,20):
        if ((((neuronsInputLayer+1)*2)/3) <= nClasses): break
        neuronsHiddenLayer.append((((neuronsInputLayer+1)*2)/3))
        neuronsInputLayer = neuronsHiddenLayer[x]
    model.add(Dense(neuronsHiddenLayer[0], input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for x in range(1,nlayers):
        model.add(Dense(neuronsHiddenLayer[x],activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def MultiClassifier_ModelVarLayerV2(num_variables, nClasses, learn_rate=0.001, nlayers = 1):
    model = Sequential()
    neuronsHiddenLayer = []
    neuronsInputLayer = 256
    for x in range(0,20):
        if ((((neuronsInputLayer+1)*2)/3) <= nClasses): break
        neuronsHiddenLayer.append((((neuronsInputLayer+1)*2)/3))
        neuronsInputLayer = neuronsHiddenLayer[x]
    model.add(Dense(neuronsHiddenLayer[0], input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    for x in range(1,nlayers):
        model.add(Dense(neuronsHiddenLayer[x],activation='relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def MultiClassifier_ModelVarLayerV3(num_variables, nClasses, learn_rate=0.001, nlayers = 1):
    model = Sequential()
    neuronsHiddenLayer = []
    neuronsInputLayer = 256
    for x in range(0,20):
        if ((((neuronsInputLayer+1)*2)/3) <= nClasses): break
        neuronsHiddenLayer.append((((neuronsInputLayer+1)*2)/3))
        neuronsInputLayer = neuronsHiddenLayer[x]
    model.add(Dense(neuronsHiddenLayer[0], input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    for x in range(1,nlayers):
        model.add(Dense(neuronsHiddenLayer[x],kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def new_model5(
               num_variables,
               nClasses,
               optimizer='Nadam',
               activation='relu',
               loss='categorical_crossentropy',
               dropout_rate=0.1,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(256, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    # model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    model.compile(loss=loss,optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def new_model6(
               num_variables,
               nClasses,
               optimizer='Nadam',
               activation='relu',
               loss='categorical_crossentropy',
               dropout_rate=0.1,
               init_mode='glorot_normal',
               learn_rate=0.001,
               metrics=METRICS
               ):
    model = Sequential()
    model.add(Dense(256, input_dim=num_variables,kernel_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128))
    # model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64))
    # model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(nClasses, activation='softmax')) ##-- softmax for mutually exclusive classification
    optimizer=Nadam(lr=learn_rate)
    # model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    model.compile(loss=loss,optimizer=optimizer,metrics=['acc']) ##--  Categorical instead of binary crossentropy
    return model

def GetFileInfo(filen, channel):

    if('GluGluToHHTo2G2Qlnu_node' in filen and channel == "SL"):
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

    elif 'GluGluHToGG' in filen and channel == "SL":
        treename=['ggh_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'GluGluHToGG'
    elif 'VBFHToGG' in filen and channel == "SL":
        treename=['vbf_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'VBFHToGG'
    elif 'VHToGG' in filen and channel == "SL":
        treename=['wzh_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'VHToGG'
    elif 'ttHJetToGG' in filen and channel == "SL":
        treename=['tth_125_13TeV_HHWWggTag_0_v1']
        process_ID = 'ttHJetToGG'
    elif 'DiPhotonJetsBox_M40_80' in filen and channel == "SL":
        treename=['DiPhotonJetsBox_M40_80_Sherpa_13TeV_HHWWggTag_0',
        ]
        process_ID = 'DiPhoton'
    elif 'DiPhotonJetsBox_MGG-80toInf' in filen and channel == "SL":
        treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_0',
        ]
        process_ID = 'DiPhoton'
    elif 'GJet_Pt-20to40' in filen and channel == "SL":
        treename=['GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'GJet_Pt-20toInf' in filen and channel == "SL":
        treename=['GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'GJet_Pt-40toInf' in filen and channel == "SL":
        treename=['GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'GJet'
    elif 'QCD_Pt-30to40' in filen and channel == "SL":
        treename=['QCD_Pt_30to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'QCD'
    elif 'QCD_Pt-30toInf' in filen and channel == "SL":
        treename=['QCD_Pt_30toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'QCD'
    elif 'QCD_Pt-40toInf' in filen and channel == "SL":
        treename=['QCD_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'QCD'
    elif 'DYJetsToLL_M-50' in filen and channel == "SL":
        treename=['DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTGG_0Jets' in filen and channel == "SL":
        treename=['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTGJets_TuneCP5' in filen and channel == "SL":
        treename=['TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-600to800' in filen and channel == "SL":
        treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-800to1200' in filen and channel == "SL":
        treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-1200to2500' in filen and channel == "SL":
        treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_HT-2500toInf' in filen and channel == "SL":
        treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'ttWJets' in filen and channel == "SL":
        treename=['ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'TTGsJets'
    elif 'TTJets_TuneCP5_extra' in filen and channel == "SL":
        treename=['TTJets_TuneCP5_13TeV_amcatnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_0-50' in filen and channel == "SL":
        treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_50-150' in filen and channel == "SL":
        treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_150-250' in filen and channel == "SL":
        treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_250-400' in filen and channel == "SL":
        treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W1JetsToLNu_LHEWpT_400-inf' in filen and channel == "SL":
        treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_0-50' in filen and channel == "SL":
        treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_50-150' in filen and channel == "SL":
        treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_150-250' in filen and channel == "SL":
        treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_250-400' in filen and channel == "SL":
        treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W2JetsToLNu_LHEWpT_400-inf' in filen and channel == "SL":
        treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W3JetsToLNu' in filen and channel == "SL":
        treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'W4JetsToLNu' in filen and channel == "SL":
        treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGGJets' in filen and channel == "SL":
        treename=['WGGJets_TuneCP5_13TeV_madgraphMLM_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGJJToLNuGJJ_EWK' in filen and channel == "SL":
        treename=['WGJJToLNuGJJ_EWK_aQGC_FS_FM_TuneCP5_13TeV_madgraph_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WGJJToLNu_EWK_QCD' in filen and channel == "SL":
        treename=['WGJJToLNu_EWK_QCD_TuneCP5_13TeV_madgraph_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WGsJets'
    elif 'WWTo1L1Nu2Q' in filen and channel == "SL":
        treename=['WWTo1L1Nu2Q_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WW'
    elif 'WW_TuneCP5' in filen and channel == "SL":
        treename=['WW_TuneCP5_13TeV_pythia8_13TeV_HHWWggTag_0',
        ]
        process_ID = 'WW'
    #====================================================================#
    ##                                                                  ##
    ##                  Fully Hadronic mapping....                      ##
    ##                                                                  ##
    #====================================================================#
    # elif 'GluGluToHHTo2G4Q' in filen and channel == "FH":
    #     treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_cHHH1_13TeV_HHWWggTag_1']
    #     process_ID = 'HH'
    #     elif ''
    elif 'GluGluToHHTo2G4Q_node_cHHH1_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_cHHH1_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G2ZTo2G4Q_node_cHHH1_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G2ZTo2G4Q_node_cHHH1_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_1_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_1_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_2_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_2_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_3_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_3_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_4_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_4_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_5_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_5_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_6_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_6_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_7_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_7_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_8_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_8_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_9_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_9_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_10_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_10_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_11_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_11_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_12_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_12_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_SM_2017' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_SM_13TeV_HHWWggTag_1']
        process_ID = 'HH'
    elif 'GluGluToHHTo2G4Q_node_cHHH1_2018' in filen:
        treename=['tagsDumper/trees/GluGluToHHTo2G4Q_node_cHHH1_13TeV_HHWWggTag_1']
        process_ID = 'HH'

    elif 'datadriven' in filen and channel == "FH":
        treename=['tagsDumper/trees/Data_13TeV_HHWWggTag_1']
        process_ID = 'QCD'
    elif 'GluGluHToGG' in filen and channel == "FH":
        treename=['tagsDumper/trees/ggh_125_13TeV_HHWWggTag_1']
        process_ID = 'GluGluHToGG'
    elif 'VBFHToGG' in filen and channel == "FH":
        treename=['tagsDumper/trees/vbf_125_13TeV_HHWWggTag_1']
        process_ID = 'VBFHToGG'
    elif 'VHToGG' in filen and channel == "FH":
        treename=['tagsDumper/trees/wzh_125_13TeV_HHWWggTag_1']
        process_ID = 'VHToGG'
    elif 'ttHJetToGG' in filen and channel == "FH":
        treename=['tagsDumper/trees/tth_125_13TeV_HHWWggTag_1']
        process_ID = 'ttHJetToGG'
    elif 'DiPhotonJetsBox_MGG-80toInf' in filen and channel == "FH":
        treename=['tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa_13TeV_HHWWggTag_1']
        process_ID = 'DiPhoton'
    elif 'TTGG_0Jets' in filen and channel == "FH":
        treename=['tagsDumper/trees/TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_13TeV_HHWWggTag_1']
        process_ID = 'TTGsJets'
    elif 'TTGJets_TuneCP5' in filen and channel == "FH":
        treename=['tagsDumper/trees/TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8_13TeV_HHWWggTag_1']
        process_ID = 'TTGsJets'
    elif '2B2G' in filen and channel == "FH":
        treename=['tagsDumper/trees/GluGluToHHTo2B2G_node_cHHH1_13TeV_HHWWggTag_1']
        process_ID = 'bbgg'

    return treename, process_ID
