from apply_DNN import apply_DNN
import matplotlib.pyplot as plt
import numpy as np
import numpy
import pandas
import pandas as pd
import optparse, json, argparse, subprocess
import ROOT
import sys
from keras.models import load_model
from array import array
sys.path.insert(0, '/afs/cern.ch/work/j/jthomasw/private/IHEP/ttHML/github/ttH_multilepton/keras-DNN/')
from plotting.plotter import plotter
from ROOT import TFile, TTree, gDirectory, gPad
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import os
from os import environ

def main():
    print('')
    DNN_applier = apply_DNN()

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)

    parser.add_argument('-p', '--processName', dest='processName', help='Process name. List of options in keys of process_filename dictionary', default=[], type=str, nargs='+')
    parser.add_argument('-d', '--modeldir', dest='modeldir', help='Option to choose directory containing trained model')

    args = parser.parse_args()
    processes = args.processName
    nClasses = 1
    modeldir=args.modeldir
    print('<run_network_evaluation> Succesfully parsed arguments: processName= [%s], model directory= %s' %(processes, modeldir))

    input_var_jsonFile = ''

    # Open and load input variable .json
    input_var_jsonFile = open('../input_variables.json','r')
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Append variables to a list of column headers for .csv file later
    column_headers = []
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('event')
    column_headers.append('weight')

    # Dictionary of filenames to be run over along with their keys.
    process_filename = {
    'HHWWgg' : ('HHWWgg-SL-SM-NLO-2017'),
    'DiPhoton' : ('DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa_Hadded'),
    'GJet_Pt-20toInf' : ('GJet_Pt-20toInf_DoubleEMEnriched_MGG-40to80_TuneCP5_13TeV_Pythia8_Hadded'),
    'GJet_Pt-20to40' : ('GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8_Hadded'),
    'GJet_Pt-40toInf' : ('GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8_Hadded') ,
    'DYJetsToLL_M-50' : ('DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_Hadded'),
    'TTGJets' : ('TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_Hadded'),
    'TTGG' : ('TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_Hadded'),
    'TTJets_HT-600to800' : ('TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'TTJets_HT-800to1200' : ('TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'TTJets_HT-1200to2500' : ('TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'TTJets_HT-2500toInf' : ('TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'W1JetsToLNu_LHEWpT_0-50' : ('W1JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W1JetsToLNu_LHEWpT_50-150' : ('W1JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W1JetsToLNu_LHEWpT_150-250' : ('W1JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W1JetsToLNu_LHEWpT_250-400' : ('W1JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W1JetsToLNu_LHEWpT_400-inf' : ('W1JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W2JetsToLNu_LHEWpT_0-50' : ('W2JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W2JetsToLNu_LHEWpT_50-150' : ('W2JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W2JetsToLNu_LHEWpT_150-250' : ('W2JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W2JetsToLNu_LHEWpT_250-400' : ('W2JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W2JetsToLNu_LHEWpT_400-inf' : ('W2JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded'),
    'W3JetsToLNu' : ('W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'W4JetsToLNu' : ('W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded'),
    'ttHJetToGG' : ('ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_Hadded')
    #'Data' : ('Data_'+JESname+region)
    }

    training_columns = column_headers[:-2]
    num_variables = len(training_columns)

    # Load trained model
    model_name_1 = os.path.join('../',modeldir,'model.h5')
    print('<run_network_evaluation> Using Model: ', model_name_1)
    model_1 = load_model(model_name_1, compile=False)
    # Make instance of plotter class
    Plotter = plotter()

    # Lists for all events in all files. Used to make diagnostic plots of networks performance over all samples.
    true_process = []
    model1_probs_ = []
    EventWeights_ = []

    succesfully_run_files = open("succesfully_run_files.txt","w+")
    # Now loop over all samples
    for process in processes:
        print('<run_network_evaluation> Process: ', process)
        current_sample_name = process_filename.get(process)
        inputs_file_path = '/Users/joshuhathomas-wilsker/Documents/work/lxplus_remote/work/private/IHEP/HH/HHWWyy/HHWWgg_DataSignalMCnTuples/2017/'
        if 'HHWWgg' in process:
            inputs_file_path += 'Signal/'
        else:
            inputs_file_path += 'Bkgs/'

        print('<run_network_evaluation> Input file directory: ', inputs_file_path)

        # Make final output directory
        samples_dir_w_appended_DNN = 'samples_w_DNN'
        if not os.path.exists(samples_dir_w_appended_DNN):
            os.makedirs(samples_dir_w_appended_DNN)
        samples_final_path_dir = os.path.join(samples_dir_w_appended_DNN,modeldir)
        if not os.path.exists(samples_final_path_dir):
            os.makedirs(samples_final_path_dir)

        dataframe_name = '%s/%s_dataframe.csv' %(samples_final_path_dir,process)
        if os.path.isfile(dataframe_name):
            print('<run_network_evaluation> Loading %s . . . . ' % dataframe_name)
            data = pandas.read_csv(dataframe_name)
        else:
            print('<run_network_evaluation> Making *new* data file from %s . . . . ' % (inputs_file_path))
            selection_criteria = '( ( (Leading_Photon_pt/CMS_hgg_mass) > 0.35 ) && ( (Subleading_Photon_pt/CMS_hgg_mass) > 0.25 ) && passbVeto==1 && ExOneLep==1 && N_goodJets>=1 )'
            data = DNN_applier.load_data(inputs_file_path,column_headers,selection_criteria,current_sample_name)
            if len(data) == 0 :
                print('<run_network_evaluation> No data! Next file.')
                continue
            print('<run_network_evaluation> Saving new data .csv file at %s . . . . ' % (dataframe_name))
            print('<run_network_evaluation> Found events passing selection. Process name will be stored in succesfully_run_files.txt')
            succesfully_run_files.write(process)

            data = data.replace(to_replace=-999.000000,value=-9.0)
            data.to_csv(dataframe_name, index=False)

        nHH = len(data.iloc[data.target.values == 1])
        nbckg = len(data.iloc[data.target.values == 0])

        print("<run_network_evaluation> Total length of HH = %i, bckg = %i" % (nHH, nbckg))

        # Create dataset from dataframe to evaluate DNN
        X_test = data[training_columns].values
        result_probs_ = model_1.predict(np.array(X_test))
        nEvent = data['event']

        if len(result_probs_) < 1.:
            print('<run_network_evaluation> Warning: only %s test values.' % (len(result_probs_)))
            print('<run_network_evaluation> Probabilities: ', result_probs_)
            print('<run_network_evaluation> Exiting now.')
            exit(0)

        # Dictionary:
        # key = event number : value = DNN output
        eventnum_resultsprob_dict = {}
        for index in range(len(nEvent)):
            #print('nEvent= %s , prob = %s' % (nEvent[index], result_probs_[index])
            eventnum_resultsprob_dict[nEvent[index]] = result_probs_[index]
            model1_probs_.append(result_probs_[index])

        print(current_sample_name)
        infile = inputs_file_path+current_sample_name+".root"
        print('<run_network_evaluation> Input file: ', infile)

        # Open file and load ttrees
        data_file = TFile.Open(infile)
        if 'HHWWgg' in current_sample_name:
            treename=['GluGluToHHTo2G2Qlnu_node_cHHH1_TuneCP5_PSWeights_13TeV_powheg_pythia8alesauva_2017_1_10_6_4_v0_RunIIFall17MiniAODv2_PU2017_12Apr2018_94X_mc2017_realistic_v14_v1_1c4bfc6d0b8215cc31448570160b99fdUSER']
        elif 'DiPhotonJetsBox_MGG' in current_sample_name:
            treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa']
        elif 'GJet_Pt-20toInf' in current_sample_name:
            treename = [
            'GJet_Pt_20toInf_DoubleEMEnriched_MGG_40to80_TuneCP5_13TeV_Pythia8'
            ]
        elif 'GJet_Pt-20to40' in current_sample_name:
            treename = [
            'GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8'
            ]
        elif 'GJet_Pt-40toInf' in current_sample_name:
            treename=['GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8'
            ]
        elif 'DYJetsToLL_M-50_TuneCP5' in current_sample_name:
            treename=['DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8'
            ]
        elif 'TTGG' in current_sample_name:
            treename=['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8'
            ]
        elif 'TTGJets' in current_sample_name:
            treename=['TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8'
            ]
        elif 'TTJets_HT-600to800' in current_sample_name:
            treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'TTJets_HT-800to1200' in current_sample_name:
            treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'TTJets_HT-1200to2500' in current_sample_name:
            treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'TTJets_HT-2500toInf' in current_sample_name:
            treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'W1JetsToLNu_LHEWpT_0-50' in current_sample_name:
            treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W1JetsToLNu_LHEWpT_50-150' in current_sample_name:
            treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W1JetsToLNu_LHEWpT_150-250' in current_sample_name:
            treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W1JetsToLNu_LHEWpT_250-400' in current_sample_name:
            treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W1JetsToLNu_LHEWpT_400-inf' in current_sample_name:
            treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W2JetsToLNu_LHEWpT_0-50' in current_sample_name:
            treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W2JetsToLNu_LHEWpT_50-150' in current_sample_name:
            treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W2JetsToLNu_LHEWpT_150-250' in current_sample_name:
            treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W2JetsToLNu_LHEWpT_250-400' in current_sample_name:
            treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W2JetsToLNu_LHEWpT_400-inf' in current_sample_name:
            treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8'
            ]
        elif 'W3JetsToLNu' in current_sample_name:
            treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'W4JetsToLNu' in current_sample_name:
            treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8'
            ]
        elif 'ttHJetToGG' in current_sample_name:
            treename=['ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8'
            ]
        else:
            print('<run_network_evaluation> Warning: Process name not recognised. Exiting.')
            exit(0)

        # Open each TTree in file and loop over events.
        # Append evaluated DNN score to DNN branch for each event.
        # Score assigned to event according to event number.
        for tname in treename:
            print('<run_network_evaluation> TTree: ', tname)
            data_tree = data_file.Get(tname)
            # Check if input file is zombie
            if data_file.IsZombie():
                raise IOError('missing file')
                exit(0)

            output_file_name = '%s/%s.root' % (samples_final_path_dir,process_filename.get(process))
            print('<run_network_evaluation> Creating new output .root file')
            output_file = TFile.Open(output_file_name,'RECREATE')

            # Clone empty tree
            output_tree = data_tree.CloneTree(0)
            output_tree.SetName("output_tree")

            # All branches on.
            # Turn off all branches except those needed to speed up run-time
            output_tree.SetBranchStatus('*',1)

            # Append DNN Branches to new TTree
            DNN_evaluation = array('f',[0.])
            DNN_evaluation_branch = output_tree.Branch('DNN_evaluation', DNN_evaluation, 'DNN_evaluation/F')

            sample_name = process

            histo_DNN_values_title = 'DNN values: %s Sample' % (sample_name)
            histo_DNN_values_name = 'histo_DNN_values_%s_sample' % (sample_name)
            histo_DNN_values = ROOT.TH1D(histo_DNN_values_name,histo_DNN_values_title,200,0,1.)

            temp_percentage_done = 0

            ######## Loop over ttree #########
            print('<run_network_evaluation> data_tree # Entries: ', data_tree.GetEntries())
            if output_tree.GetEntries() != 0:
                print('<run_network_evaluation> output_tree # Entries: ', output_tree.GetEntries())
                print('This tree should be empty at this point!!!!! check cloning correctly')

            for i in range(data_tree.GetEntries()):
                DNN_evaluation[0]= -1.

                percentage_done = int(100*float(i)/float(data_tree.GetEntries()))
                if percentage_done % 10 == 0:
                    if percentage_done != temp_percentage_done:
                        print(percentage_done)
                        temp_percentage_done = percentage_done
                data_tree.GetEntry(i)

                Eventnum_ = array('d',[0])
                Eventnum_ = data_tree.event
                EventWeight_ = array('d',[0])
                EventWeight_ = data_tree.weight
                passbVeto  = array('d',[0])
                passbVeto = data_tree.passbVeto
                ExOneLep  = array('d',[0])
                ExOneLep = data_tree.ExOneLep
                Leading_Photon_pt = array('d',[0])
                Leading_Photon_pt = data_tree.Leading_Photon_pt
                Subleading_Photon_pt = array('d',[0])
                Subleading_Photon_pt = data_tree.Subleading_Photon_pt
                CMS_hgg_mass = array('d',[0])
                CMS_hgg_mass = data_tree.CMS_hgg_mass
                N_goodJets = array('d',[0])
                N_goodJets = data_tree.N_goodJets

                if ( (Leading_Photon_pt/CMS_hgg_mass)>0.35 and (Subleading_Photon_pt/CMS_hgg_mass)>0.25 and passbVeto==1 and ExOneLep==1 and N_goodJets>=1):
                    pass_selection = 1
                else:
                    pass_selection = 0

                if pass_selection==0:
                        continue

                if 'HHWWgg' in process:
                    true_process.append(1)
                else:
                    true_process.append(0)

                EventWeights_.append(EventWeight_)
                histo_DNN_values.Fill(eventnum_resultsprob_dict.get(Eventnum_)[0] , EventWeight_)
                DNN_evaluation[0] = eventnum_resultsprob_dict.get(Eventnum_)[0]
                output_tree.Fill()

        eventnum_resultsprob_dict.clear()
        output_file.Write()
        output_file.Close()
        data_file.Close()

    #plots_dir = os.path.join(samples_final_path_dir,'plots/')
    #Plotter.plots_directory = plots_dir
    #model1_probs_ = np.array(model1_probs_)
    #Plotter.ROC_sklearn(true_process, model1_probs_, true_process, model1_probs_, 0 , 'ttHnode')

    exit(0)

main()
