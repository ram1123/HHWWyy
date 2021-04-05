# The purpose of this module is to plot DNN outputs using an existing training output, eliminating the need to re-train, and study a frozen model 

######################################################################################################################################################################
# Example Commands: 
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh 
# source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc7-opt/setup.sh
# python plot-DNN.py -s DryRunCorrelation --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass
# python plot-DNN.py -s 50Epochs-Multiclass --Website /eos/user/a/atishelm/www/HHWWgg/DNN/ --MultiClass 
# python plot-DNN.py -s FastCheck-Multiclass --MultiClass --Website /eos/user/a/atishelm/www/HHWWgg/DNN/
######################################################################################################################################################################

##-- Helpful Links:
# http://alexlenail.me/NN-SVG/index.html
# https://medium.com/hackernoon/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
# https://github.com/Tony607/ROC-Keras/blob/master/ROC-Keras.ipynb
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
# https://github.com/Wilsker/ttH_multilepton/blob/master/keras-DNN/train-DNN.py
# https://stackoverflow.com/questions/47843265/how-can-i-get-the-a-keras-models-history-after-loading-it-from-a-file-in-python

from tensorflow import keras 
import argparse
import pickle 
import os 
import pandas as pd 
import h5py 
from keras.utils import plot_model
from plotting.plotter import plotter

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--suff', dest='suffix', help='Option to choose suffix for training', default='', type=str)
parser.add_argument("--MultiClass",action="store_true",help="Plot multiclassifier ROC curves")
parser.add_argument("--Website", type = str, default = "", help = "Website path for output files")
args = parser.parse_args()

suffix = args.suffix

weights='BalanceYields'# 'BalanceYields' or 'BalanceNonWeighted'
output_directory = '%sHHWWyyDNN_%s_%s/' % (args.Website,suffix,weights) 
plots_dir = os.path.join(output_directory,'plots/')

##-- Load training outputs 
print("[plot-DNN.py] - Loading outputs from pickle files")
objectsToLoad = ["X_test","Y_test","X_train","Y_train","train_df", "labels"]
for objToLoad in objectsToLoad:
    print("Loading %s..."%(objToLoad))
    executeLine = "%s = pickle.load( open( '%s/%s.p', 'rb' ) )"%(objToLoad, output_directory, objToLoad) 
    # executeLine = "%s = pickle.load( open( '%s/%s.p', 'rb' ), encoding='latin1' )"%(objToLoad, output_directory, objToLoad) 
    exec(executeLine) ##-- encoding='latin1' to avoid python 2 / 3 numpy array incompatibility: https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3

modelFilename = "%s/model.h5"%(output_directory)

print("Loading model...")
model = keras.models.load_model(modelFilename)

history = pd.read_csv("%s/training.log"%(output_directory), sep=',', engine='python')

model.summary()
model_schematic_name = os.path.join(output_directory,'model_schematic_Horizontal.png')
plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True, rankdir = "LR")
model_schematic_name = os.path.join(output_directory,'model_schematic_Horizontal.pdf')
plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True, rankdir = "LR")  

# Make instance of plotter tool
Plotter = plotter(args.Website)

## Input Variable Correlation plot
withHiggsMass = 0 ##-- Plot correlation matrix expecting CMS_hgg_mass (diphoton mass - with purpose of checking correlation between input variables and signal region variable)
correlation_plot_file_name = 'correlation_plot'
Plotter.correlation_matrix(train_df, withHiggsMass)
Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.png')
Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name+'.pdf')

# Make plot of loss function evolution
Plotter.plot_training_progress_acc(histories, labels)
acc_progress_filename = 'DNN_acc_wrt_epoch'
Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename+'.png')
Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename+'.pdf') 

from_log = 1 ##-- obtaining history information from log file 
Plotter.history_plot(history, from_log, label='loss')
Plotter.save_plots(dir=plots_dir, filename='history_loss.png')
Plotter.save_plots(dir=plots_dir, filename='history_loss.pdf')  

Plotter.history_plot(history, from_log, label='acc')
Plotter.save_plots(dir=plots_dir, filename='history_acc.png')
Plotter.save_plots(dir=plots_dir, filename='history_acc.pdf')  

# Initialise output directory.
Plotter.plots_directory = plots_dir
Plotter.output_directory = output_directory

if(args.MultiClass):
    Plotter.ROC_MultiClassifier(model, X_test, Y_test, X_train, Y_train)
else: 
    Plotter.ROC(model, X_test, Y_test, X_train, Y_train)
    Plotter.save_plots(dir=plots_dir, filename='ROC.png')
    Plotter.save_plots(dir=plots_dir, filename='ROC.pdf')