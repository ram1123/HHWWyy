# MultiClass Deep Neural Network

The purpose of this directory is to provide the necessary files to train a Multiclassifer Deep Neural Network. The modules provided should also be able to produce a Binary DNN. 

## Setup

Begin by cloning the repository and changing the the MultiClassifer directory:


    git clone git@github.com:atishelmanch/HHWWyy.git -b MultiClassifier
    cd HHWWyy/MultiClassifier
    
You should then source the following lxplus environment:

    source /cvmfs/sft.cern.ch/lcg/views/LCG_94/x86_64-centos7-gcc7-opt/setup.sh
    
## Example: MultiClass DNN

And can try running a MultiClass DNN with default settings and input files (Note that you may need read access to the input files):

    python train-DNN.py -t 1 -s Test_MultiClass_DNN -i /eos/user/b/bmarzocc/HHWWgg/January_2021_Production/2017/ --Website /eos/user/<lxplusUsernameLetter>/<lxplusUsername>/www/DNNoutputDirectory/ --MultiClass --SaveOutput -e 3 --useKinWeight --VHToGGClassWeightFactor 2.0
    --ttHJetToGGClassWeightFactor 2.0 --BkgClassWeightFactor 1.0 --LessSamples
    
If this works properly, a MultiClass DNN will be trained with minimum samples (--LessSamples flag) from the directory specified by the -i flag. The model as well as required information in order to reproduce 
output plots such as ROC curves, loss and acc vs. epoch are output due to --SaveOutput, and output to the specified --Website. If no website is specified, the default plots and output should be output in the current directory.
