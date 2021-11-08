# -*- coding: utf-8 -*-
# @Author: Ram Krishna Sharma
# @Date:   2021-04-06 12:05:34
# @Last Modified by:   Ram Krishna Sharma
# @Last Modified time: 2021-08-03

##
## USER MODIFIED STRING
##

import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dirTag', dest='dirTag', help='name of directory tag', default="TEST_args", type=str)
parser.add_argument('-j', '--jobName', dest='jobName', help='Slurm job name', default="DNN", type=str)
parser.add_argument('-s', '--scan', dest='scan', help='do RandomizedSearchCV scan or not', default=False, type=bool)
parser.add_argument('-isTrain', '--isTrain', dest='isTrain', help='train model or not?', default=1, type=int)
parser.add_argument('-w', '--weights', dest='weights', help='weights to use', default='BalanceYields', type=str,choices=['BalanceYields','BalanceNonWeighted'])
parser.add_argument('-dlr', '--dynamic_lr', dest='dynamic_lr', help='vary learn rate with epoch', default=False, type=bool)
parser.add_argument('-lr', '--lr', dest='learnRate', help='Learn rate', default=0.1, type=float)
parser.add_argument("-e", "--epochs", type=int, default=10, help = "Number of epochs to train")
parser.add_argument("-b", "--batch_size", type=int, default=100, help = "Number of batch_size to train")
parser.add_argument("-o", "--optimizer", type=str, default="Nadam", help = "Name of optimizer to train with")
parser.add_argument("-a", "--activation", type=str, default="relu", help = "activation to be used. default is the relu")
parser.add_argument("-dropout_rate", "--dropout_rate", type=float, default=0.2, help = "dropout rate to be used. Default value is 0.2")
parser.add_argument('-cw', '--classweight', dest='classweight', help='classweight to use', default=False, type=bool)
parser.add_argument('-sw', '--sampleweight', dest='sampleweight', help='sampleweight to use', default=False, type=bool)
parser.add_argument("-nHiddenLayer", "--nHiddenLayer", type=int, default=1, help = "Number of Hidden layers")
parser.add_argument("-dropoutLayer", "--dropoutLayer", type=int, default=0, help = "If you want to include dropoutLayer with the first two hidden layers")
parser.add_argument('-json', '--json', dest='json', help='input variable json file', default='input_variables.json', type=str)
parser.add_argument('-c', dest="cutString", type=str, default="( 1.>0. )", help="cut selection to apply")
parser.add_argument("-nlayers", "--nlayers", type=int, default=1, help = "Number of hidden layers in the network")
parser.add_argument("-ModelToUse", "--ModelToUse", type=str, default="FH_ANv5", help = "Name of optimizer to train with")
parser.add_argument("-BBggsum_weightFactor", "--BBggsum_weightFactor", type = float, default = 1., help = "Factor to adjust bbgg class weights")
parser.add_argument("-ClassWeightTargetDividedby", "--ClassWeightTargetDividedby", type = float, default = 1., help = "Factor to adjust bbgg class weights")

args = parser.parse_args()

dirTag=args.dirTag
MacroPath = '/hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/'
# LogDirPath = "/hpcfs/bes/mlgpu/sharma/ML_GPU/HHWWyy/HHWWBBDNN_binary_"+dirTag+"_BalanceYields/"
LogDirPath = MacroPath + "/HHWWBBDNN_binary_"+dirTag+"_"+args.weights+"/"

# print ("args.scan: ",args.scan)
if args.scan:
  CommandToRun = "python train-BinaryDNN_WWvsBB.py -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -p 1 -g 0 -r 1"  # Scan using RandomizedSearchCV
  # CommandToRun = "python train-BinaryDNN_WWvsBB.py -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -p 1 -g 1 -r 0"  # Scan using RandomizedSearchCV
elif args.dynamic_lr:
  CommandToRun = "python train-BinaryDNN_WWvsBB.py -i /hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v5_BScoreBugFix/ -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -w "+args.weights +" -lr "+str(args.learnRate)+" -e "+str(args.epochs)+" -b "+str(args.batch_size)+" -o "+args.optimizer + " -j "+args.json + " -nHiddenLayer "+str(args.nHiddenLayer) + " -dropoutLayer "+str(args.dropoutLayer) + " -dlr "+str(args.dynamic_lr)
elif args.classweight:
  CommandToRun = "python train-BinaryDNN_WWvsBB.py -i /hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v5_BScoreBugFix/ -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -w "+args.weights +" -lr "+str(args.learnRate)+" -e "+str(args.epochs)+" -b "+str(args.batch_size)+" -o "+args.optimizer + " -j "+args.json + " -nHiddenLayer "+str(args.nHiddenLayer) + " -dropoutLayer "+str(args.dropoutLayer) + " -a " + str(args.activation) + " -d " + str(args.dropout_rate) +" -cw "+str(args.classweight)
elif args.sampleweight:
  CommandToRun = "python train-BinaryDNN_WWvsBB.py -i /hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v5_BScoreBugFix/ -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -w "+args.weights +" -lr "+str(args.learnRate)+" -e "+str(args.epochs)+" -b "+str(args.batch_size)+" -o "+args.optimizer + " -j "+args.json + " -nHiddenLayer "+str(args.nHiddenLayer) + " -dropoutLayer "+str(args.dropoutLayer) + " -a " + str(args.activation) + " -d " + str(args.dropout_rate) +" -sw "+str(args.sampleweight)
else:
  CommandToRun = "python train-BinaryDNN_WWvsBB.py -i /hpcfs/bes/mlgpu/sharma/ML_GPU/Samples/DNN_MoreVar_v5_BScoreBugFix/ -t "+ str(args.isTrain) + " -ModelToUse "+ str(args.ModelToUse) +" -s "+dirTag+" -w "+args.weights +" -lr "+str(args.learnRate)+" -e "+str(args.epochs)+" -b "+str(args.batch_size)+" -o "+args.optimizer + " -j "+args.json + " -nHiddenLayer "+str(args.nHiddenLayer) + " -dropoutLayer "+str(args.dropoutLayer)
#===================================================================
import os
def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

def CopyImportFiles(dir):
    # os.system("cp dnn_parameter.json "+dir)
    os.system("cp  "+args.json+" "+dir)
    os.system("cp train-BinaryDNN_WWvsBB.py "+dir)

check_dir(LogDirPath.replace("//","/"))
CopyImportFiles(LogDirPath.replace("//","/"))

from datetime import datetime
CURRENT_DATETIME = datetime.now()

DateTimeString = (str(CURRENT_DATETIME.year)[-2:]
              +str(format(CURRENT_DATETIME.month,'02d'))
              +str(format(CURRENT_DATETIME.day,'02d'))
              +"_"
              +str(format(CURRENT_DATETIME.hour,'02d'))
              +str(format(CURRENT_DATETIME.minute,'02d'))
              +str(format(CURRENT_DATETIME.second,'02d'))
              )
SlurmScriptName = "slurm_submit_"+DateTimeString+".sh"

SlurmJobName = args.jobName+"_"+DateTimeString

print("Name of slurm script: %s"%(SlurmScriptName))

LimitOutTextFile = open(LogDirPath.replace("//","/")+"/"+SlurmScriptName, "w")
message = """#! /bin/bash

######## Part 1 #########
# Script parameters     #
#########################

# Specify the partition name from which resources will be allocated, mandatory option
#SBATCH --partition=gpu

# Specify the QOS, mandatory option
#SBATCH --qos=normal

# Specify which group you belong to, mandatory option
# This is for the accounting, so if you belong to many group,
# write the experiment which will pay for your resource consumption
#SBATCH --account=mlgpu

# Specify your job name, optional option, but strongly recommand to specify some name
#SBATCH --job-name=%s

# Specify how many cores you will need, default is one if not specified
#SBATCH --ntasks=1

# Specify the output file path of your job
# Attention!! Your afs account must have write access to the path
# Or the job will be FAILED!
#SBATCH --output=%s/logSlurm_%%u_%%x_%%j.out

# Specify memory to use, or slurm will allocate all available memory in MB
#SBATCH --mem-per-cpu=30000

# Specify how many GPU cards to use
#SBATCH --gres=gpu:v100:1

######## Part 2 ######
# Script workload    #
######################

# Replace the following lines with your real workload
########################################
# TF: 2.3.0, python: 3.7.6
# source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc8-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_100cuda/x86_64-centos7-gcc8-opt/setup.sh
# source /cvmfs/sft.cern.ch/lcg/views/dev4python2/latest/x86_64-centos7-gcc8-opt/setup.sh
# pip install shap
export QT_QPA_PLATFORM=offscreen
# Reference: https://stackoverflow.com/a/55210689/2302094
export XDG_RUNTIME_DIR=/hpcfs/bes/mlgpu/sharma/ML_GPU/someRuntimeFix
date
cd %s
time(%s)
echo ""
echo "==================================================="
time(python -m json.tool HHWWBBDNN_binary_%s_%s/model_serialised.json > HHWWBBDNN_binary_%s_%s/model_serialised_nice.json)
echo ""
echo "==================================================="
echo ""
#echo "Convert the model.h5 to model.pb"
#time(python convert_hdf5_2_pb.py --input HHWWBBDNN_binary_%s_%s/model.h5 --output HHWWBBDNN_binary_%s_%s/model.pb)
#echo "==================================================="
echo "Time stamp:"
date
##########################################
# Work load end

# Do not remove below this line

# list the allocated hosts
srun -l hostname

# list the GPU cards of the host
/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

sleep 180

"""

message = message %(SlurmJobName,LogDirPath.replace("//","/"),MacroPath,CommandToRun,dirTag,args.weights,dirTag,args.weights,dirTag,args.weights,dirTag,args.weights)

LimitOutTextFile.write(message+"\n")

LimitOutTextFile.close()
# os.system('cp '+SlurmScriptName+' '+LogDirPath.replace("//","/"))
print("Log directory name: %s"%LogDirPath.replace("//","/"))
print("Run slurm script using:")
print("\tsbatch %s/%s"%(LogDirPath.replace("//","/"),SlurmScriptName))
os.system("sbatch %s/%s"%(LogDirPath.replace("//","/"),SlurmScriptName))


time.sleep(3)
print("Check if *.out file exists or not...")
for files_ in sorted(os.listdir(LogDirPath)):
  # print ("files_: %s"%files_)
  if files_.endswith(".out"):
    print("tail -f {}".format(os.path.join(LogDirPath,files_)))
  # else:
    # print("No log file found...")

