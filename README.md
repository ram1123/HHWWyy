# HHWWyy_DNN
### Authors: Joshuha Thomas-Wilsker
### Institutes: IHEP Beijing, CERN
Package used to train deep neural network for HH->WWyy analysis.

## Environment settings
Several non-standard libraries must be present in your python environment.
To ensure they are present I suggest cloning this library onto a machine/area
for which you have root access. Then setup a conda environment for python 3.7
```
conda create -n <env_title> python=3.7 anaconda
```

```
conda create -n py3 python=3.7 anaconda
```


```
conda activate py3
```

Check the python version you are now using:
```
python --version
```

Check the following libraries are present:

- python 3.7
- shap
- keras
- tensorflow 2.2
- root
- root_numpy
- numpy

If any packages (including those I may have missed from the list above) are missing the code,
you can add the package to the environment easily assuming it doesnt clash or require something
you haven't got in the enviroment setup:
```
conda install <new_library>
```

```bash
conda create -n tfv24 tensorflow
conda activate tfv24
# by default conda installed tf2.0. 
# So, to upgrade tensorflow to the latest version run below command 
# pip install tensorflow --upgrade
pip install tensorflow==2.2
conda install -c anaconda pydot
conda install -c anaconda seaborn
conda install -c conda-forge shap
conda install -c conda-forge uproot uproot-base
# conda install -c conda-forge matplotlib
conda install -c anaconda keras
conda install -c conda-forge root_numpy
```

* ***With tensorflow v2.4 shap is not working. It gives error saying "AttributeError: 'KerasTensor' object has no attribute 'graph'". Till now it seems that there is no solution for this. This issue is mentioned here: https://github.com/slundberg/shap/issues/1694 .***

* while downgrading tensorflow 2.4 to 2.2 I got following error/waring:
   ```
   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
   shap 0.37.0 requires slicer==0.0.3, but you have slicer 0.0.7 which is incompatible.
   ```
   * But downgrading works. 

## Basic training
Running the code:
```
python train-BinaryDNN.py -t <0 or 1> -s <suffix_for_output_dir> -i <input_files_path>
```

The script 'train-BinaryDNN.py' performs several tasks:
- From 'input_variables.json' a list of input variables to use during training is compiled.
- With this information the 'input_files_path' will be used to locate two directories: 1 (Signal) containing the signal ntuples and the other containing the background samples (Bkgs).
- These files are used by the 'load_data' function to create a pandas dataframe.
- So you don't have to recreate the dataframe each time you want to run a new training using the same input variables, the dataframe is stored in the training output directory (in human readable format if you want to inspect it).
- The dataframe is split into a training and a testing sample (events are divided up randomly).
- If class/event weights are needed in order to overcome the class imbalance in the dataset, there are currently two methods to do this. The method used is defined in the hyper-parameter definition section. Search for the 'weights' variable. Other hyper-paramters can be hard coded here as well.
- If one chooses, the code can be used to perform a hyper-parameter scan using the '-p' argument.
- The code can be run in two mode:
    - If you want to perform the fit -t 1 = train new model from scratch.
    - If you just wanted to edit the plots (see plotting/plotter.py) -t 0 = make plots from the pre-trained model in training directory.
- The model is then fit.
- Several diagnostic plots are made by default: input variable correlations, input variable ranking, ROC curves, overfitting plots.
- The model along with a schematic diagram and .json containing a human readable version of the moel parameters is also saved.
- Diagnostic plots along with the model '.h5' and the dataframe will be stored in the output directory.

## The Plotting package

## Evaluating the networks performance
