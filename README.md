# Useful information

- Input samples: `/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets`
- Input variables json: `input_variables.json`

# Setup

```bash
source setup_env.sh
time python train-multiclassDNN.py --inputPath <InputSamplesPath>  --output_dir <outputPath>  --json <Json file path having input feature list> --num_events 0 --job_name <AnyTags> --epochs 25
```

Command options:
- `--inputPath`: Path to the input samples
- `--output_dir`: Path to save the output model and plots
- `--json`: Json file path having input feature list
- `--num_events`: Number of events to use for training (0 means all events). For testing, use a smaller number like 10_000
- `--job_name`: Any tag to identify the training job
- `--epochs`: Number of epochs to train the model






# OLD

## Setup

```bash
python -m venv xzz2l2nu_env
source xzz2l2nu_env/bin/activate
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
pip install -r requirement.txt
# Run the training
ulimit -s unlimited
python train-BinaryDNN_WWvsBB.py -t 1 -i /eos/user/a/avijay/HZZ_mergedrootfiles/
```

## Training

```bash
. /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
source xzz2l2nu_env/bin/activate
ulimit -s unlimited
time python train-multiclassDNN.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets  --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --json input_variables_more_vars.json --num_events 0 --job_name train_more_vars_v2_13Oct --epochs 25
```
