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
python train-BinaryDNN_WWvsBB.py -t 1 -i /eos/user/a/avijay/HZZ_mergedrootfiles/
```


# Submit the condor jobs

```bash
# test
python prepare_condor_jobs.py --job_name "test_workday_26Nov" --max_events 1000 --job_flavour "workday" --json "input_variables_test.json"

# final
python prepare_condor_jobs.py --job_name "Final_25Nov_tomorrow_MoreVariables" --max_events -1 --job_flavour "tomorrow"

python prepare_condor_jobs.py --job_name "Final_25Nov_tomorrow_parametric" --max_events -1 --job_flavour "tomorrow"


```


# New script

```bash
python train-BinaryDNN_WWvsBB_NEW.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name DNN_test_26Nov
python train-BinaryDNN_WWvsBB_parametric.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name DNN_test_parametric

python train-BinaryDNN_WWvsBB_parametric.py --inputPath /eos/user/a/avijay/HZZ_mergedrootfiles/ --output_dir /eos/user/r/rasharma/HZZ2l2nu/  --num_events 1000 --job_name DNN_test_parametric --json input_variables_test.json
```
