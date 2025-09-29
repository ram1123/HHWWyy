# 27 September 2024

```bash
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_relativeEBEonly --epochs 100 2>&1 | tee -a output.log
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_relativeEBEonly_NoClassWgt --epochs 100 2>&1 | tee -a output.log

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_BothEBE_WithClassWgt --epochs 100 2>&1 | tee -a log_DNN_BothEBE_WithClassWgt.log
```


# 26 September 2024
```bash
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_Removed_EBE --epochs 100

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_Removed_EBEv2 --epochs 100

#DNN_Removed_EBE: removed the EBE variable from the input features
#DNN_Removed_EBEv2: removed the EBE variable and added the class imbalance handling in the model training
time python train-BinaryDNN_WWvsBB_parametric_tfDataset_EBEWeight.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_Removed_EBEv2_SampleWgt --epochs 100
```


# 16 September 2024

```bash
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_Scan_Quick --epochs 100 --retrain
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_Scan_Quick --epochs 100

```

# 15 September 2024

```bash
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/MultiClassDNN_Outputs/HMuMu/  --num_events 1000 --job_name DNN_test_parametric
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/check_highMass_Ntuples  --num_events 1000 --job_name DNN_test_parametric
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 1000 --job_name DNN_test_parametric



time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats --epochs 100 --batch_size 256 --use_gateway


time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV1 --epochs 100 --batch_size 30000

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV2 --epochs 100 --batch_size 30000
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV2 --epochs 100 --batch_size 30000 --scan
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_test  --batch_size 30000 --bayes --max_trials 75 --executions_per_trial 1 --epochs 100

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_Scan  --batch_size 30000 --bayes --max_trials 75 --executions_per_trial 3 --epochs 100 2>&1 | tee DNN_multiclass_fullStats_Scan.log
```



# OLD commands

python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/DNNOutputs  --num_events 1000 --job_name DNN_test_parametric

python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/DNNOutputs  --num_events 1000 --job_name DNN_test_parametric --retrain
