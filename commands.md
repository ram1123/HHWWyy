# 15 September 2024

```bash
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/MultiClassDNN_Outputs/HMuMu/  --num_events 1000 --job_name DNN_test_parametric
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/check_highMass_Ntuples  --num_events 1000 --job_name DNN_test_parametric
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 1000 --job_name DNN_test_parametric



time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats --epochs 100 --batch_size 256 --use_gateway


time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV1 --epochs 100 --batch_size 30000

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV2 --epochs 100 --batch_size 30000
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStatsV2 --epochs 100 --batch_size 30000 --scan
```

time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_test  --batch_size 30000 --bayes --max_trials 75 --executions_per_trial 1 --epochs 100
time python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/ --output_dir /depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/  --num_events 0 --job_name DNN_multiclass_fullStats_Scan  --batch_size 30000 --bayes --max_trials 75 --executions_per_trial 3 --epochs 100 2>&1 | tee DNN_multiclass_fullStats_Scan.log


# OLD commands

python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/DNNOutputs  --num_events 1000 --job_name DNN_test_parametric

python train-BinaryDNN_WWvsBB_parametric_tfDataset.py --inputPath /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/ --output_dir /depot/cms/users/shar1172/HighMassSearch/HZZ2l2nu/HZZ_mergedrootfiles/DNNOutputs  --num_events 1000 --job_name DNN_test_parametric --retrain
