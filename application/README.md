- Step: 1: Get DNN score for all samples.
   ```bash
   python run_network_evaluation.py -d HHWWyyDNN_binary_test_BalanceYields -p HHWWgg TTGJets QCD_Pt-30to40 QCD_Pt-40toInf GJet_Pt-20to40 GJet_Pt-40toInf
   ```

- Step: 2: Get Data-MC
   ```bash
   python DNN_Evaluation_Control_Plots.py -i HHWWyyDNN_binary_test_BalanceYields
   ```
   