# Obtain the tagging fractions of ggH and VBF Higgs to 𝛍𝛍

- To obtain this, I need to get the ggH signal sample, and note total number of events. Then out of them how many are tagged as ggH and how many as VBF.
- This I need to do for both ggH and VBF MC samples, using two different methods:
  - Using the DNN score (if VBF score > 0.5, it is tagged as VBF else it is background), and
  - Using the cut-based method
- Then I can get the fractions of ggH and VBF events tagged as ggH and VBF, for both methods.
- Also, for the two method also obtain the fraction of background events (from all backgrounds combined) that are tagged as ggH and VBF. So, that I can also compare the significance for the two methods.


# Cut based numbers

```bash
Sample: ggh_powhegPS         , n_start:  552382, n_total:  552382, n_ggh:  540253(97.80%), n_vbf:   12129( 2.20%)
Sample: vbf_powheg_dipole    , n_start: 1087869, n_total: 1087869, n_ggh:  650583(59.80%), n_vbf:  437286(40.20%)
Sample: dy_VBF_filter        , n_start: 2331136, n_total: 2331136, n_ggh: 1534965(65.85%), n_vbf:  796171(34.15%)
Sample: dy_M-100To200_MiNNLO , n_start:  314756, n_total:  314756, n_ggh:  313586(99.63%), n_vbf:    1170( 0.37%)
Sample: dy_M-50_MiNNLO       , n_start:    3518, n_total:    3518, n_ggh:    3505(99.63%), n_vbf:      13( 0.37%)
Sample: ewk_lljj_mll50_mjj120, n_start:   17580, n_total:   17580, n_ggh:    9353(53.20%), n_vbf:    8227(46.80%)
Sample: ttjets_dl            , n_start:  199003, n_total:  199003, n_ggh:  189643(95.30%), n_vbf:    9360( 4.70%)
Sample: ttjets_sl            , n_start:    3065, n_total:    3065, n_ggh:    2859(93.28%), n_vbf:     206( 6.72%)
```

Significance (s/sqrt(s+b)) =

# DNN based numbers

## VBF score > 0.90 is tagged as VBF, else ggH

```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  546762(98.98%), n_vbf:    5620( 1.02%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  797031(73.27%), n_vbf:  290838(26.73%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 2319792(99.51%), n_vbf:   11344( 0.49%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  314207(99.83%), n_vbf:     549( 0.17%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3516(99.94%), n_vbf:       2( 0.06%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   16375(93.15%), n_vbf:    1205( 6.85%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  198817(99.91%), n_vbf:     186( 0.09%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    3057(99.74%), n_vbf:       8( 0.26%)
```

## VBF score > 0.85 is tagged as VBF, else ggH

```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  540173(97.79%), n_vbf:   12209( 2.21%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  696401(64.02%), n_vbf:  391468(35.98%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 2306612(98.95%), n_vbf:   24524( 1.05%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  313502(99.60%), n_vbf:    1254( 0.40%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3509(99.74%), n_vbf:       9( 0.26%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   15844(90.13%), n_vbf:    1736( 9.87%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  198534(99.76%), n_vbf:     469( 0.24%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    3046(99.38%), n_vbf:      19( 0.62%)
```

## VBF score > 0.80 is tagged as VBF, else ggH

```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  529753(95.90%), n_vbf:   22629( 4.10%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  613234(56.37%), n_vbf:  474635(43.63%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 2290239(98.25%), n_vbf:   40897( 1.75%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  312226(99.20%), n_vbf:    2530( 0.80%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3504(99.60%), n_vbf:      14( 0.40%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   15362(87.38%), n_vbf:    2218(12.62%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  198096(99.54%), n_vbf:     907( 0.46%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    3034(98.99%), n_vbf:      31( 1.01%)
```

## VBF score > 0.75 is tagged as VBF, else ggH

```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  512040(92.70%), n_vbf:   40342( 7.30%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  541429(49.77%), n_vbf:  546440(50.23%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 2270916(97.42%), n_vbf:   60220( 2.58%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  310123(98.53%), n_vbf:    4633( 1.47%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3494(99.32%), n_vbf:      24( 0.68%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   14910(84.81%), n_vbf:    2670(15.19%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  197512(99.25%), n_vbf:    1491( 0.75%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    3025(98.69%), n_vbf:      40( 1.31%)
```

## VBF score > 0.5 is tagged as VBF, else ggH
```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  336652(60.95%), n_vbf:  215730(39.05%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  272187(25.02%), n_vbf:  815682(74.98%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 2123227(91.08%), n_vbf:  207909( 8.92%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  286727(91.10%), n_vbf:   28029( 8.90%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3377(95.99%), n_vbf:     141( 4.01%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   12529(71.27%), n_vbf:    5051(28.73%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  191050(96.00%), n_vbf:    7953( 4.00%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    2879(93.93%), n_vbf:     186( 6.07%)
```

## VBF score > 0.3 is tagged as VBF, else ggH
```bash
Sample: ggh_powhegPS         , n_selected:  552382, n_ggh:  203771(36.89%), n_vbf:  348611(63.11%)
Sample: vbf_powheg_dipole    , n_selected: 1087869, n_ggh:  135122(12.42%), n_vbf:  952747(87.58%)
Sample: dy_VBF_filter        , n_selected: 2331136, n_ggh: 1937997(83.14%), n_vbf:  393139(16.86%)
Sample: dy_M-100To200_MiNNLO , n_selected:  314756, n_ggh:  258437(82.11%), n_vbf:   56319(17.89%)
Sample: dy_M-50_MiNNLO       , n_selected:    3518, n_ggh:    3148(89.48%), n_vbf:     370(10.52%)
Sample: ewk_lljj_mll50_mjj120, n_selected:   17580, n_ggh:   10496(59.70%), n_vbf:    7084(40.30%)
Sample: ttjets_dl            , n_selected:  199003, n_ggh:  179076(89.99%), n_vbf:   19927(10.01%)
Sample: ttjets_sl            , n_selected:    3065, n_ggh:    2647(86.36%), n_vbf:     418(13.64%)
```

---


# ---------- CONFIG (edit paths as needed) ----------
INPUT_ROOT = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"  # directories per process with parquet
FEATURES_JSON = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"  # same JSON you train with (for feature order)
SCALER_NPZ = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"
OUT_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)
-----------------------------------------------------------

1. Use the   `dask_awkward as dak` to read the parquet files.
2. Selection should be made like this:
    ```python
    ddf = selection.applyRegionCatCuts(ddf,
                                    category=tag,
                                    region_name="h-peak",
                                    process=subdir,
                                    variation="nominal",
                                    do_vbf_filter_study=filter_func)
    ```

-----

After below step:
```python
    ddf_sel_skim = selection.applyRegionCatCuts(
        ddf,
        category=tag,
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=filter_func,
    )
```

Evaluate the model based on following information:
FEATURES_JSON = (
    "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
)
SCALER_NPZ = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"

