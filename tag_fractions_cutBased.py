import os, json, numpy as np, pandas as pd
import dask_awkward as dak
import awkward as ak
# from tensorflow.keras.models import load_model

from rich import print

# ----------- Dask Gateway Client -----------
from dask_gateway import Gateway
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
client = gateway.connect(cluster_info.name).get_client()
print("Gateway Client created")

# ---------- CONFIG (edit paths as needed) ----------
INPUT_ROOT = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = (
    "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
)
SCALER_NPZ = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"
OUT_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------------------------

# Your selection module
import selection

# Processes to read
SAMPLES = {
    # "SampleName": (category_tag, if DY then true else false)
    "ggh_powhegPS": ("notbtag", "False"),
    "vbf_powheg_dipole": ("notbtag", "False"),
    # backgrounds (union of ggh or vbf selections)
    "dy_VBF_filter": ("notbtag", "True"),
    "dy_M-100To200_MiNNLO": ("notbtag", "True"),
    "dy_M-50_MiNNLO": ("notbtag", "True"),
    "ewk_lljj_mll50_mjj120": ("notbtag", "False"),
    "ttjets_dl": ("notbtag", "False"),
    "ttjets_sl": ("notbtag", "False"),
}

# --- Load feature order from the same JSON used for training
with open(FEATURES_JSON, "r") as f:
    variables_json = json.load(f)
FEATURES = [k for k, _ in variables_json.items()]

for sample, (tag, filter_func) in SAMPLES.items():
    true_class = 'ggh'

    # df_base = load_selected_df(sample, tag_for_selection=tag_for_baseline)
    pat = os.path.join(INPUT_ROOT, sample, "*.parquet")
    ddf = dak.from_parquet(pat, columns=FEATURES+["dimuon_mass", "year", "nBtagLoose_nominal", "nBtagMedium_nominal", "gjj_mass"])

    ddf_sel_ggh = selection.applyRegionCatCuts(
        ddf,
        category='ggh',
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=filter_func,
    )
    ddf_sel_vbf = selection.applyRegionCatCuts(
        ddf,
        category="vbf",
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=filter_func,
    )

    n_start = ak.num(ddf, axis=0).compute()
    n_ggh = ak.num(ddf_sel_ggh, axis=0).compute()
    n_vbf = ak.num(ddf_sel_vbf, axis=0).compute()
    n_total = n_ggh + n_vbf

    print(
        f"Sample: {sample:<{21}}, n_start: {n_start:>{7}}, n_total: {n_total:>{7}}, n_ggh: {n_ggh:>{7}}({n_ggh/n_total:.2%}), n_vbf: {n_vbf:>{7}}({n_vbf/n_total:6.2%})"
    )
