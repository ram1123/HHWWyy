# tag_fractions.py
import os, json, numpy as np, pandas as pd
import dask_awkward as dak
import awkward as ak
from tensorflow.keras.models import load_model

# ---------- CONFIG (edit paths as needed) ----------
INPUT_ROOT   = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
SCALER_NPZ    = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH    = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"
OUT_DIR       = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------------------------------

# Your selection module
import selection

# Processes to read
SAMPLES = {
    "ggh_powhegPS":      {"true": "ggh"},
    "vbf_powheg_dipole": {"true": "vbf"},
    # all backgrounds combined for B fractions
    "dy_VBF_filter":         {"true": "bkg"},
    "dy_M-100To200_MiNNLO":  {"true": "bkg"},
    "dy_M-50_MiNNLO":        {"true": "bkg"},
    "ewk_lljj_mll50_mjj120": {"true": "bkg"},
    "ttjets_dl":             {"true": "bkg"},
    "ttjets_sl":             {"true": "bkg"},
}

# --- Load feature order from the same JSON used for training
with open(FEATURES_JSON, "r") as f:
    variables_json = json.load(f)
FEATURES = [k for k, _ in variables_json.items() if k not in ("target","process_ID","classweight")]

# --- Load saved scaler (mean/scale) in the same order
scaler_blob = np.load(SCALER_NPZ, allow_pickle=False)
saved_features = scaler_blob["features"].astype(str).tolist()
if saved_features != FEATURES:
    missing = [c for c in FEATURES if c not in saved_features]
    extra   = [c for c in saved_features if c not in FEATURES]
    raise RuntimeError(f"Feature mismatch!\nMissing in scaler: {missing}\nExtra in scaler: {extra}")
mean  = scaler_blob["mean"].astype(np.float32)
scale = scaler_blob["scale"].astype(np.float32)

def scale_numpy(X: np.ndarray) -> np.ndarray:
    # standardize using saved mean/scale; guard zero scale
    safe_scale = np.where(scale==0, 1.0, scale)
    Z = (X - mean) / safe_scale
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return Z

# --- Load Keras model
model = load_model(MODEL_PATH)

# --- Helpers
def to_pandas_block(arr_dict, cols):
    """Turn dict of Awkward arrays into a pandas DataFrame with given column order."""
    # dak / awkward -> numpy (flatten first axis)
    data = {c: ak.to_numpy(arr_dict[c]) for c in cols}
    return pd.DataFrame(data, columns=cols)

def apply_hpeak_selection(ddf, tag_for_selection, process_name, do_vbf_filter_study=False):
    """
    Apply your selection exactly as requested:
        selection.applyRegionCatCuts(ddf,
                                     category=tag_for_selection,
                                     region_name="h-peak",
                                     process=process_name,
                                     variation="nominal",
                                     do_vbf_filter_study=do_vbf_filter_study)
    Returns filtered ddf.
    """
    return selection.applyRegionCatCuts(
        ddf,
        category=tag_for_selection,
        region_name="h-peak",
        process=process_name,
        variation="nominal",
        do_vbf_filter_study=do_vbf_filter_study
    )

def load_selected_df(sample, tag_for_selection):
    """
    Read parquet into dak, apply the 'h-peak' selection, return pandas DataFrame of FEATURES.
    """
    pat = os.path.join(INPUT_ROOT, sample, "**", "*.parquet")
    ddf = dak.from_parquet(pat, columns=FEATURES, gather_statistics=False)
    # Apply selection (do_vbf_filter_study default False unless you want to study)
    ddf_sel = apply_hpeak_selection(ddf,
                                    category=tag_for_selection,
                                    region_name="h-peak",
                                    process=sample,
                                    variation="nominal",
                                    do_vbf_filter_study=False
    )
    # Materialize only the features we need, to pandas
    feat_dict = {c: ddf_sel[c] for c in FEATURES}
    df = to_pandas_block(feat_dict, FEATURES)
    return df

def dnn_tag_counts(df_features):
    """
    Return counts tagged by DNN:
      - ggH tag if p_ggh > 0.5
      - VBF tag if p_vbf > 0.5
    (Note: an event can satisfy both if both >0.5. If you want disjoint, change logic.)
    """
    if df_features.empty:
        return dict(N=0, tag_ggh=0, tag_vbf=0)

    X = df_features.to_numpy(dtype=np.float32)
    Xs = scale_numpy(X)
    proba = model.predict(Xs, verbose=0)  # columns: [p_ggh, p_vbf, p_bkg]
    tag_ggh = int((proba[:,0] > 0.5).sum())
    tag_vbf = int((proba[:,1] > 0.5).sum())
    return dict(N=len(df_features), tag_ggh=tag_ggh, tag_vbf=tag_vbf)

def cut_based_counts(sample):
    """
    Cut-based tagging using your selection masks themselves:
      - tag ggH = passes ggh selection on h-peak
      - tag VBF = passes vbf selection on h-peak
    We count within the *same baseline sample’s events* (re-reading once per tag to keep masks simple).
    """
    # ggH-tagged by cuts
    df_g = load_selected_df(sample, tag_for_selection="ggh")
    # VBF-tagged by cuts
    df_v = load_selected_df(sample, tag_for_selection="vbf")
    return dict(tag_ggh=len(df_g), tag_vbf=len(df_v))

# --- Accumulate per-class, then combine backgrounds
rows = []
bkg_accum = {
    "method_dnn":   {"N":0, "tag_ggh":0, "tag_vbf":0},
    "method_cuts":  {"N":0, "tag_ggh":0, "tag_vbf":0},
}

for sample, meta in SAMPLES.items():
    # true_cls = meta["true"]  # 'ggh' | 'vbf' | 'bkg'
    true_cls = 'ggh'

    # For “total N” we’ll define baseline as h-peak selection with the process’ own tag:
    #   ggH sample -> category='ggh'
    #   VBF sample -> category='vbf'
    #   background -> category='bkg'
    tag_for_baseline = true_cls

    # Load once for DNN tagging (baseline selection = h-peak with that tag)
    df_base = load_selected_df(sample, tag_for_selection=tag_for_baseline)
    dnn_counts = dnn_tag_counts(df_base)

    # Cut-based counts (how many in this sample pass the ggH/VBF cut-based selections)
    cuts_counts = cut_based_counts(sample)
    cuts_counts["N"] = len(df_base)  # same baseline total N for normalization

    # Compose row
    def frac(n_tag, N): return (n_tag / N) if N > 0 else 0.0
    row = {
        "sample": sample,
        "true_class": true_cls,
        "N_baseline_hpeak": dnn_counts["N"],

        # DNN method
        "DNN_tag_ggh": dnn_counts["tag_ggh"],
        "DNN_tag_vbf": dnn_counts["tag_vbf"],
        "DNN_frac_ggh": frac(dnn_counts["tag_ggh"], dnn_counts["N"]),
        "DNN_frac_vbf": frac(dnn_counts["tag_vbf"], dnn_counts["N"]),

        # Cut-based method
        "CUT_tag_ggh": cuts_counts["tag_ggh"],
        "CUT_tag_vbf": cuts_counts["tag_vbf"],
        "CUT_frac_ggh": frac(cuts_counts["tag_ggh"], cuts_counts["N"]),
        "CUT_frac_vbf": frac(cuts_counts["tag_vbf"], cuts_counts["N"]),
    }
    rows.append(row)

    # Accumulate backgrounds for the “all backgrounds combined” view
    if true_cls == "bkg":
        bkg_accum["method_dnn"]["N"]       += dnn_counts["N"]
        bkg_accum["method_dnn"]["tag_ggh"] += dnn_counts["tag_ggh"]
        bkg_accum["method_dnn"]["tag_vbf"] += dnn_counts["tag_vbf"]
        bkg_accum["method_cuts"]["N"]       += cuts_counts["N"]
        bkg_accum["method_cuts"]["tag_ggh"] += cuts_counts["tag_ggh"]
        bkg_accum["method_cuts"]["tag_vbf"] += cuts_counts["tag_vbf"]

# --- Per-sample table
df_out = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "tag_fractions_per_sample.csv")
df_out.to_csv(csv_path, index=False)

# --- Compact class-wise summary (ggH, VBF, All-Bkg)
def summarize_class(df, true_cls):
    sub = df[df["true_class"] == true_cls]
    N = int(sub["N_baseline_hpeak"].sum())
    dnn_g = int(sub["DNN_tag_ggh"].sum()); dnn_v = int(sub["DNN_tag_vbf"].sum())
    cut_g = int(sub["CUT_tag_ggh"].sum());  cut_v = int(sub["CUT_tag_vbf"].sum())
    def frac(n): return (n / N) if N > 0 else 0.0
    return dict(
        N=N,
        DNN_tag_ggh=dnn_g, DNN_tag_vbf=dnn_v, DNN_frac_ggh=frac(dnn_g), DNN_frac_vbf=frac(dnn_v),
        CUT_tag_ggh=cut_g, CUT_tag_vbf=cut_v, CUT_frac_ggh=frac(cut_g),  CUT_frac_vbf=frac(cut_v),
    )

summary = {
    "ggH": summarize_class(df_out, "ggh"),
    "VBF": summarize_class(df_out, "vbf"),
    "AllBackgrounds_DNN": {
        "N": bkg_accum["method_dnn"]["N"],
        "tag_ggh": bkg_accum["method_dnn"]["tag_ggh"],
        "tag_vbf": bkg_accum["method_dnn"]["tag_vbf"],
        "frac_ggh": (bkg_accum["method_dnn"]["tag_ggh"] / bkg_accum["method_dnn"]["N"]) if bkg_accum["method_dnn"]["N"]>0 else 0.0,
        "frac_vbf": (bkg_accum["method_dnn"]["tag_vbf"] / bkg_accum["method_dnn"]["N"]) if bkg_accum["method_dnn"]["N"]>0 else 0.0,
    },
    "AllBackgrounds_CUT": {
        "N": bkg_accum["method_cuts"]["N"],
        "tag_ggh": bkg_accum["method_cuts"]["tag_ggh"],
        "tag_vbf": bkg_accum["method_cuts"]["tag_vbf"],
        "frac_ggh": (bkg_accum["method_cuts"]["tag_ggh"] / bkg_accum["method_cuts"]["N"]) if bkg_accum["method_cuts"]["N"]>0 else 0.0,
        "frac_vbf": (bkg_accum["method_cuts"]["tag_vbf"] / bkg_accum["method_cuts"]["N"]) if bkg_accum["method_cuts"]["N"]>0 else 0.0,
    }
}

# Optional: simple significance proxies (per tag) using combined S (ggH+VBF) and combined B
for method in ("DNN", "CUT"):
    S_g = summary["ggH"][f"{method}_tag_ggh"] + summary["VBF"][f"{method}_tag_ggh"]
    S_v = summary["ggH"][f"{method}_tag_vbf"] + summary["VBF"][f"{method}_tag_vbf"]
    B_g = summary["AllBackgrounds_"+method]["tag_ggh"]
    B_v = summary["AllBackgrounds_"+method]["tag_vbf"]
    def sig(S,B): return (S/np.sqrt(B)) if B>0 else 0.0
    summary[f"significance_{method}"] = {
        "ggH_tag_bin": sig(S_g, B_g),
        "VBF_tag_bin": sig(S_v, B_v),
    }

# Save summary
with open(os.path.join(OUT_DIR, "tag_fractions_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"[done] Wrote: {csv_path}")
print(f"[done] Wrote: {os.path.join(OUT_DIR, 'tag_fractions_summary.json')}")
