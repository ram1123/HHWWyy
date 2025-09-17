import dask_awkward as dak
import dask.array as da
import awkward as ak
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import json

from rich import print

import selection

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

# ----------- CONFIG -----------
INPUT_ROOT = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = (
    "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
)
SCALER_NPZ = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"
OUT_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)


# --- Load feature order from the same JSON used for training
def load_features_from_json(path):
    with open(path, "r") as f:
        items = json.load(f).items()
    # keep the exact training order and drop non-features if present
    return [k for k, _ in items if k not in ("target", "process_ID", "classweight")]


FEATURES = load_features_from_json(FEATURES_JSON)

# Extra physics columns to preserve for output/plots
EXTRA_KEEP = [
    "dimuon_mass",
    "dimuon_pt",
    "jj_mass_nominal",
    "year",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
    "gjj_mass",
]

# Processes to read  (second item is a BOOL, not a string)
SAMPLES = {
    "ggh_powhegPS": ("notbtag", False),
    "vbf_powheg_dipole": ("notbtag", False),
    "dy_VBF_filter": ("notbtag", True),
    # "dy_M-100To200_MiNNLO": ("notbtag", True),
    # "dy_M-50_MiNNLO": ("notbtag", True),
    # "ewk_lljj_mll50_mjj120": ("notbtag", False),
    # "ttjets_dl": ("notbtag", False),
    # "ttjets_sl": ("notbtag", False),
}

# ----------- LOAD MODEL & SCALER -----------
print("[bold]Loading model…[/]")
model = load_model(MODEL_PATH)
# model.summary()

sc_npz = np.load(SCALER_NPZ)
mean = sc_npz.get("mean_", sc_npz.get("mean"))
scale = sc_npz.get("scale_", sc_npz.get("scale"))
if mean is None or scale is None:
    raise RuntimeError(f"Scaler npz missing mean_/scale_: {SCALER_NPZ}")
mean = mean.astype(np.float64)
scale = scale.astype(np.float64)
safe_scale = np.where(scale == 0, 1.0, scale)

# ----------- LOAD DATA, PREDICT, and SAVE -----------
# print("[bold]Loading data & evaluating…[/]")
for sample, (tag, vbf_filter_bool) in SAMPLES.items():
    pat = os.path.join(INPUT_ROOT, sample, "*.parquet")

    # Prefer reading only needed columns, but fall back gracefully
    read_cols = list(
        dict.fromkeys(FEATURES + EXTRA_KEEP + ["process_ID", "target"])
    )  # dedup
    try:
        ddf = dak.from_parquet(pat, columns=[c for c in read_cols if c != "target"])
    except Exception as e:
        print(
            f"[warn] {sample}: selective column read failed ({e}); reading all columns."
        )
        ddf = dak.from_parquet(pat)

    # print(f"Applying selections for {sample}...")
    ddf_sel_skim = selection.applyRegionCatCuts(
        ddf,
        category=tag,
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=vbf_filter_bool,
    )

    ak_array = ddf_sel_skim.compute()

    # Helper: Awkward column -> flat numpy (OptionType -> NaN)
    def col_to_np(name):
        return ak.to_numpy(ak_array[name])

    # Build DataFrame
    df_eval = pd.DataFrame({var: col_to_np(var) for var in FEATURES})
    for c in [
        "dimuon_mass",
        "year",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "gjj_mass",
    ]:
        df_eval[c] = col_to_np(c)

    # Clean & standardize using saved stats
    df_eval[FEATURES] = (
        df_eval[FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float64)
    )
    X = df_eval[FEATURES].to_numpy(copy=False)
    X = (X - mean) / safe_scale
    X = np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)

    # Predict
    scores = model.predict(X, batch_size=4096, verbose=0)
    # print(f"{sample}: scores shape = {scores.shape}")

    # Save per-event scores
    out = df_eval
    # .loc[
    #     :,
    #     [
    #         "dimuon_mass",
    #         "year",
    #         "nBtagLoose_nominal",
    #         "nBtagMedium_nominal",
    #         "gjj_mass",
    #     ],
    # ].copy()

    # name outputs as score_0, score_1, …
    # dict for scores ggh: 0, vbf: 1, bkg: 2
    score_dict = {0: "ggh", 1: "vbf", 2: "bkg"}
    for i in range(scores.shape[1] if scores.ndim > 1 else 1):
        # out[f"score_{i}"] = scores[:, i] if scores.ndim > 1 else scores
        out[f"score_{score_dict[i]}"] = scores[:, i] if scores.ndim > 1 else scores
    out_path = os.path.join(OUT_DIR, f"{sample}_scores.parquet")

    # print the field names for debugging
    print(f"Output columns: {out.columns.tolist()}")

    out.to_parquet(out_path, index=False)
    # print(f"[green]Saved[/] → {out_path}")

    # Count the number of events that passed the selection vbf_score > 0.5
    # n_events = len(out)
    # n_passed = (out["score_vbf"] > 0.5).sum()
    # n_failed = n_events - n_passed
    # # print(
    # # f"Sample: {sample:<{21}}, n_start: {n_start:>{7}}, n_total: {n_total:>{7}}, n_ggh: {n_ggh:>{7}}({n_ggh/n_total:.2%}), n_vbf: {n_vbf:>{7}}({n_vbf/n_total:6.2%})"
    # # )
    # print(
    #     f"Sample: {sample:<{21}}, n_selected: {n_events:>{7}}, n_ggh: {n_failed:>{7}}({n_failed/n_events:.2%}), n_vbf: {n_passed:>{7}}({n_passed/n_events:6.2%})"
    # )

    # if score_vbf > 0.5, then plot njets_nominal
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.hist(
        out["njets_nominal"],
        bins=np.arange(-0.5, 10.5, 1),
        alpha=0.5,
        label="All events",
        color="blue",
        density=True,
    )
    plt.hist(
        out[out["score_vbf"] > 0.5]["njets_nominal"],
        bins=np.arange(-0.5, 10.5, 1),
        alpha=0.5,
        label="VBF-like (score_vbf > 0.5)",
        color="orange",
        density=True,
    )
    plt.xlabel("Number of Jets (nominal)")
    plt.ylabel("Normalized Entries")
    plt.title(f"Jet Multiplicity after Selection for {sample}")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(OUT_DIR, f"{sample}_njets_nominal.pdf"))
    plt.close()


print("[bold green]Done.[/]")
