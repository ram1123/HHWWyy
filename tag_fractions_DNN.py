import os, json, pandas as pd
import dask_awkward as dak
import awkward as ak
import dask.dataframe as dd
from tensorflow.keras.models import load_model
from rich import print
import numpy as np

# ----------- CONFIG -----------
INPUT_ROOT   = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
SCALER_NPZ    = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/scaler.npz"
MODEL_PATH    = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/model.keras"
OUT_DIR       = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)

# Your selection module
import selection

# Processes to read  (second item is a BOOL, not a string)
SAMPLES = {
    "ggh_powhegPS":        ("notbtag", False),
    "vbf_powheg_dipole":   ("notbtag", False),
    "dy_VBF_filter":       ("notbtag", True),
    "dy_M-100To200_MiNNLO":("notbtag", True),
    "dy_M-50_MiNNLO":      ("notbtag", True),
    "ewk_lljj_mll50_mjj120":("notbtag", False),
    "ttjets_dl":           ("notbtag", False),
    "ttjets_sl":           ("notbtag", False),
}

# --- Load feature order from the same JSON used for training
def load_features_from_json(path):
    with open(path, "r") as f:
        items = json.load(f).items()
    # keep the exact training order and drop non-features if present
    return [k for k, _ in items if k not in ("target", "process_ID", "classweight")]

FEATURES = load_features_from_json(FEATURES_JSON)

# Extra physics columns to preserve for output/plots
EXTRA_KEEP = ["dimuon_mass", "dimuon_pt", "jj_mass_nominal", "year",
              "nBtagLoose_nominal", "nBtagMedium_nominal", "gjj_mass"]

# --------------------
# Utilities
# --------------------
def load_npz_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def load_scaler_npz(path):
    z = np.load(path, allow_pickle=False)
    return {
        "mean":   z["mean"].astype(np.float32),
        "scale":  z["scale"].astype(np.float32),
        "var":    z["var"].astype(np.float32),
        "features": [str(x) for x in z["features"]],
    }

def apply_saved_scaler(X_df: pd.DataFrame, scaler_npz: dict):
    feat_saved = scaler_npz["features"]
    # ensure all required features exist
    missing = [c for c in feat_saved if c not in X_df.columns]
    if missing:
        raise KeyError(f"Missing features in dataframe (needed by scaler): {missing}")
    X = X_df[feat_saved].to_numpy(dtype=np.float32, copy=False)
    mean  = scaler_npz["mean"]
    scale = scaler_npz["scale"]
    scale_safe = np.where(scale == 0.0, 1.0, scale)
    X = (X - mean) / scale_safe
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def to_dask_dataframe(obj):
    """Convert selection output (often a dak.Array of records) to dd.DataFrame
    robustly by cleaning option types and providing explicit meta."""
    # Already a Dask DataFrame?
    if isinstance(obj, dd.DataFrame):
        return obj

    # Unwrap common wrappers
    if isinstance(obj, tuple) and obj:
        obj = obj[0]
    if isinstance(obj, dict) and "array" in obj:
        obj = obj["array"]

    # dask-awkward Array → dask.dataframe (preferred path)
    if isinstance(obj, dak.Array):
        # Must be record-like
        try:
            fields = dak.fields(obj)
        except Exception:
            fields = []
        if not fields:
            raise TypeError("selection output is a non-record dask_awkward Array (no named fields).")

        # Keep only the columns we actually need downstream:
        # FEATURES + any extras present in the array
        needed = []
        if "FEATURES" in globals():
            needed += [c for c in FEATURES if c in fields]
        if "EXTRA_KEEP" in globals():
            needed += [c for c in EXTRA_KEEP if c in fields]
        needed = list(dict.fromkeys(needed))  # de-dup, keep order
        if not needed:
            # fallback to all fields
            needed = fields

        # Clean each field: fill None→0 and cast to float64 (compatible with scaler)
        cleaned = {}
        for c in needed:
            col = obj[c]
            # fill None before casting
            col = dak.fill_none(col, 0)
            # cast to numeric (float64 to be safe for pandas/meta)
            col = dak.astype(col, np.float64)
            cleaned[c] = col

        # Re-zip to a record with only cleaned numeric columns
        rec = dak.zip(cleaned)

        # Provide explicit pandas meta to dodge typetracer masked dtypes
        meta = pd.DataFrame({c: np.array([], dtype="float64") for c in needed})

        # Convert using the cleaned record + explicit meta
        return dak.to_dataframe(rec, meta=meta)  # -> dd.DataFrame

    # In-memory awkward Array → pandas → dask
    if isinstance(obj, ak.Array):
        # Clean option types before conversion
        # (iterate over fields if it's a record)
        if ak.is_record(obj) or (hasattr(ak, "fields") and ak.fields(obj)):
            data = {}
            for c in ak.fields(obj):
                col = ak.fill_none(obj[c], 0)
                data[c] = ak.values_astype(col, "float64")
            obj = ak.zip(data)
        pdf = ak.to_dataframe(obj)
        if isinstance(pdf.columns, pd.MultiIndex):
            pdf.columns = ["_".join(str(x) for x in tup if x not in (None, "")) for tup in pdf.columns]
        parts = max(1, (len(pdf) // 200_000) or 1)
        return dd.from_pandas(pdf.reset_index(drop=True), npartitions=parts)

    raise TypeError(f"Unsupported selection output type: {type(obj)}")

def ensure_truth_column(df: pd.DataFrame):
    if "target" in df.columns:
        return df["target"].astype(np.int64)
    if "process_ID" in df.columns:
        proc = df["process_ID"].astype(str).str.lower()
        def _map(p):
            if "ggh" in p: return 0
            if "vbf" in p: return 1
            return 2
        return proc.map(_map).astype(np.int64)
    return None

# --------------------
# ROOT output (optional)
# --------------------
import ROOT
def write_root_tree(df_scored: pd.DataFrame, outpath: str, extra_branches=()):
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile(outpath, "RECREATE")
    t = ROOT.TTree("Events", "Scored events")
    import array
    buffers = {}
    # scores + pred
    for name in ["score_ggh","score_vbf","score_bkg"]:
        buffers[name] = array.array("f", [0.])
        t.Branch(name, buffers[name], f"{name}/F")
    buffers["pred_idx"] = array.array("i", [0])
    t.Branch("pred_idx", buffers["pred_idx"], "pred_idx/I")
    # optional physics vars
    for b in extra_branches:
        if b in df_scored.columns:
            buffers[b] = array.array("f", [0.])
            t.Branch(b, buffers[b], f"{b}/F")
    # fill
    cols = df_scored.columns
    for _, row in df_scored.iterrows():
        buffers["score_ggh"][0] = float(row["score_ggh"])
        buffers["score_vbf"][0] = float(row["score_vbf"])
        buffers["score_bkg"][0] = float(row["score_bkg"])
        buffers["pred_idx"][0]  = int(row["pred_idx"])
        for b in extra_branches:
            if b in cols:
                try:
                    buffers[b][0] = float(row[b])
                except Exception:
                    buffers[b][0] = 0.0
        t.Fill()
    t.Write(); f.Close()

# --------------------
# Scoring
# --------------------
def score_dataframe(df: pd.DataFrame, model, feature_order, scaler_npz, batch_size=4096):
    X_df = df[feature_order].copy()
    X = apply_saved_scaler(X_df, scaler_npz)
    yhat = model.predict(X, batch_size=batch_size, verbose=0).astype(np.float32)  # (N,3)
    out = df.copy()
    out["score_ggh"] = yhat[:, 0]
    out["score_vbf"] = yhat[:, 1]
    out["score_bkg"] = yhat[:, 2]
    out["pred_idx"]  = np.argmax(yhat, axis=1).astype(np.int64)
    out["pred_label"]= out["pred_idx"].map({0:"ggh",1:"vbf",2:"bkg"})
    return out

# --------------------
# Main evaluation
# --------------------
def evaluate_and_plot(
    ddf_sel_skim,
    output_dir,
    features_json=FEATURES_JSON,
    scaler_npz_path=SCALER_NPZ,
    model_path=MODEL_PATH,
    persist_scored_parquet=True,
    write_root_out=True,
):
    os.makedirs(output_dir, exist_ok=True)

    print("[debug] selection output type:", type(ddf_sel_skim))
    try:
        print(" selection fields (if dak.Array):", dak.fields(ddf_sel_skim))
    except Exception:
        pass

    ddf = to_dask_dataframe(ddf_sel_skim)

    feature_order = load_features_from_json(features_json)
    keep_cols = [c for c in feature_order if c in ddf.columns]

    # carry truth/process if present + extra physics vars for output
    maybe_truth_cols = [c for c in ("target", "process_ID") if c in ddf.columns]
    extra_cols = [c for c in EXTRA_KEEP if c in ddf.columns]

    ddf_small = ddf[keep_cols + maybe_truth_cols + extra_cols]

    df = ddf_small.compute()
    if df.empty:
        raise RuntimeError("Empty dataframe after selection — nothing to evaluate.")

    model = load_model(model_path)
    scaler_npz = load_scaler_npz(scaler_npz_path)

    # IMPORTANT: use scaler’s feature order (authoritative from training time)
    df_scored = score_dataframe(df, model,
                                feature_order=scaler_npz["features"],
                                scaler_npz=scaler_npz)

    y_true = ensure_truth_column(df_scored)
    if y_true is not None:
        df_scored["target"] = y_true

    if persist_scored_parquet:
        out_parq = os.path.join(output_dir, "scored.parquet")
        df_scored.to_parquet(out_parq, index=False)
        print(f"[saved] {out_parq}")

    if write_root_out:
        out_root = os.path.join(output_dir, "scored.root")
        write_root_tree(df_scored, out_root, extra_branches=tuple(EXTRA_KEEP))
        print(f"[saved] {out_root}")

    print("[done] Evaluation finished.")

# --------------------
# Driver
# --------------------
for sample, (tag, vbf_filter_bool) in SAMPLES.items():
    pat = os.path.join(INPUT_ROOT, sample, "*.parquet")

    # Prefer reading only needed columns, but fall back gracefully
    read_cols = list(dict.fromkeys(FEATURES + EXTRA_KEEP + ["process_ID", "target"]))  # dedup
    try:
        ddf = dak.from_parquet(pat, columns=[c for c in read_cols if c != "target"])
    except Exception as e:
        print(f"[warn] {sample}: selective column read failed ({e}); reading all columns.")
        ddf = dak.from_parquet(pat)

    ddf_sel_skim = selection.applyRegionCatCuts(
        ddf,
        category=tag,
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=vbf_filter_bool,
    )

    evaluate_and_plot(
        ddf_sel_skim,
        output_dir=os.path.join(OUT_DIR, sample),
        features_json=FEATURES_JSON,
        scaler_npz_path=SCALER_NPZ,
        model_path=MODEL_PATH,
        persist_scored_parquet=True,
        write_root_out=True,
    )
