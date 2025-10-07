#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import json
import glob
import numpy as np
import pandas as pd
import awkward as ak
import dask_awkward as dak
from tensorflow.keras.models import load_model
from rich import print
from typing import List, Optional

import selection  # your module

# ================== USER CONFIG ==================
INPUT_DIR = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"

# # With both EBE mass res inputs
# MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/"

# # Removed both EBE mass res inputs
# MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBE/"

# # With class weights only to fix the imbalance in training
# MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBEv2/"

# # No EBE mass res inputs, With sample weights to fix the imbalance in training (sample weight takes class weight into account)
# MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBEv2_SampleWgt/"

# # With class weight and with relative EBE mass res as input
# MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_relativeEBEonly/"

# # Without class weight and with relative EBE mass res as input
# MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_relativeEBEonly_NoClassWgt/"

# # With class weight and with absolute EBE mass res and relative EBE mass res as input
# MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_BothEBE_WithClassWgt/"


# # With class weight and both relative and absolute EBE mass res as input
MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/Run2_DNN_BothEBE_WithClassWgt/"

# # With class weight and only relative EBE mass res as input
MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/Run2_DNN_relEBE_WithClassWgt/"

SCALER_NPZ = f"{MODEL_DIR}/scaler.npz"
MODEL_PATH = f"{MODEL_DIR}/model.keras"
OUT_DIR = f"{MODEL_DIR}/tag_fractions"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional secondary input with event weights. When left as None, the code will
# attempt to infer a matching directory that still retains the nominal weights
# (useful when working with the lightweight skimmed parquet files that dropped
# them).
WEIGHT_INPUT_DIR = None

# Columns that are safe to use when building a hashing key to align skimmed
# events with the original files that still contain weights.
WEIGHT_JOIN_COLUMNS = [
    "mu1_pt_over_mass",
    "mu2_pt_over_mass",
    "mu1_eta",
    "mu2_eta",
    "dimuon_mass",
    "dimuon_pt",
    "dimuon_pt_log",
    "jj_mass_nominal",
    "jj_dEta_nominal",
    "nBtagLoose_nominal",
    "nBtagMedium_nominal",
]

# Floating point precision (decimal places) used while building join keys that
# mix skimmed features with the weight-bearing parquet files.
WEIGHT_KEY_DECIMALS = 6

# Processes to read  (second item is a BOOL for your selection)
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

# ===== Variables & cuts to scan =====
# (df column name,  x-axis title,         (nbins, xmin, xmax))
VAR_SPECS = [
    ("njets_nominal", "Number of Jets (nominal)", (10, 0, 10)),
    # ("mu1_eta", "#eta(#mu_{1})", (40, -5, 5)),
    # ("mu2_eta", "#eta(#mu_{2})", (40, -5, 5)),
    # ("mu1_pt", "p_{T}(#mu_{1})", (40, 0, 200)),
    # ("mu2_pt", "p_{T}(#mu_{2})", (40, 0, 200)),
    ("dimuon_mass", "m_{#mu#mu} [GeV]", (51, 115, 135)),
    # ("dimuon_pt", "p_{T}(#mu#mu) [GeV]", (40, 0, 200)),
    # ("dimuon_rapidity", "y(#mu#mu)", (40, -5, 5)),
    # ("dimuon_eta", "#eta(#mu#mu)", (40, -5, 5)),
    ("dimuon_ebe_mass_res", "Event-by-event mass resolution [GeV]", (40, 0, 10)),
    ("dimuon_ebe_mass_res_rel", "Relative event-by-event mass resolution", (40, 0, 0.1)),
    # ("jj_mass_nominal", "m_{jj} [GeV]", (40, 0, 2000)),
    # ("jj_dEta_nominal", "#Delta#eta_{jj}", (40, 0, 10)),
]

DNN_CUTS = [0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95]
SCORE_COL = "score_vbf"    # change to "score_ggh_over_sigbkg" if desired
PLOT_COMPLEMENT = True
NORM_TO_UNIT_AREA = True
WRITE_ROOT_FILE = True
# ===================================================


def _infer_weight_dir_from_input(input_dir: str, override: Optional[str]) -> Optional[str]:
    """Resolve the directory that still retains event weights.

    When the evaluation runs on the lightweight skimmed parquet files (which
    only carry the features needed for the DNN), we need to recover the
    nominal weights from the original production parquet files. This helper
    tries to locate them automatically, but also respects a user-specified
    override.
    """

    if override:
        return override

    norm_in = os.path.normpath(input_dir)
    year_token = os.path.basename(norm_in)
    parent = os.path.basename(os.path.dirname(norm_in))
    root = os.path.dirname(os.path.dirname(norm_in))

    if parent != "skimmed_for_dnn":
        return None

    # Search a few known stage1 subdirectories (ordered by preference)
    stage1_root = os.path.join(
        root,
        "copperheadV1clean",
        "Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt",
        "stage1_output",
        year_token,
    )

    candidate_names = (
        "compacted",
        "compacted_19September_FixDimuonMass",
        "compacted_03September_FixDimuonMass",
        "compacted_13August_FixDimuonMass",
    )

    for name in candidate_names:
        candidate = os.path.join(stage1_root, name)
        if os.path.isdir(candidate):
            return candidate

    return None


WEIGHT_DATA_DIR = _infer_weight_dir_from_input(INPUT_DIR, WEIGHT_INPUT_DIR)
if WEIGHT_DATA_DIR and not os.path.isdir(WEIGHT_DATA_DIR):
    print(
        f"[yellow]WARN[/] Weight directory '{WEIGHT_DATA_DIR}' is not accessible; "
        "disabling external weight recovery."
    )
    WEIGHT_DATA_DIR = None


def load_features_from_json(path):
    with open(path, "r") as f:
        items = json.load(f).items()
    return [k for k, _ in items if k not in ("target", "process_ID", "classweight")]

# ----------- ROOT helpers -----------
import ROOT as R
R.gROOT.SetBatch(True)
R.gStyle.SetOptStat(0)

def _make_hist(name, title, nbins, xmin, xmax):
    h = R.TH1F(name, title, nbins, xmin, xmax)
    h.Sumw2()
    return h

def _fill_hist_from_np(hist, arr):
    import math
    for v in arr:
        try:
            fv = float(v)
            if math.isfinite(fv):
                hist.Fill(fv)
        except Exception:
            pass

def _style_hist(h, color, marker, width=2):
    h.SetLineColor(color)
    h.SetMarkerColor(color)
    h.SetLineWidth(width)
    h.SetMarkerStyle(marker)

def _normalize(h):
    if h.Integral() > 0:
        h.Scale(1.0 / h.Integral())

def _save_canvas(c, pdf_path):
    c.SaveAs(pdf_path)


def make_variable_plots_for_cuts(df, sample_name,
                                 var_specs=VAR_SPECS,
                                 cuts=DNN_CUTS,
                                 score_col=SCORE_COL,
                                 plot_complement=PLOT_COMPLEMENT,
                                 norm_to_unit=NORM_TO_UNIT_AREA,
                                 write_root=WRITE_ROOT_FILE,
                                 out_dir=OUT_DIR):
    fout = None
    if write_root:
        root_path = os.path.join(out_dir, f"{sample_name}_shapes.root")
        fout = R.TFile(root_path, "RECREATE")

    if score_col not in df.columns:
        print(f"[yellow]WARN[/] score column '{score_col}' not found for {sample_name}; skipping plots.")
        if fout:
            fout.Close()
        return

    score = df[score_col].to_numpy()

    w = (
        df["wgt_nominal"].to_numpy()
        if "wgt_nominal" in df.columns
        else np.ones_like(score, dtype=np.float64)
    )

    for (vname, xtitle, (nb, xmin, xmax)) in var_specs:
        if vname not in df.columns:
            print(f"[yellow]WARN[/] variable '{vname}' not found for {sample_name}; skipping {vname}.")
            continue

        var = df[vname].to_numpy()

        for cut in cuts:
            mask_pass = score > cut      # "VBF-like"
            mask_fail = score <= cut     # "ggH-like"

            tag = str(cut).replace(".", "p")
            base = f"{sample_name}_{vname}_cut_{tag}"

            h_all  = _make_hist(f"h_all__{base}",  "", nb, xmin, xmax)
            h_pass = _make_hist(f"h_pass__{base}", "", nb, xmin, xmax)
            h_fail = _make_hist(f"h_fail__{base}", "", nb, xmin, xmax)

            # _fill_hist_from_np(h_all,  var)
            # _fill_hist_from_np(h_pass, var[mask_pass])
            # _fill_hist_from_np(h_fail, var[mask_fail])

            _fill_hist_from_np_w(h_all,  var,             w)
            _fill_hist_from_np_w(h_pass, var[mask_pass],  w[mask_pass])
            _fill_hist_from_np_w(h_fail, var[mask_fail],  w[mask_fail])


            # if norm_to_unit:
            #     _normalize(h_all);
            #     _normalize(h_pass);
            #     _normalize(h_fail)

            if norm_to_unit:
                _normalize_safe(h_all);
                _normalize_safe(h_pass);
                _normalize_safe(h_fail)

            # Style
            _style_hist(h_all, R.kBlack, 20)
            _style_hist(h_pass, R.kBlue,     24)
            _style_hist(h_fail, R.kGreen+2,  25)

            # Axes
            h_all.GetXaxis().SetTitle(xtitle)
            h_all.GetYaxis().SetTitle("Normalized entries" if norm_to_unit else "Entries")
            h_all.GetYaxis().SetNdivisions(505)

            # Canvas & draw
            c = R.TCanvas(f"c_{base}", f"c_{base}", 800, 650)
            c.SetMargin(0.12, 0.04, 0.12, 0.06)
            ymax = max(h_all.GetMaximum(), h_pass.GetMaximum(), h_fail.GetMaximum()) * 1.25
            h_all.SetMaximum(ymax)

            h_all.Draw("HIST")
            h_pass.Draw("HIST SAME")
            if plot_complement:
                h_fail.Draw("HIST SAME")

            # Legend
            leg = R.TLegend(0.50, 0.72, 0.95, 0.89)
            leg.SetBorderSize(0); leg.SetFillStyle(0); leg.SetTextSize(0.045)
            leg.AddEntry(h_all,  "All events", "l")
            leg.AddEntry(h_pass, f"{score_col} > {cut}", "l")
            if plot_complement:
                leg.AddEntry(h_fail, f"{score_col} #le {cut}", "l")
            leg.Draw()

            # Label
            latex = R.TLatex()
            latex.SetNDC(True)
            latex.SetTextSize(0.040)
            latex.DrawLatex(0.14, 0.91, f"{sample_name}  |  cut: {score_col} > {cut}")

            # Save
            pdf_name = f"{sample_name}_{vname}_{score_col}_gt_{tag}.pdf"
            _save_canvas(c, os.path.join(out_dir, pdf_name))

            if write_root:
                fout.cd()
                h_all.Write()
                h_pass.Write()
                if plot_complement:
                    h_fail.Write()
                c.Write()

            c.Close()
            R.gDirectory.Delete(f"{h_all.GetName()};*")
            R.gDirectory.Delete(f"{h_pass.GetName()};*")
            R.gDirectory.Delete(f"{h_fail.GetName()};*")

    if fout:
        fout.Close()
        print(f"[green]Wrote ROOT histograms[/] → {root_path}")


def _fill_hist_from_np_w(hist, arr, weights):
    import math
    if weights is None:
        _fill_hist_from_np(hist, arr)
        return
    for v, w in zip(arr, weights):
        try:
            fv = float(v)
            fw = float(w)
            if math.isfinite(fv) and math.isfinite(fw):
                hist.Fill(fv, fw)
        except Exception:
            pass

def _normalize_safe(h):
    integ = h.Integral()
    # With NLO samples you can get small/negative sums; only normalize if positive.
    if integ > 0:
        h.Scale(1.0 / integ)


def _ak_to_numpy_column(array: ak.Array, name: str, default: float = np.nan) -> np.ndarray:
    """Safely convert an Awkward column into a dense numpy array."""

    if name in getattr(array, "fields", []):
        try:
            return ak.to_numpy(array[name])
        except Exception:
            return np.asarray(ak.flatten(array[name], axis=None))

    return np.full(len(array), default, dtype=np.float64)


def _discover_weight_files(base_dir: Optional[str], sample: str) -> List[str]:
    if not base_dir:
        return []

    patterns = (
        os.path.join(base_dir, sample, "*.parquet"),
        os.path.join(base_dir, sample, "*", "*.parquet"),
        os.path.join(base_dir, sample, "**", "*.parquet"),
    )

    for pat in patterns:
        files = glob.glob(pat, recursive="**" in pat)
        if files:
            return sorted(files)

    return []


def _make_join_key(df: pd.DataFrame, columns: List[str], decimals: int) -> np.ndarray:
    if df.empty:
        return np.empty(0, dtype=np.uint64)

    if not columns:
        raise ValueError("At least one column is required to build a join key")

    aligned = []
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Join column '{col}' missing while aligning weights")
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.floating):
            values = np.round(values.astype(np.float64, copy=False), decimals)
        aligned.append(values)

    stacked = np.column_stack(aligned)
    hashed = pd.util.hash_pandas_object(
        pd.DataFrame(stacked), index=False, categorize=False
    )
    return hashed.to_numpy(dtype=np.uint64)


def _needs_weight_recovery(sample: str, weights: pd.Series) -> bool:
    if "data" in sample.lower():
        return False
    if weights is None or weights.empty:
        return True
    if weights.isna().all():
        return True
    valid = weights.dropna()
    if valid.empty:
        return True
    return np.allclose(valid.to_numpy(), 1.0)


def _load_weight_frame(
    sample: str,
    tag: str,
    vbf_filter: bool,
    read_columns: List[str],
) -> Optional[pd.DataFrame]:
    files = _discover_weight_files(WEIGHT_DATA_DIR, sample)
    if not files:
        return None

    cols = list(dict.fromkeys(read_columns + ["wgt_nominal"]))

    try:
        ddf = dak.from_parquet(files, columns=cols)
    except Exception as exc:
        print(
            f"[yellow]WARN[/] {sample}: failed to read weight columns selectively "
            f"({exc}); reading full schema instead."
        )
        ddf = dak.from_parquet(files)

    ddf_sel = selection.applyRegionCatCuts(
        ddf,
        category=tag,
        region_name="h-peak",
        process=sample,
        variation="nominal",
        do_vbf_filter_study=vbf_filter,
    )

    ak_weight = ddf_sel.compute()
    if len(ak_weight) == 0:
        return pd.DataFrame(columns=cols)

    keep_cols = [c for c in WEIGHT_JOIN_COLUMNS if c in ak_weight.fields]
    data = {c: _ak_to_numpy_column(ak_weight, c) for c in keep_cols}
    data["wgt_nominal"] = _ak_to_numpy_column(ak_weight, "wgt_nominal", default=1.0)
    return pd.DataFrame(data)


def _merge_weights(
    sample: str,
    df_eval: pd.DataFrame,
    weight_df: pd.DataFrame,
    decimals: int,
) -> pd.DataFrame:
    join_cols = [
        c for c in WEIGHT_JOIN_COLUMNS if c in df_eval.columns and c in weight_df.columns
    ]

    if not join_cols:
        print(
            f"[yellow]WARN[/] {sample}: unable to find common columns to align "
            "weights; keeping unity weights."
        )
        return df_eval

    key_eval = _make_join_key(df_eval, join_cols, decimals)
    key_weight = _make_join_key(weight_df, join_cols, decimals)
    weight_values = weight_df["wgt_nominal"].to_numpy(dtype=np.float64, copy=False)

    weight_map: dict[int, float] = {}
    duplicates = 0
    for key, value in zip(key_weight, weight_values):
        if key in weight_map and not np.isclose(weight_map[key], value, rtol=1e-6, atol=1e-8):
            duplicates += 1
        weight_map.setdefault(key, value)

    recovered = np.array([weight_map.get(k, np.nan) for k in key_eval], dtype=np.float64)
    missing = np.count_nonzero(~np.isfinite(recovered))

    if missing:
        print(
            f"[yellow]WARN[/] {sample}: missing weights for {missing} events; "
            "defaulting those to 1.0"
        )
        recovered[~np.isfinite(recovered)] = 1.0

    if duplicates:
        print(
            f"[yellow]WARN[/] {sample}: encountered {duplicates} duplicate join keys "
            "while merging weights; kept the first occurrence."
        )

    df_eval["wgt_nominal"] = recovered
    return df_eval


def ensure_weight_column(
    sample: str,
    tag: str,
    vbf_filter: bool,
    read_columns: List[str],
    df_eval: pd.DataFrame,
) -> pd.DataFrame:
    weights = df_eval.get("wgt_nominal", pd.Series(dtype=np.float64))
    if WEIGHT_DATA_DIR is None:
        return df_eval

    if not _needs_weight_recovery(sample, weights):
        return df_eval

    weight_df = _load_weight_frame(sample, tag, vbf_filter, read_columns)
    if weight_df is None or weight_df.empty:
        print(
            f"[yellow]WARN[/] {sample}: could not recover event weights; keeping unity values."
        )
        return df_eval

    print(
        f"[blue]INFO[/] {sample}: recovered weights from {WEIGHT_DATA_DIR}"
    )
    return _merge_weights(sample, df_eval, weight_df, WEIGHT_KEY_DECIMALS)

# ================== MAIN ==================
if __name__ == "__main__":
    if WEIGHT_DATA_DIR:
        print(f"[blue]INFO[/] Weight recovery directory: {WEIGHT_DATA_DIR}")
    else:
        print(
            "[cyan]INFO[/] No auxiliary weight directory configured; "
            "will rely on the weights stored alongside the evaluation parquet files."
        )

    print("[bold]Loading model…[/]")
    model = load_model(MODEL_PATH)

    sc_npz = np.load(SCALER_NPZ)
    mean = sc_npz.get("mean_", sc_npz.get("mean"))
    scale = sc_npz.get("scale_", sc_npz.get("scale"))
    if mean is None or scale is None:
        raise RuntimeError(f"Scaler npz missing mean_/scale_: {SCALER_NPZ}")
    mean = mean.astype(np.float64)
    scale = scale.astype(np.float64)
    safe_scale = np.where(scale == 0, 1.0, scale)

    FEATURES = load_features_from_json(FEATURES_JSON)

    extra_from_vars = [v[0] for v in VAR_SPECS]
    EXTRA_KEEP = list(dict.fromkeys(
        extra_from_vars + [
            "year",
            "njets_nominal",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            "gjj_mass",
            "jj_mass_nominal",
            "dimuon_mass",
            "dimuon_pt",
            "wgt_nominal",
        ]
    ))

    score_dict = {0: "ggh", 1: "vbf", 2: "bkg"}

    # Accumulate cutflow rows across all samples & cuts
    cutflow_rows = []

    for sample, (tag, vbf_filter_bool) in SAMPLES.items():
        pat = os.path.join(INPUT_DIR, sample, "*.parquet")
        print(f"[bold blue]Processing[/] {sample}  (pattern: {pat})")

        read_cols = list(
            dict.fromkeys(
                FEATURES + EXTRA_KEEP + WEIGHT_JOIN_COLUMNS + ["process_ID"]
            )
        )
        try:
            ddf = dak.from_parquet(pat, columns=read_cols)
        except Exception as e:
            print(f"[yellow]WARN[/] {sample}: selective column read failed ({e}); reading all columns.")
            ddf = dak.from_parquet(pat)

        ddf_sel_skim = selection.applyRegionCatCuts(
            ddf, category=tag, region_name="h-peak",
            process=sample, variation="nominal",
            do_vbf_filter_study=vbf_filter_bool,
        )

        ak_array = ddf_sel_skim.compute()

        df_eval = pd.DataFrame(
            {var: _ak_to_numpy_column(ak_array, var) for var in FEATURES}
        )
        for col_name in EXTRA_KEEP:
            if col_name not in df_eval.columns:
                df_eval[col_name] = _ak_to_numpy_column(ak_array, col_name)

        if "wgt_nominal" in df_eval.columns:
            df_eval["wgt_nominal"] = pd.to_numeric(
                df_eval["wgt_nominal"], errors="coerce"
            )
        else:
            df_eval["wgt_nominal"] = np.nan

        df_eval = ensure_weight_column(
            sample=sample,
            tag=tag,
            vbf_filter=vbf_filter_bool,
            read_columns=read_cols,
            df_eval=df_eval,
        )

        df_eval["wgt_nominal"] = (
            pd.to_numeric(df_eval["wgt_nominal"], errors="coerce")
            .fillna(1.0)
            .astype(np.float64)
        )

        df_eval[FEATURES] = (
            df_eval[FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float64)
        )
        X = df_eval[FEATURES].to_numpy(copy=False)
        X = (X - mean) / safe_scale
        X = np.nan_to_num(X, copy=False, posinf=0.0, neginf=0.0)

        scores = model.predict(X, batch_size=4096, verbose=0)

        if scores.ndim == 1:
            df_eval["score"] = scores
        else:
            for i in range(scores.shape[1]):
                df_eval[f"score_{score_dict.get(i, str(i))}"] = scores[:, i]

        # Optional composite discriminant:
        if {"score_ggh", "score_bkg"}.issubset(set(df_eval.columns)):
            df_eval["score_ggh_over_sigbkg"] = df_eval["score_ggh"] / (
                df_eval["score_ggh"] + df_eval["score_bkg"] + 1e-12
            )

        # Save per-event parquet
        out_path = os.path.join(OUT_DIR, f"{sample}_scores.parquet")
        print(f"Output columns: {sorted(df_eval.columns.tolist())}")
        df_eval.to_parquet(out_path, index=False)
        print(f"[green]Saved[/] → {out_path}")

        # -------- CUT-FLOW TABLE (per cut) --------
        if SCORE_COL not in df_eval.columns:
            print(f"[yellow]WARN[/] score column '{SCORE_COL}' missing; skipping cut-flow for {sample}.")
        else:
            score = df_eval[SCORE_COL].to_numpy()
            w = df_eval["wgt_nominal"].to_numpy()
            n_selected = len(df_eval)
            w_selected = float(w.sum())

            for cut in DNN_CUTS:
                mask_vbf = score > cut
                mask_ggh = ~mask_vbf
                n_vbf = int(mask_vbf.sum())
                n_ggh = int(n_selected - n_vbf)

                # Weighted counts (normalized to x-sec via wgt_nominal)
                w_vbf = float(w[mask_vbf].sum())
                w_ggh = float(w[mask_ggh].sum())

                frac_vbf = (n_vbf / n_selected * 100.0) if n_selected else 0.0
                frac_ggh = (n_ggh / n_selected * 100.0) if n_selected else 0.0
                frac_vbf_w = (w_vbf / w_selected * 100.0) if w_selected > 0 else 0.0
                frac_ggh_w = (w_ggh / w_selected * 100.0) if w_selected > 0 else 0.0

                # Print like your example
                print(
                    f"Sample: {sample:<22}, cut {SCORE_COL:>16} > {cut:>4.2f}, "
                    f"UNW: n_sel:{n_selected:9d}, n_ggh:{n_ggh:9d}({frac_ggh:6.2f}%), n_vbf:{n_vbf:9d}({frac_vbf:6.2f}%)"
                )
                print(
                    f"{'':<22}  {'':>16}   {'':>7}  "
                    f"WGT: w_sel:{w_selected:9.2f}, w_ggh:{w_ggh:9.2f}({frac_ggh_w:6.2f}%), w_vbf:{w_vbf:9.2f}({frac_vbf_w:6.2f}%)"
                )

                # Save row to CSV accumulator
                cutflow_rows.append(
                    {
                        "sample": sample,
                        "cut_var": SCORE_COL,
                        "cut_threshold": cut,
                        # unweighted
                        "n_selected": n_selected,
                        "n_ggh_like": n_ggh,
                        "frac_ggh_like_pct": frac_ggh,
                        "n_vbf_like": n_vbf,
                        "frac_vbf_like_pct": frac_vbf,
                        # weighted (x-sec normalized)
                        "w_selected": w_selected,
                        "w_ggh_like": w_ggh,
                        "w_frac_ggh_like_pct": frac_ggh_w,
                        "w_vbf_like": w_vbf,
                        "w_frac_vbf_like_pct": frac_vbf_w,
                    }
                )

        # -------- ROOT plots across many vars & cuts --------
        make_variable_plots_for_cuts(
            df_eval,
            sample_name=sample,
            var_specs=VAR_SPECS,
            cuts=DNN_CUTS,
            score_col=SCORE_COL,   # switch to "score_ggh_over_sigbkg" if desired
            plot_complement=PLOT_COMPLEMENT,
            norm_to_unit=NORM_TO_UNIT_AREA,
            write_root=WRITE_ROOT_FILE,
            out_dir=OUT_DIR,
        )

    # -------- Save combined cut-flow CSV --------
    if cutflow_rows:
        cutflow_df = pd.DataFrame(cutflow_rows)
        cutflow_csv = os.path.join(OUT_DIR, "cutflow_summary.csv")
        cutflow_df.to_csv(cutflow_csv, index=False)
        print(f"[bold green]Cut-flow summary saved[/] → {cutflow_csv}")

    print("[bold green]Done.[/]")
