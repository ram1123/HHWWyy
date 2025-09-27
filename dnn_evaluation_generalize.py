#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import json
import numpy as np
import pandas as pd
import awkward as ak
import dask_awkward as dak
from tensorflow.keras.models import load_model
from rich import print

import selection  # your module

# ================== USER CONFIG ==================
INPUT_DIR = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
FEATURES_JSON = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"

# With class weights only to fix the imbalance in training
MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBEv2/"
# With sample weights (optional alternative)
# MODEL_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_Removed_EBEv2_SampleWgt/"

SCALER_NPZ = f"{MODEL_DIR}/scaler.npz"
MODEL_PATH = f"{MODEL_DIR}/model.keras"
OUT_DIR = f"{MODEL_DIR}/tag_fractions_NewCode"
os.makedirs(OUT_DIR, exist_ok=True)

# Processes to read  (second item is a BOOL)
SAMPLES = {
    "ggh_powhegPS": ("notbtag", False),
    "vbf_powheg_dipole": ("notbtag", False),
    "dy_VBF_filter": ("notbtag", True),
    "dy_M-100To200_MiNNLO": ("notbtag", True),
    "dy_M-50_MiNNLO": ("notbtag", True),
    "ewk_lljj_mll50_mjj120": ("notbtag", False),
    "ttjets_dl": ("notbtag", False),
    "ttjets_sl": ("notbtag", False),
}

# ===== Variables & cuts to scan (add more here) =====
# (df column name,  x-axis title,         (nbins, xmin, xmax))
VAR_SPECS = [
    ("njets_nominal", "Number of Jets (nominal)", (10, 0, 10)),
    ("mu1_eta", "#eta(#mu_{1})", (40, -5, 5)),
    ("mu2_eta", "#eta(#mu_{2})", (40, -5, 5)),
    ("mu1_pt", "p_{T}(#mu_{1})", (40, 0, 200)),
    ("mu2_pt", "p_{T}(#mu_{2})", (40, 0, 200)),
    ("dimuon_rapidity", "y(#mu#mu)", (40, -5, 5)),
    ("dimuon_ebe_mass_res", "Event-by-event mass resolution [GeV]", (40, 0, 10)),
    ("dimuon_ebe_mass_res_rel", "Relative event-by-event mass resolution", (40, 0, 0.1)),
    ("dimuon_mass", "m_{#mu#mu} [GeV]", (51, 115, 135)),
    ("dimuon_pt", "p_{T}(#mu#mu) [GeV]", (40, 0, 200)),
    ("dimuon_eta", "#eta(#mu#mu)", (40, -5, 5)),
    ("jj_mass_nominal", "m_{jj} [GeV]", (40, 0, 2000)),
    ("jj_dEta_nominal", "#Delta#eta_{jj}", (40, 0, 10)),
]

# Thresholds to scan on the chosen discriminant
DNN_CUTS = [0.50, 0.65, 0.75, 0.80, 0.85, 0.90]

# Which discriminant to cut on by default
SCORE_COL = "score_vbf"  # change to "score_ggh_over_sigbkg" after it's created below to use composite score
PLOT_COMPLEMENT = True   # also draw the <=cut (ggH-like) distribution
NORM_TO_UNIT_AREA = True
WRITE_ROOT_FILE = True
# ===================================================


def load_features_from_json(path):
    with open(path, "r") as f:
        items = json.load(f).items()
    # keep exact training order, drop non-features if present
    return [k for k, _ in items if k not in ("target", "process_ID", "classweight")]


# ----------- ROOT helpers (for plotting) -----------
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
    # Prepare ROOT file if requested
    fout = None
    if write_root:
        root_path = os.path.join(out_dir, f"{sample_name}_shapes.root")
        fout = R.TFile(root_path, "RECREATE")

    # Numpy views for masks
    if score_col not in df.columns:
        print(f"[yellow]WARN[/] score column '{score_col}' not found for {sample_name}; skipping plots.")
        if fout:
            fout.Close()
        return

    score = df[score_col].to_numpy()

    for (vname, xtitle, (nb, xmin, xmax)) in var_specs:
        if vname not in df.columns:
            print(f"[yellow]WARN[/] variable '{vname}' not found for {sample_name}; skipping.")
            continue

        var = df[vname].to_numpy()

        for cut in cuts:
            mask_pass = score > cut
            mask_fail = score <= cut

            tag = str(cut).replace(".", "p")
            base = f"{sample_name}_{vname}_cut_{tag}"

            h_all  = _make_hist(f"h_all__{base}",  "", nb, xmin, xmax)
            h_pass = _make_hist(f"h_pass__{base}", "", nb, xmin, xmax)
            h_fail = _make_hist(f"h_fail__{base}", "", nb, xmin, xmax)

            _fill_hist_from_np(h_all,  var)
            _fill_hist_from_np(h_pass, var[mask_pass])
            _fill_hist_from_np(h_fail, var[mask_fail])

            if norm_to_unit:
                _normalize(h_all); _normalize(h_pass); _normalize(h_fail)

            # Style
            _style_hist(h_all,  R.kBlack,    20)
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
            latex.SetNDC(True); latex.SetTextSize(0.040)
            latex.DrawLatex(0.14, 0.91, f"{sample_name}  |  cut: {score_col} > {cut}")

            # Save
            pdf_name = f"{sample_name}_{vname}_{score_col}_gt_{tag}.pdf"
            _save_canvas(c, os.path.join(out_dir, pdf_name))

            # Optionally write to ROOT file
            if write_root:
                fout.cd()
                h_all.Write(); h_pass.Write()
                if plot_complement: h_fail.Write()
                c.Write()

            # Cleanup
            c.Close()
            R.gDirectory.Delete(f"{h_all.GetName()};*")
            R.gDirectory.Delete(f"{h_pass.GetName()};*")
            R.gDirectory.Delete(f"{h_fail.GetName()};*")

    if fout:
        fout.Close()
        print(f"[green]Wrote ROOT histograms[/] → {root_path}")


# ================== MAIN ==================
if __name__ == "__main__":
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

    # Extra physics columns to preserve for output/plots
    # Ensure we keep everything referenced in VAR_SPECS
    extra_from_vars = [v[0] for v in VAR_SPECS]
    EXTRA_KEEP = list(dict.fromkeys(
        extra_from_vars + [
            "year",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            "gjj_mass",
            "jj_mass_nominal",
            "dimuon_mass",
            "dimuon_pt",
            "njets_nominal",
        ]
    ))

    # Dict for class ordering in your model (ggh:0, vbf:1, bkg:2)
    score_dict = {0: "ggh", 1: "vbf", 2: "bkg"}

    for sample, (tag, vbf_filter_bool) in SAMPLES.items():
        pat = os.path.join(INPUT_DIR, sample, "*.parquet")
        print(f"[bold blue]Processing[/] {sample}  (pattern: {pat})")

        # Prefer reading only needed columns
        read_cols = list(dict.fromkeys(FEATURES + EXTRA_KEEP + ["process_ID"]))  # dedup
        try:
            ddf = dak.from_parquet(pat, columns=read_cols)
        except Exception as e:
            print(f"[yellow]WARN[/] {sample}: selective column read failed ({e}); reading all columns.")
            ddf = dak.from_parquet(pat)

        # Apply your region/category selections
        ddf_sel_skim = selection.applyRegionCatCuts(
            ddf,
            category=tag,
            region_name="h-peak",
            process=sample,
            variation="nominal",
            do_vbf_filter_study=vbf_filter_bool,
        )

        # Bring to memory as Awkward
        ak_array = ddf_sel_skim.compute()

        # Helper: Awkward column -> flat numpy (OptionType -> NaN)
        def col_to_np(name):
            if name in ak_array.fields:
                try:
                    return ak.to_numpy(ak_array[name])
                except Exception:
                    return np.asarray(ak.flatten(ak_array[name], axis=None))
            # missing column -> NaNs
            return np.full(len(ak_array), np.nan, dtype=np.float64)

        # Build DataFrame with features
        df_eval = pd.DataFrame({var: col_to_np(var) for var in FEATURES})
        # Add extra columns if present
        for c in EXTRA_KEEP:
            if c not in df_eval.columns:
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

        # Attach per-class scores
        if scores.ndim == 1:
            df_eval["score"] = scores
        else:
            for i in range(scores.shape[1]):
                df_eval[f"score_{score_dict.get(i, str(i))}"] = scores[:, i]

        # Optional composite discriminants (uncomment to use)
        # More calibrated ggH-vs-bkg-like score:
        if {"score_ggh", "score_bkg"}.issubset(set(df_eval.columns)):
            df_eval["score_ggh_over_sigbkg"] = df_eval["score_ggh"] / (
                df_eval["score_ggh"] + df_eval["score_bkg"] + 1e-12
            )

        # Save per-event scores parquet
        out_path = os.path.join(OUT_DIR, f"{sample}_scores.parquet")
        print(f"Output columns: {sorted(df_eval.columns.tolist())}")
        df_eval.to_parquet(out_path, index=False)
        print(f"[green]Saved[/] → {out_path}")

        # ========= Plot many variables for many cuts (ROOT) =========
        make_variable_plots_for_cuts(
            df_eval,
            sample_name=sample,
            var_specs=VAR_SPECS,
            cuts=DNN_CUTS,
            score_col=SCORE_COL,         # set to "score_ggh_over_sigbkg" if desired
            plot_complement=PLOT_COMPLEMENT,
            norm_to_unit=NORM_TO_UNIT_AREA,
            write_root=WRITE_ROOT_FILE,
            out_dir=OUT_DIR,
        )

    print("[bold green]Done.[/]")
