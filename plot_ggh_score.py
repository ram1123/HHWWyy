#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import ROOT as R

from rich import print

R.gROOT.SetBatch(True)
R.gStyle.SetOptStat(0)

IN_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
IN_DIR = "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_multiclass_fullStats_Scan_Quick/tag_fractions"
OUT_PDF = os.path.join(IN_DIR, "ggH_vs_Bkg_score_ggh_after_vbfCut.pdf")

# Define which samples are signal and which to exclude from background
SIGNAL_SAMPLES = {"ggh_powhegPS"}
EXCLUDE_FROM_BKG = {"vbf_powheg_dipole"}  # keep DY, EWK, tt, etc. as background

# Collect parquet files
files = sorted(glob.glob(os.path.join(IN_DIR, "*_scores.parquet")))
if not files:
    raise SystemExit(f"No parquet files found in: {IN_DIR}")

sig_vals, bkg_vals = [], []


def safe_series(df, name):
    s = df[name]
    # make sure numeric and finite
    return pd.to_numeric(s, errors="coerce")


for fp in files:
    name = os.path.basename(fp).replace("_scores.parquet", "")
    # print(f"Found file: {fp}")
    # print(f"Processing: {name}")
    if "ggh_powhegPS" not in name and "dy_VBF_filter" not in name:
        print(f"Skipping {name}, not signal or relevant background")
        continue
    print(f"Processing: {name}")

    # Read only needed columns for speed
    # df = pd.read_parquet(fp, columns=["score_ggh", "score_vbf", "score_bkg", "dimuon_mass"])
    df = pd.read_parquet(fp, columns=["score_ggh", "score_vbf", "score_bkg"])
    print(f"columns in dataframe: {list(df.columns)}")
    sg = safe_series(df, "score_ggh")
    sv = safe_series(df, "score_vbf")
    sb = safe_series(df, "score_bkg")
    # dm = safe_series(df, "dimuon_mass")

    # plot ggh/(ggh+bkg) after VBF veto
    # xtitle = "score_ggh"; savetag = "score_ggh"

    sg = sg / (sg + sb)
    xtitle = "score_ggh / (score_ggh + score_bkg)"; savetag = "score_ggHby_ggHplusbkg"

    OUT_PDF = OUT_PDF.replace("score_ggh", savetag)

    # Apply VBF veto: score_vbf <= 0.5
    mask = (sv <= 0.85) & np.isfinite(sg) & np.isfinite(sv)

    if name in SIGNAL_SAMPLES:
        sig_vals.append(sg[mask].to_numpy())
    elif name not in EXCLUDE_FROM_BKG:
        bkg_vals.append(sg[mask].to_numpy())

sig = np.concatenate(sig_vals) if sig_vals else np.array([], dtype=float)
bkg = np.concatenate(bkg_vals) if bkg_vals else np.array([], dtype=float)

print(f"Entries → signal: {sig.size}, background: {bkg.size}")

# ----------------- ROOT plotting -----------------
# Bin edges (0..1)
edges = np.linspace(0.0, 1.0, 51)
nbins = len(edges) - 1
h_sig = R.TH1F("h_sig", "", nbins, edges.astype(np.double))
h_bkg = R.TH1F("h_bkg", "", nbins, edges.astype(np.double))

for v in sig:
    h_sig.Fill(float(v))
for v in bkg:
    h_bkg.Fill(float(v))

# Normalize to unit area
if h_sig.Integral() > 0:
    h_sig.Scale(1.0 / h_sig.Integral())
if h_bkg.Integral() > 0:
    h_bkg.Scale(1.0 / h_bkg.Integral())

# Style
h_sig.SetLineColor(R.kRed + 1)
h_sig.SetFillColorAlpha(R.kRed - 9, 0.35)
h_sig.SetLineWidth(2)
h_bkg.SetLineColor(R.kBlue + 1)
h_bkg.SetFillColorAlpha(R.kBlue - 9, 0.35)
h_bkg.SetLineWidth(2)

h_bkg.GetXaxis().SetTitle(f"{xtitle} (score_vbf <= 0.5)")
# h_sig.GetYaxis().SetTitle("(1/N) dN/dx")
# h_sig.GetYaxis().SetNdivisions(505)

c = R.TCanvas("c", "c", 800, 600)
c.SetMargin(0.12, 0.04, 0.12, 0.06)

# Draw background first so signal overlays, or swap if you prefer
h_bkg.Draw("HIST")
h_sig.Draw("HIST SAME")

leg = R.TLegend(0.60, 0.72, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.AddEntry(h_sig, "Signal (ggH), score_vbf <= 0.5", "lf")
leg.AddEntry(h_bkg, "Background, score_vbf <= 0.5", "lf")
leg.Draw()

latex = R.TLatex()
latex.SetNDC(True)
latex.SetTextSize(0.04)
# latex.DrawLatex(0.14, 0.91, "Classifier Output: ggH vs Background")

c.SaveAs(OUT_PDF)

c.SetLogy()
c.SaveAs(OUT_PDF.replace(".pdf", "_logy.pdf"))
print(f"Saved → {OUT_PDF}")
c.Close()
