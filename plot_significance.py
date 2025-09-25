# plot_significance.py
import ROOT
from array import array

# --- Data ---
methods = [
    "Cut-based",
    "DNN > 0.90",
    "DNN > 0.85",
    "DNN > 0.80",
    "DNN > 0.75",
    "DNN > 0.50",
    "DNN > 0.30",
]
significance = [390.74, 527.38, 604.42, 657.42, 696.50, 790.42, 796.73]
vbf_frac = [40.20, 26.73, 35.98, 43.63, 50.23, 74.98, 87.58]
dy_frac = [34.15, 0.49, 1.05, 1.75, 2.58, 8.92, 16.86]

# numeric x positions for categories
x = array("d", [float(i) for i in range(len(methods))])
y_sig = array("d", significance)
y_vbf = array("d", vbf_frac)
y_dy = array("d", dy_frac)
n = len(methods)

# --- Canvas/style ---
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("c", "Selection Method vs Significance", 900, 600)
c.SetBottomMargin(0.25)  # leave room for rotated labels
c.SetGridy()
c.SetGridx()

# --- Frame with labeled bins on X (categorical axis) ---
frame = ROOT.TH1F("frame", ";;Significance(S/#sqrt{S+B})", n, -0.5, n - 0.5)
for i, lab in enumerate(methods):
    frame.GetXaxis().SetBinLabel(i + 1, lab)
frame.GetXaxis().LabelsOption("v")  # vertical labels (use "h" for horizontal)
frame.GetXaxis().SetLabelSize(0.04)
frame.GetYaxis().SetTitleOffset(1.1)
frame.SetMinimum(min(significance) * 0.95)
frame.SetMaximum(max(significance) * 1.05)
frame.SetLineColor(0)
frame.Draw("AXIS")  # draw axes only with labels

# --- Graph ---
g_sig = ROOT.TGraph(n, x, y_sig)
g_sig.SetMarkerStyle(20)
g_sig.SetMarkerSize(1.2)
g_sig.SetLineWidth(2)
g_sig.SetLineColor(ROOT.kBlue + 1)
g_sig.SetMarkerColor(ROOT.kBlue + 1)
g_sig.Draw("LP SAME")

# --- Scale fractions to right axis ---
right_min, right_max = 0, max(max(vbf_frac), max(dy_frac)) * 1.2
left_min, left_max = frame.GetMinimum(), frame.GetMaximum()


def scale(val):
    return left_min + (val - right_min) / (right_max - right_min) * (
        left_max - left_min
    )


y_vbf_scaled = array("d", [scale(v) for v in vbf_frac])
y_dy_scaled = array("d", [scale(v) for v in dy_frac])

g_vbf = ROOT.TGraph(n, x, y_vbf_scaled)
g_vbf.SetMarkerStyle(21)
g_vbf.SetMarkerColor(ROOT.kRed)
g_vbf.SetLineColor(ROOT.kRed)
g_vbf.SetLineWidth(2)
g_vbf.Draw("LP SAME")

g_dy = ROOT.TGraph(n, x, y_dy_scaled)
g_dy.SetMarkerStyle(22)
g_dy.SetMarkerColor(ROOT.kGreen + 2)
g_dy.SetLineColor(ROOT.kGreen + 2)
g_dy.SetLineWidth(2)
g_dy.Draw("LP SAME")

# --- Right axis ---
axis = ROOT.TGaxis(
    n - 0.5, left_min, n - 0.5, left_max, right_min, right_max, 510, "+L"
)
axis.SetTitle("VBF and DY_VBF Fractions (%)")
axis.SetTitleOffset(1.2)
axis.SetTitleSize(0.04)
axis.SetLabelSize(0.035)
axis.Draw()

# --- Legend ---
leg = ROOT.TLegend(0.15, 0.75, 0.45, 0.9)
leg.AddEntry(g_sig, "Significance", "lp")
leg.AddEntry(g_vbf, "VBF signal fraction", "lp")
leg.AddEntry(g_dy, "DY_VBF filter fraction", "lp")
leg.Draw()


# --- Save ---
c.SaveAs("significance_plot.pdf")
c.SaveAs("significance_plot.png")
print("Saved significance_plot.pdf and .png")
