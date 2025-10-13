import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import awkward as ak
import dask.dataframe as dak
import selection

from rich import print

print("Loading plot_all_vars.py")

df = pd.read_parquet(
    "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2016preVFP/ggh_powhegPS/",
    # columns=[
    # "jet1_pt_nominal", "jet1_eta_nominal", "jet1_phi_nominal", "jet1_mass_nominal",
    # "jet2_pt_nominal", "jet2_eta_nominal", "jet2_phi_nominal", "jet2_mass_nominal",
    # "jet3_pt_nominal", "jet3_eta_nominal", "jet3_phi_nominal", "jet3_mass_nominal",
    # "jet4_pt_nominal", "jet4_eta_nominal", "jet4_phi_nominal", "jet4_mass_nominal",
    # ],
)

# plot all the variables in the dataframe in one pdf file
output_dir = "plots_all_vars_2017_vbf"
os.makedirs(output_dir, exist_ok=True)


n_vars = len(df.columns)
n_cols = 4
n_rows = (n_vars + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    ax = axes[i]
    sns.histplot(df[col], bins=50, kde=False, ax=ax)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("Counts")


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
# set log scale for y axis
for ax in axes:
    ax.set_yscale("log")
plt.savefig(os.path.join(output_dir, "all_vars_histograms_11Oct.pdf"))
plt.close()

# # plot only jet2_pt_nominal variable
# col = "jet2_pt_nominal"
# plt.figure(figsize=(8, 6))
# sns.histplot(df[col], bins=50, kde=False)
# plt.title(col)
# plt.xlabel(col)
# plt.ylabel("Counts")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, f"{col}_histogram.pdf"))
# plt.close()
