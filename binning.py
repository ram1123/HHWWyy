import numpy as np

from rich import print

# ------------------------------
# Asimov per-bin Z^2 (additive)
# ------------------------------
def z2_asimov(S, B, eps=1e-9):
    """
    Return Z^2 per bin using the Asimov formula (additive across bins).
    Uses a conservative Poisson limit (Z^2 ≈ 2S) when B ~ 0.
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    # guard against tiny negative weights from MC/systematics
    S = np.maximum(S, 0.0)
    B = np.maximum(B, 0.0)

    out  = np.zeros_like(S, dtype=float)
    mask = B > eps
    # out[mask] = np.sqrt(2.0 *((S[mask] + B[mask]) * np.log1p(S[mask] / B[mask]) - S[mask]))
    out[mask] = 2.0 *((S[mask] + B[mask]) * np.log1p(S[mask] / B[mask]) - S[mask])
    out[~mask] = 2.0 * S[~mask]
    return out
    return out if out.ndim else out.item()

# -----------------------------------------------------
# Significance-optimized binning via dynamic programming
# -----------------------------------------------------
def make_significance_binning(
    sig_score, bkg_score,
    sig_w=None, bkg_w=None,
    fine_bins=500,
    score_min=None, score_max=None,
    min_total_events_per_bin=0.0,  # require (S+B) >= this
    min_signal_per_bin=0.3,        # require S >= this
    tol_frac=0.05,                 # allow merge if local Z² drop <= 5%
    clamp_edges=True
):
    """
    Greedy R->L merging with a 5% loss veto:
      - Merge neighboring fine bins from the right as long as merging causes
        at most `tol_frac` fractional loss in local Z².
      - If the accumulator fails S or (S+B) guards, keep merging regardless of loss
        until guards are satisfied.
      - No cap on the final number of bins.
    Returns:
        edges, S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins, Z_tot
    """
    # -------- clean inputs --------
    sig_score = np.asarray(sig_score, float)
    bkg_score = np.asarray(bkg_score, float)
    sig_w = np.ones_like(sig_score) if sig_w is None else np.asarray(sig_w, float)
    bkg_w = np.ones_like(bkg_score) if bkg_w is None else np.asarray(bkg_w, float)

    def _clean(x, w):
        m = np.isfinite(x) & np.isfinite(w)
        return x[m], w[m]
    sig_score, sig_w = _clean(sig_score, sig_w)
    bkg_score, bkg_w = _clean(bkg_score, bkg_w)
    if sig_score.size == 0 or bkg_score.size == 0:
        raise ValueError("Empty signal or background arrays after cleaning.")

    if score_min is None: score_min = min(sig_score.min(), bkg_score.min())
    if score_max is None: score_max = max(sig_score.max(), bkg_score.max())
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max <= score_min:
        raise ValueError("Invalid score range.")
    sig_score = np.clip(sig_score, score_min, score_max)
    bkg_score = np.clip(bkg_score, score_min, score_max)

    # -------- fine prebinning --------
    fine_edges = np.linspace(score_min, score_max, int(fine_bins)+1)
    S_hist, _    = np.histogram(sig_score, bins=fine_edges, weights=sig_w)
    B_hist, _    = np.histogram(bkg_score, bins=fine_edges, weights=bkg_w)
    S_hist_nw, _ = np.histogram(sig_score, bins=fine_edges)
    B_hist_nw, _ = np.histogram(bkg_score, bins=fine_edges)
    nF = len(S_hist)

    def SB(a, b):    return S_hist[a:b].sum(),    B_hist[a:b].sum()
    def SBnw(a, b):  return S_hist_nw[a:b].sum(), B_hist_nw[a:b].sum()

    # -------- greedy merge from right --------
    bins_idx = []                  # list of (i_start, i_end) for final bins
    acc_l = nF-1; acc_r = nF       # accumulator is [acc_l, acc_r)
    S_acc, B_acc = SB(acc_l, acc_r)

    i = acc_l - 1
    while True:
        can_merge = (i >= 0)
        # force-merge if guards not satisfied
        need_guard_merge = (S_acc < min_signal_per_bin) or ((S_acc + B_acc) < min_total_events_per_bin)

        if can_merge:
            # local separate vs merged Z²
            S_L, B_L = SB(i, i+1)
            z2_sep = z2_asimov(S_L, B_L) + z2_asimov(S_acc, B_acc)
            S_m, B_m = (S_L + S_acc, B_L + B_acc)
            z2_mrg = z2_asimov(S_m, B_m)

            # fractional drop if we merge (robust to z2_sep ~ 0)
            frac_drop = 0.0 if z2_sep <= 0.0 else max(0.0, (z2_sep - z2_mrg) / z2_sep)

            if need_guard_merge or (frac_drop <= tol_frac):
                # accept merge and continue extending left
                acc_l = i
                S_acc, B_acc = S_m, B_m
                i -= 1
            else:
                # veto merge -> freeze current accumulator as a bin; start new acc at left bin
                bins_idx.append((acc_l, acc_r))
                acc_r = acc_l
                acc_l = i
                S_acc, B_acc = SB(acc_l, acc_r)
                i -= 1
        else:
            # nothing left to merge: finalize accumulator
            bins_idx.append((acc_l, acc_r))
            break

    # left->right order
    bins_idx = bins_idx[::-1]

    # -------- edges & summaries --------
    edges_idx = [bins_idx[0][0]] + [b for (_, b) in bins_idx]
    edges = fine_edges[edges_idx]

    if clamp_edges:
        edges[0]  = max(edges[0], score_min)
        edges[-1] = min(edges[-1], score_max)
        for t in range(1, len(edges)):
            if edges[t] <= edges[t-1]:
                edges[t] = np.nextafter(edges[t-1], np.inf)

    S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins = [], [], [], [], []
    for a,b in bins_idx:
        S,B     = SB(a,b)
        Sn,Bn   = SBnw(a,b)
        S_bins.append(S); B_bins.append(B)
        S_bins_nw.append(Sn); B_bins_nw.append(Bn)
        Z_bins.append(np.sqrt(z2_asimov(S,B)))
    S_bins     = np.array(S_bins)
    B_bins     = np.array(B_bins)
    S_bins_nw  = np.array(S_bins_nw)
    B_bins_nw  = np.array(B_bins_nw)
    Z_bins     = np.array(Z_bins)
    Z_tot      = np.sqrt(np.sum(z2_asimov(S_bins, B_bins)))

    return edges, S_bins, B_bins, S_bins_nw, B_bins_nw, Z_bins, Z_tot

# ---------------------------------------------------
# Scan nbins and pick the highest total Asimov Z setup
# ---------------------------------------------------
def scan_nbins_for_best_edges(
    sig_score, bkg_score, sig_w=None, bkg_w=None,
    nbins_list=range(2, 14),
    fine_bins=300,
    score_min=None, score_max=None,
    min_total_events_per_bin=0.0,  # stability: (S+B) >= this
    min_signal_per_bin=0.3,        # CMS H→μμ-style guard: S >= this
    clamp_edges=True,              # clamp first/last edges to [0,1]
):
    """
    Scan multiple nbins and return the one with the highest total Asimov Z.
    See make_significance_binning() for parameter details.
    """
    best_Z = -np.inf
    best_result = (None, None, None, None, None)
    nb_list = []
    Z_tot_list = []
    S_NoWgt_bins_list = []
    B_NoWgt_bins_list = []
    for nb in nbins_list:
        edges, S_bins, B_bins, S_NoWgt_bins, B_NoWgt_bins, Z_bins, Z_tot = make_significance_binning(
            sig_score, bkg_score,
            sig_w=sig_w, bkg_w=bkg_w,
            fine_bins=fine_bins,
            score_min=score_min,
            score_max=score_max,
            min_total_events_per_bin=min_total_events_per_bin,
            min_signal_per_bin=min_signal_per_bin,
            clamp_edges=clamp_edges,
        )
        nb_list.append(nb)
        Z_tot_list.append(Z_tot if Z_tot is not None else -np.inf)
        S_NoWgt_bins_list.append(S_NoWgt_bins[-1])
        B_NoWgt_bins_list.append(B_NoWgt_bins[-1])
        print(f"nbins={nb:>3}: Z_tot = {Z_tot:<2.2f}, S_bins (NoWgt) = {S_NoWgt_bins}")
        # count spaces taken by last statement before S_bins
        spaces = " " * (len(f"nbins={nb:>3}: Z_tot = {Z_tot:<2.2f}")+1)
        print(f"{spaces} B_bins (NoWgt) = {B_NoWgt_bins}")
        print(f"{spaces} Edges = {edges}")
        if Z_tot is not None and Z_tot > best_Z:
            best_Z = Z_tot
            best_result = (nb, edges, S_bins, B_bins, Z_bins, Z_tot)

    # plot for the nb vs Z_tot scan

    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = 'tab:blue'
        color2 = 'tab:green'
        color3 = 'tab:red'
        ax1.set_xlabel("Number of bins")
        ax1.set_ylabel("Total Asimov Z", color=color1)
        l1, = ax1.plot(nb_list, Z_tot_list, marker='o', color=color1, label='Total Asimov Z')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(nb_list)
        ax1.grid(True)

        # Second y-axis for S_NoWgt_bins_list
        ax2 = ax1.twinx()
        ax2.set_ylabel('S_NoWgt (last bin)', color=color2)
        l2, = ax2.plot(nb_list, S_NoWgt_bins_list, marker='s', color=color2, label='S_NoWgt (last bin)')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Third y-axis for B_NoWgt_bins_list
        ax3 = ax1.twinx()
        # Offset the third axis to the right
        ax3.spines["right"].set_position(('axes', 1.2))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)
        for sp in ax3.spines.values():
            sp.set_visible(True)
        ax3.set_ylabel('B_NoWgt (last bin)', color=color3)
        l3, = ax3.plot(nb_list, B_NoWgt_bins_list, marker='^', color=color3, label='B_NoWgt (last bin)')
        ax3.tick_params(axis='y', labelcolor=color3)

        # Set different y-limits for S and B if needed
        if len(S_NoWgt_bins_list) > 0:
            smin, smax = min(S_NoWgt_bins_list), max(S_NoWgt_bins_list)
            ax2.set_ylim(smin * 0.9, smax * 1.1)
        if len(B_NoWgt_bins_list) > 0:
            bmin, bmax = min(B_NoWgt_bins_list), max(B_NoWgt_bins_list)
            ax3.set_ylim(bmin * 0.9, bmax * 1.1)

        # Add legends
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title("Scan of binning configurations")
        fig.tight_layout()
        plt.savefig("binning_scan_nbins_vs_Ztot.pdf")
        plt.close(fig)
        print("Saved binning scan plot to binning_scan_nbins_vs_Ztot.pdf")
    except ImportError:
        print("matplotlib not available")
        pass  # matplotlib not available

    if best_Z < 0.0:
        raise RuntimeError("Failed to find a valid binning configuration.")
    return best_result


# ------------------------------------
# Background collection (example usage)
# ------------------------------------
def collect_scores(process_globs, selection, category="vbf", region_name="h-peak"):
    """
    Build score and weight arrays by concatenating any set of processes.
    - process_globs: mapping or iterable of (process_name, parquet_glob)
    - selection: your modules.selection (needs applyRegionCatCuts)
    """
    import dask_awkward as dak

    if hasattr(process_globs, "items"):
        items = process_globs.items()
    else:
        try:
            items = dict(process_globs).items()
        except (TypeError, ValueError) as exc:
            raise TypeError("process_globs must be a mapping or iterable of (name, glob) pairs.") from exc

    scores, weights = [], []
    do_vbf_filter_study = False
    for name, globpath in items:
        if "dy_" in name:
            do_vbf_filter_study = True
        print(f"Processing {name} from {globpath} (do_vbf_filter_study={do_vbf_filter_study})")
        ev = dak.from_parquet(globpath)
        ev = selection.applyRegionCatCuts(
            ev,
            category=category,
            region_name=region_name,
            process=name,
            variation="nominal",
            do_vbf_filter_study=do_vbf_filter_study,
        )
        scores.append(ev["dnn_vbf_score_atanh"].compute().to_numpy())
        weights.append(ev["wgt_nominal"].compute().to_numpy())
    return np.concatenate(scores), np.concatenate(weights)

def collect_bkg(process_globs, selection, category="vbf", region_name="h-peak"):
    """Backward-compatible alias for background collection."""
    return collect_scores(process_globs, selection, category=category, region_name=region_name)

# -----------------------
# Example driver snippet
# -----------------------
if __name__ == "__main__":
    import selection

    sig_globs = {
        "vbf_powheg_dipole": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/vbf_powheg_dipole/**/*.parquet",
        # "ggh_powhegPS": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/ggh_powhegPS/**/*.parquet",
    }
    sig_score, sig_w = collect_scores(sig_globs, selection)

    bkg_globs = {
        "dy_VBF_filter": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/dy_VBF_filter/**/*.parquet",
        # "dy_M-50_aMCatNLO": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/dy_M-50_aMCatNLO/**/*.parquet",
        # "dy_M-100To200_aMCatNLO": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/dy_M-100To200_aMCatNLO/**/*.parquet",
        # "ewk_lljj_mll50_mjj120": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/ewk_lljj_mll50_mjj120/**/*.parquet",
        # "ttjets_dl": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/ttjets_dl/**/*.parquet",
        # "ttjets_sl": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/ttjets_sl/**/*.parquet",
        # "zz": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_19September_FixDimuonMass/zz/**/*.parquet",
    }
    bkg_score, bkg_w = collect_scores(bkg_globs, selection)

    score_lower = 0.0
    score_min = float(min(sig_score.max(), bkg_score.max()))
    score_upper = float(max(sig_score.max(), bkg_score.max()))
    print(f"Derived dnn_vbf_score_atanh range: [{score_min:.6f}, {score_upper:.6f}]")

    nb, edges, Sbins, Bbins, Zbins, Ztot = scan_nbins_for_best_edges(
        sig_score, bkg_score, sig_w, bkg_w,
        nbins_list=range(2, 5),
        fine_bins=500,
        score_min=score_lower,
        score_max=score_upper,
        min_total_events_per_bin=5.0,
        min_signal_per_bin=0.03,
        clamp_edges=True,
    )

    print(f"Best nbins = {nb}, total Asimov Z = {Ztot:.3f}")
    print("edges = np.array([")
    for e in edges:
        print(f"  {e:.6f},")
    print("])")
    print("Per-bin (S, B, Z):")
    for i, (S, B, Z) in enumerate(zip(Sbins, Bbins, Zbins), 1):
        print(f"  bin {i}: S={S:.3f}  B={B:.3f}  Z={Z:.3f}")
