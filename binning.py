import numpy as np

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
    out[mask]  = 2.0 * ((S[mask] + B[mask]) * np.log(1 + S[mask] / B[mask]) - S[mask])
    out[~mask] = 2.0 * S[~mask]
    return out

# -----------------------------------------------------
# Significance-optimized binning via dynamic programming
# -----------------------------------------------------
def make_significance_binning(
    sig_score, bkg_score,
    sig_w=None, bkg_w=None,
    nbins=10,
    fine_bins=300,
    score_min=None, score_max=None,
    min_total_events_per_bin=0.0,  # stability: (S+B) >= this
    min_signal_per_bin=0.3,        # CMS H→μμ-style guard: S >= this
    clamp_edges=True,              # clamp first/last edges to [0,1]
):
    """
    Compute non-uniform bin edges in the score that maximize total Asimov significance.
    - Dynamic programming over a fine prebinning (optimal, not greedy).

    Returns:
        edges: (nbins+1,) array of score edges
        S_bins, B_bins: per-bin S and B
        Z_bins: per-bin Asimov Z
        Z_tot: total Asimov Z
    """
    sig_score = np.asarray(sig_score, dtype=float)
    bkg_score = np.asarray(bkg_score, dtype=float)
    sig_w = np.ones_like(sig_score, float) if sig_w is None else np.asarray(sig_w, float)
    bkg_w = np.ones_like(bkg_score, float) if bkg_w is None else np.asarray(bkg_w, float)

    # Drop NaNs/Infs safely
    def _clean(x, w):
        m = np.isfinite(x) & np.isfinite(w)
        return x[m], w[m]
    sig_score, sig_w = _clean(sig_score, sig_w)
    bkg_score, bkg_w = _clean(bkg_score, bkg_w)

    if sig_score.size == 0 or bkg_score.size == 0:
        raise ValueError("Empty signal or background arrays after cleaning.")

    # score range
    if score_min is None: score_min = min(sig_score.min(), bkg_score.min())
    if score_max is None: score_max = max(sig_score.max(), bkg_score.max())
    if not np.isfinite(score_min) or not np.isfinite(score_max) or score_min >= score_max:
        raise ValueError("Invalid score range computed for binning.")

    # fine prebinning over score
    fine_edges = np.linspace(score_min, score_max, int(fine_bins) + 1)
    S_hist, _  = np.histogram(sig_score, bins=fine_edges, weights=sig_w)
    B_hist, _  = np.histogram(bkg_score, bins=fine_edges, weights=bkg_w)
    nF = len(S_hist)

    # prefix sums for O(1) range queries
    S_cum = np.concatenate([[0.0], np.cumsum(S_hist)])
    B_cum = np.concatenate([[0.0], np.cumsum(B_hist)])
    def SB(i, j):  # inclusive i .. j-1
        return (S_cum[j] - S_cum[i], B_cum[j] - B_cum[i])

    # precompute Z^2 for any contiguous fine-bin range [i, j)
    Z2 = np.full((nF + 1, nF + 1), -np.inf, float)
    for i in range(nF):
        Sj = 0.0; Bj = 0.0
        for j in range(i + 1, nF + 1):
            Sj += S_hist[j - 1]
            Bj += B_hist[j - 1]
            # stability guards
            if Sj < min_signal_per_bin:
                continue
            if (Sj + Bj) < min_total_events_per_bin:
                continue
            Z2[i, j] = z2_asimov(Sj, Bj)

    # dynamic programming: dp[k, j] = best Z^2 using k bins up to fine index j
    dp   = np.full((nbins + 1, nF + 1), -np.inf, float)
    prev = np.full((nbins + 1, nF + 1), -1, int)
    dp[0, 0] = 0.0

    for k in range(1, nbins + 1):
        for j in range(1, nF + 1):
            best = -np.inf; best_i = -1
            i_min = k - 1  # need at least k-1 parts before i
            for i in range(i_min, j):
                val_last = Z2[i, j]
                if val_last == -np.inf:
                    continue
                val = dp[k - 1, i] + val_last
                if val > best:
                    best, best_i = val, i
            dp[k, j] = best
            prev[k, j] = best_i

    # backtrack optimal cut positions
    edges_idx = [nF]
    k, j = nbins, nF
    while k > 0:
        i = prev[k, j]
        if i < 0:
            # fallback (constraints too tight) — return uniform
            edges = np.linspace(score_min, score_max, nbins + 1)
            return edges, None, None, None, None
        edges_idx.append(i)
        j = i; k -= 1
    edges_idx = edges_idx[::-1]
    edges = fine_edges[edges_idx]

    if clamp_edges:
        # clamp numerical fuzz and enforce monotonicity
        edges[0]  = max(edges[0], score_min)
        edges[-1] = min(edges[-1], score_max)
        for t in range(1, len(edges)):
            if edges[t] <= edges[t-1]:
                edges[t] = np.nextafter(edges[t-1], np.inf)

    # per-bin summaries
    S_bins, B_bins, Z_bins = [], [], []
    for a, b in zip(edges_idx[:-1], edges_idx[1:]):
        S, B = SB(a, b)
        S_bins.append(S); B_bins.append(B)
        Z_bins.append(np.sqrt(max(0.0, Z2[a, b])))
    S_bins = np.array(S_bins); B_bins = np.array(B_bins); Z_bins = np.array(Z_bins)
    Z_tot  = np.sqrt(np.sum(Z_bins**2))

    return edges, S_bins, B_bins, Z_bins, Z_tot

# ---------------------------------------------------
# Scan nbins and pick the highest total Asimov Z setup
# ---------------------------------------------------
def scan_nbins_for_best_edges(
    sig_score, bkg_score, sig_w=None, bkg_w=None,
    nbins_list=range(2, 14), **kwargs
):
    best = (-np.inf, None, None, None, None, None)
    for nb in nbins_list:
        edges, S, B, Z, Ztot = make_significance_binning(
            sig_score, bkg_score, sig_w, bkg_w, nbins=nb, **kwargs
        )
        if Ztot is not None and Ztot > best[0]:
            best = (Ztot, nb, edges, S, B, Z)
    Zbest, nb_best, edges_best, S_best, B_best, Z_bins = best
    return nb_best, edges_best, S_best, B_best, Z_bins, Zbest

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
    for name, globpath in items:
        ev = dak.from_parquet(globpath)
        ev = selection.applyRegionCatCuts(
            ev,
            category=category,
            region_name=region_name,
            process=name,
            variation="nominal",
            do_vbf_filter_study=False,
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
        "vbf_powheg_dipole": (
            "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_11August_FixDimuonMass/vbf_powheg_dipole/0/*.parquet"
        ),
    }
    sig_score, sig_w = collect_scores(sig_globs, selection)

    bkg_globs = {
        # "dy_MiNNLO": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_11August_FixDimuonMass/dy_M-100To200_MiNNLO/**/*.parquet",
        "dy_VBFilt": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted_03September_FixDimuonMass/dy_VBF_filter/**/*.parquet",
        # "tt_st": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_11August_FixDimuonMass/ttjets_dl/**/*.parquet",
        # "vv": "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/*/compacted_11August_FixDimuonMass/ww_wz_zz/**/*.parquet",
    }
    bkg_score, bkg_w = collect_scores(bkg_globs, selection)

    nb, edges, Sbins, Bbins, Zbins, Ztot = scan_nbins_for_best_edges(
        sig_score, bkg_score, sig_w, bkg_w,
        nbins_list=range(3, 14),
        fine_bins=400,
        score_min=0.0,
        score_max=1.0,
        min_total_events_per_bin=5.0,
        min_signal_per_bin=0.3,
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
