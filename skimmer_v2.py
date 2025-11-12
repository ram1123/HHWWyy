"""skimmer.py

Skim parquet inputs for DNN training and add new variables
"""

import os
import numpy as np
import dask_awkward as dak
from pathlib import Path
import awkward as ak

from rich import print

import selection

def sanitize_partition(rec: ak.Array) -> ak.Array:
    out = rec
    for f in out.fields:
        x = out[f]
        # replace None → -9
        x = ak.fill_none(x, -9.0)
        # replace NaN and ±Inf → -9
        x = ak.nan_to_num(x, nan=-9.0, posinf=-9.0, neginf=-9.0)
        # replace sentinel values → -9
        x = ak.where((x == -999.0) | (x == -99.0), -9.0, x)
        out = ak.with_field(out, x, f)
    return out

def skim(input_dir, out_dir, feature_columns, additional_columns_for_skimming):
    """Read input parquet (dask_awkward), apply selection, add derived
    variables and write out to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    sample_dict = {
        # "SampleName": (category_tag, if DY then true else false)
        # "ggh_powhegPS": ("notbtag", "False"),
        # "vbf_powheg_dipole": ("notbtag", "False"),
        # # backgrounds (union of ggh or vbf selections)
        # "dy_VBF_filter": ("notbtag", "True"),
        # "dy_M-100To200_MiNNLO": ("notbtag", "True"),
        # "dy_M-50_MiNNLO": ("notbtag", "True"),
        # "ewk_lljj_mll50_mjj120": ("notbtag", "False"),
        # "ttjets_dl": ("notbtag", "False"),
        # "ttjets_sl": ("notbtag", "False"),
        "data_B": ("notbtag", "False"),
        "data_C": ("notbtag", "False"),
        "data_D": ("notbtag", "False"),
        "data_E": ("notbtag", "False"),
        "data_F": ("notbtag", "False"),
    }
    columns_to_read = feature_columns + additional_columns_for_skimming

    for subdir, (tag, filter_func) in sample_dict.items():
        pattern = os.path.join(input_dir, subdir, "**", "*.parquet")
        print(f"Reading: {pattern}")
        ddf = dak.from_parquet(pattern, columns=columns_to_read)

        # apply selection
        ddf = selection.applyRegionCatCuts(
            ddf,
            category=tag,
            region_name="all",
            process=subdir,
            variation="nominal",
            do_vbf_filter_study=filter_func,
        )

        # if nan or inf or -999.0 or -99.0 values are present, then replace them with -9.0
        ddf = dak.map_partitions(sanitize_partition, ddf)

        # add additional lepton-related variables partition-wise,
        ddf = dak.map_partitions(_add_additional_lep_vars, ddf)
        # ddf = _add_additional_lep_vars(ddf)

        # # # # add four-jet derived variables partition-wise
        # # # if njets >= 4:
        # if ak.all(ddf["njets_nominal"] >= 4):
        print("Adding four-jet related variables")
        ddf = dak.map_partitions(_add_additional_vars_four_jets, ddf)
        # ddf = _add_additional_vars_four_jets(ddf)

        out_subdir = os.path.join(out_dir, subdir)
        os.makedirs(out_subdir, exist_ok=True)
        print(f"Writing skim to: {out_subdir}")

        try:
            ddf.to_parquet(out_subdir)
        except Exception as e:
            print(f"Warning: failed to write out skimmed parquet: {e}")


def _get_dPhi(phi1, phi2):
    """Compute Δφ in [0, π], honoring INVALID sentinels."""
    INVALID = -9.0
    # Valid where neither is INVALID
    valid = (phi1 != INVALID) & (phi2 != INVALID)

    # Raw difference
    d = phi1 - phi2

    # Wrap to (-π, π] via modulo, then |.| -> [0, π]
    d_wrapped = abs(((d + np.pi) % (2 * np.pi)) - np.pi)

    # Keep INVALID where not valid
    return ak.where(valid, d_wrapped, -1.0)


def _get_dR(eta1, phi1, eta2, phi2):
    """Compute ΔR = sqrt((Δη)^2 + (Δφ)^2), honoring INVALID sentinels."""
    INVALID = -9.0
    # Valid where all inputs are valid
    valid = (
        (eta1 != INVALID) & (eta2 != INVALID) & (phi1 != INVALID) & (phi2 != INVALID)
    )

    dphi = _get_dPhi(phi1, phi2)  # already INVALID-aware
    deta = abs(eta1 - eta2)

    dR = np.sqrt(deta**2 + dphi**2)
    return ak.where(valid, dR, -1.0)


def _get_kT(pt1, pt2, dR):
    """Compute kT = min(pt1, pt2) * dR, honoring INVALID sentinels."""
    INVALID = -9.0
    # valid where all inputs are valid
    valid = (pt1 != INVALID) & (pt2 != INVALID) & (dR != INVALID)

    pt_min = ak.where(pt1 < pt2, pt1, pt2)  # element-wise min
    kT_val = pt_min * dR

    # enforce positive-only dR and apply masking
    kT_val = ak.where((valid) & (dR > 0), kT_val, -1.0)
    return kT_val


def _get_Z(pt1, pt2):
    """Compute Z = min(pt1, pt2) / (pt1 + pt2), honoring INVALID sentinels."""
    INVALID = -9.0
    valid = (pt1 != INVALID) & (pt2 != INVALID)

    pt_min = ak.where(pt1 < pt2, pt1, pt2)  # element-wise min
    denom = pt1 + pt2

    # safe divide: only where denom>0 and valid
    Z_val = ak.where((valid) & (denom > 0), pt_min / denom, -1.0)
    return Z_val

def _get_invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    """Compute invariant mass of two objects given their pt, eta, phi and mass.
    Returns invariant mass.
    """
    INVALID = -9.0
    valid = (pt1 != INVALID) & (eta1 != INVALID) & (phi1 != INVALID) & (mass1 != INVALID) & \
            (pt2 != INVALID) & (eta2 != INVALID) & (phi2 != INVALID) & (mass2 != INVALID)

    # Compute invariant mass using the formula: m^2 = (E1 + E2)^2 - (p1 + p2)^2
    E1 = np.sqrt(pt1**2 + mass1**2)
    E2 = np.sqrt(pt2**2 + mass2**2)
    p1 = pt1 * np.cosh(eta1)
    p2 = pt2 * np.cosh(eta2)

    invariant_mass = np.sqrt((E1 + E2)**2 - (p1 + p2)**2)

    return ak.where(valid, invariant_mass, -1.0)


def _get_transverse_mass(pt1, phi1, pt2, phi2):
    """Compute transverse mass of two objects given their pt and phi.
    Returns transverse mass.
    """
    INVALID = -9.0
    valid = (pt1 != INVALID) & (phi1 != INVALID) & (pt2 != INVALID) & (phi2 != INVALID)

    dphi = _get_dPhi(phi1, phi2)  # already INVALID-aware

    transverse_mass = np.sqrt(2 * pt1 * pt2 * (1 - np.cos(dphi)))

    return ak.where(valid, transverse_mass, -1.0)

# function to get four pairwise features between two objects: dR, kT, Z and invariant mass
def _get_pairwise_features(ddf, var_postfix, obj1, obj2):
    """Compute pairwise features between two objects given their pt, eta, phi and mass.
    Returns a dictionary with keys: dR, kT, Z, invariantMass
    """
    ddf[f"dR_{var_postfix}"] = _get_dR(obj1["eta"], obj1["phi"], obj2["eta"], obj2["phi"])
    # print eta, phi and dr for debug
    # dphi and deta for debug
    # print(f"obj1 eta: {ak.to_list(obj1['eta'][:5])}")
    # print(f"obj2 eta: {ak.to_list(obj2['eta'][:5])}")
    # print(f"deta: {ak.to_list(abs(obj1['eta'] - obj2['eta'])[:5])}\n\n")

    # print(f"obj1 phi: {ak.to_list(obj1['phi'][:5])}")
    # print(f"obj2 phi: {ak.to_list(obj2['phi'][:5])}")
    # print(f"dphi: {ak.to_list(_get_dPhi(obj1['phi'], obj2['phi'])[:5])}\n")

    # print(f"dR_{var_postfix}: {ak.to_list(ddf[f'dR_{var_postfix}'][:5])}\n\n")
    ddf[f"kT_{var_postfix}"] = _get_kT(obj1["pt"], obj2["pt"], ddf[f"dR_{var_postfix}"])
    ddf[f"Z_{var_postfix}"] = _get_Z(obj1["pt"], obj2["pt"])
    ddf[f"invariantMass_{var_postfix}"] = _get_invariant_mass(obj1["pt"], obj1["eta"], obj1["phi"], obj1["mass"],
                                                               obj2["pt"], obj2["eta"], obj2["phi"], obj2["mass"])
    return ddf

# function to get four pairwise features between two objects (if one is MET then): dPhi, kT, Z and transverse mass
def _get_pairwise_features_met(ddf, var_postfix, obj, met):
    """Compute pairwise features between an object and MET given their pt, eta, phi and mass.
    Returns a dictionary with keys: dPhi, kT, Z, transverseMass
    """
    ddf[f"dPhi_{var_postfix}"] = _get_dPhi(obj["phi"], met["phi"])
    ddf[f"kT_{var_postfix}"] = _get_kT(obj["pt"], met["pt"], ddf[f"dPhi_{var_postfix}"])
    ddf[f"Z_{var_postfix}"] = _get_Z(obj["pt"], met["pt"])
    ddf[f"transverseMass_{var_postfix}"] = _get_transverse_mass(obj["pt"], obj["phi"], met["pt"], met["phi"])
    return ddf


def _add_additional_lep_vars(ddf):
    """Add additional lepton-related variables to the dataframe."""

    ddf["mu1_mass"] = 0.1056583
    ddf["mu2_mass"] = 0.1056583

    ddf = _get_pairwise_features(ddf, "mu1_mu2",
                                {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"], "mass": ddf["mu1_mass"]},
                                {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"], "mass": ddf["mu2_mass"]})
    ddf = _get_pairwise_features_met(ddf, "mu1_MET",
                                    {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})
    ddf = _get_pairwise_features_met(ddf, "mu2_MET",
                                    {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})

    # dimuon pt/dimuon mass and its log
    ddf["dimuon_pt_over_mass"] = ddf["dimuon_pt"] / ddf["dimuon_mass"]
    ddf["dimuon_pt_over_mass_log"] = np.log(ddf["dimuon_pt_over_mass"])
    return ddf


def _add_additional_vars_four_jets(ddf):
    """Add additional four-jet related variables to the dataframe.

    Following additional variables, dR, kT, Z and invariant mass are added for following pairs:
    1. jet1 and jet2
    2. jet1 and jet3
    3. jet1 and jet4
    4. jet2 and jet3
    5. jet2 and jet4
    6. jet3 and jet4
    7. jet1 and mu1
    8. jet1 and mu2
    9. jet1 and MET
    10. jet2 and mu1
    11. jet2 and mu2
    12. jet2 and MET
    13. jet3 and mu1
    14. jet3 and mu2
    15. jet3 and MET
    16. jet4 and mu1
    17. jet4 and mu2
    18. jet4 and MET
    """
    ddf = _get_pairwise_features(ddf, "jet1_jet2",
                                {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"], "mass": ddf["jet1_mass_nominal"]},
                                {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"], "mass": ddf["jet2_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "jet1_jet3",
                                {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"], "mass": ddf["jet1_mass_nominal"]},
                                {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"], "mass": ddf["jet3_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "jet1_jet4",
                                {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"], "mass": ddf["jet1_mass_nominal"]},
                                {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"], "mass": ddf["jet4_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "jet2_jet3",
                                {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"], "mass": ddf["jet2_mass_nominal"]},
                                {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"], "mass": ddf["jet3_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "jet2_jet4",
                                {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"], "mass": ddf["jet2_mass_nominal"]},
                                {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"], "mass": ddf["jet4_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "jet3_jet4",
                                {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"], "mass": ddf["jet3_mass_nominal"]},
                                {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"], "mass": ddf["jet4_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "mu1_jet1",
                                {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"], "mass": ddf["mu1_mass"]},
                                {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"], "mass": ddf["jet1_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "mu2_jet1",
                                {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"], "mass": ddf["mu2_mass"]},
                                {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"], "mass": ddf["jet1_mass_nominal"]})
    ddf = _get_pairwise_features_met(ddf, "jet1_MET",
                                    {"pt": ddf["jet1_pt_nominal"], "eta": ddf["jet1_eta_nominal"], "phi": ddf["jet1_phi_nominal"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})
    ddf = _get_pairwise_features(ddf, "mu1_jet2",
                                {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"], "mass": ddf["mu1_mass"]},
                                {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"], "mass": ddf["jet2_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "mu2_jet2",
                                {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"], "mass": ddf["mu2_mass"]},
                                {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"], "mass": ddf["jet2_mass_nominal"]})
    ddf = _get_pairwise_features_met(ddf, "jet2_MET",
                                    {"pt": ddf["jet2_pt_nominal"], "eta": ddf["jet2_eta_nominal"], "phi": ddf["jet2_phi_nominal"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})
    ddf = _get_pairwise_features(ddf, "mu1_jet3",
                                {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"], "mass": ddf["mu1_mass"]},
                                {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"], "mass": ddf["jet3_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "mu2_jet3",
                                {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"], "mass": ddf["mu2_mass"]},
                                {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"], "mass": ddf["jet3_mass_nominal"]})
    ddf = _get_pairwise_features_met(ddf, "jet3_MET",
                                    {"pt": ddf["jet3_pt_nominal"], "eta": ddf["jet3_eta_nominal"], "phi": ddf["jet3_phi_nominal"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})
    ddf = _get_pairwise_features(ddf, "mu1_jet4",
                                {"pt": ddf["mu1_pt"], "eta": ddf["mu1_eta"], "phi": ddf["mu1_phi"], "mass": ddf["mu1_mass"]},
                                {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"], "mass": ddf["jet4_mass_nominal"]})
    ddf = _get_pairwise_features(ddf, "mu2_jet4",
                                {"pt": ddf["mu2_pt"], "eta": ddf["mu2_eta"], "phi": ddf["mu2_phi"], "mass": ddf["mu2_mass"]},
                                {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"], "mass": ddf["jet4_mass_nominal"]})
    ddf = _get_pairwise_features_met(ddf, "jet4_MET",
                                    {"pt": ddf["jet4_pt_nominal"], "eta": ddf["jet4_eta_nominal"], "phi": ddf["jet4_phi_nominal"]},
                                    {"pt": ddf["MET_pt"], "phi": ddf["MET_phi"]})
    return ddf


def main():
    input_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output/2016preVFP/f1_0/"
    out_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2016preVFP/"

    input_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output/2016postVFP/f1_0/"
    out_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2016postVFP/"

    input_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output/2017/f1_0/"
    out_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2017/"

    # input_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output/2018/f1_0/"
    # out_dir = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2018/"

    feature_columns = [
        "mu1_pt",
        "mu1_eta",
        "mu1_phi",
        "mu2_pt",
        "mu2_eta",
        "mu2_phi",
        "mu1_pt_over_mass",
        "mu2_pt_over_mass",
        "dimuon_mass",
        "dimuon_pt",
        "dimuon_eta",
        "dimuon_phi",
        "dimuon_pt_log",
        "dimuon_rapidity",
        "dimuon_phi_cs",
        "dimuon_dR",
        "dimuon_ebe_mass_res",
        "dimuon_ebe_mass_res_rel",
        "dimuon_cos_theta_cs",
        "njets_nominal",
        "jet1_pt_nominal",
        "jet1_eta_nominal",
        "jet1_phi_nominal",
        "jet1_mass_nominal",
        "jet1_qgl_nominal",
        "jet2_pt_nominal",
        "jet2_eta_nominal",
        "jet2_phi_nominal",
        "jet2_mass_nominal",
        "jet2_qgl_nominal",
        "jet3_pt_nominal",
        "jet3_eta_nominal",
        "jet3_phi_nominal",
        "jet3_mass_nominal",
        "jet3_qgl_nominal",
        "jet4_pt_nominal",
        "jet4_eta_nominal",
        "jet4_phi_nominal",
        "jet4_mass_nominal",
        "jet4_qgl_nominal",
        "jj_mass_nominal",
        "jj_mass_log_nominal",
        "jj_dEta_nominal",
        "jj_dPhi_nominal",
        "htsoft2_nominal",
        "nsoftjets5_nominal",
        "rpt_nominal",
        "mmj_min_dEta_nominal",
        "mmj_min_dPhi_nominal",
        "ll_zstar_log_nominal",
        "pt_centrality_nominal",
        "zeppenfeld_nominal",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "nfatJets_drmuon",
        "MET_pt",
        "MET_phi",
        "wgt_nominal",
        "event",
        "fraction",
        "year",
    ]

    additional_column_for_skimming = [
        "dimuon_mass",
        "nBtagLoose_nominal",
        "nBtagMedium_nominal",
        "gjj_mass",
        ]

    # If using a Dask gateway, create client here (optional)
    try:
        from dask_gateway import Gateway

        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[-1]
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
    except Exception:
        pass

    skim(input_dir, out_dir, feature_columns, additional_column_for_skimming)

if __name__ == "__main__":
    main()
