"""skimmer.py

Skim parquet inputs for DNN training and optionally add derived variables
for zero-jet events. Splits output into per-njets subfolders.
"""

import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask_awkward as dak
import awkward as ak

# your module:
import selection  # must provide applyRegionCatCuts(pdf, category=...) or similar


def _ensure_bool_series(mask, pdf, category):
    # Accept Series/ndarray/dict from selection; return boolean Series aligned to pdf
    if isinstance(mask, (pd.Series, np.ndarray)):
        m = np.asarray(mask).astype(bool)
        return pd.Series(m, index=pdf.index)
    if isinstance(mask, dict):
        key = category if category in mask else f"is_{category}"
        if key not in mask:
            raise ValueError(f"selection mask dict missing key '{key}'")
        m = np.asarray(mask[key]).astype(bool)
        return pd.Series(m, index=pdf.index)
    raise TypeError(f"Unsupported mask type from selection: {type(mask)}")


def _filter_partition_bkg(pdf):
    # keep signal that passes either ggh OR vbf selection
    m_g = _ensure_bool_series(selection.applyRegionCatCuts(pdf, category="ggh"), pdf, "ggh")
    m_v = _ensure_bool_series(selection.applyRegionCatCuts(pdf, category="vbf"), pdf, "vbf")
    return pdf.loc[(m_g | m_v)]


def _add_zero_jet_vars(part):
    """Awkward partition transform: add mu_mu_dR, k_T, z_ratio, dimuon_mass_computed
    for events with njets_nominal == 0. Uses PDG muon mass if per-muon masses are
    not present.
    """
    try:
        required = {"njets_nominal", "mu1_pt", "mu2_pt", "mu1_eta", "mu2_eta", "mu1_phi", "mu2_phi"}
        if not required.issubset(set(part.fields)):
            return part

        nj = part["njets_nominal"]
        mask0 = nj == 0

        pt1 = part["mu1_pt"]
        pt2 = part["mu2_pt"]
        eta1 = part["mu1_eta"]
        eta2 = part["mu2_eta"]
        phi1 = part["mu1_phi"]
        phi2 = part["mu2_phi"]

        # dphi with wrapping to [-pi, pi]
        dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
        deta = eta1 - eta2
        dr = ak.sqrt(deta * deta + dphi * dphi)

        # k_T and z_ratio
        kt = ak.minimum(pt1, pt2) * dr
        z = ak.minimum(pt1, pt2) / (pt1 + pt2)

        # use PDG muon mass
        mu_mass = 0.1056583745

        # compute energy and momentum components
        px1 = pt1 * ak.cos(phi1)
        py1 = pt1 * ak.sin(phi1)
        pz1 = pt1 * ak.sinh(eta1)
        E1 = ak.sqrt((pt1 * ak.cosh(eta1)) ** 2 + mu_mass ** 2)

        px2 = pt2 * ak.cos(phi2)
        py2 = pt2 * ak.sin(phi2)
        pz2 = pt2 * ak.sinh(eta2)
        E2 = ak.sqrt((pt2 * ak.cosh(eta2)) ** 2 + mu_mass ** 2)

        Etot = E1 + E2
        pxtot = px1 + px2
        pytot = py1 + py2
        pztot = pz1 + pz2
        m2tot = Etot * Etot - (pxtot * pxtot + pytot * pytot + pztot * pztot)
        # guard negatives
        m2tot = ak.where(m2tot < 0, 0.0, m2tot)
        mass = ak.sqrt(m2tot)

        # set values only where mask0, else nan
        mu_mu_dR = ak.where(mask0, dr, ak.nan)
        k_T = ak.where(mask0, kt, ak.nan)
        z_ratio = ak.where(mask0, z, ak.nan)
        dimuon_mass = ak.where(mask0, mass, ak.nan)

        part = ak.with_field(part, mu_mu_dR, "mu_mu_dR")
        part = ak.with_field(part, k_T, "k_T")
        part = ak.with_field(part, z_ratio, "z_ratio")
        part = ak.with_field(part, dimuon_mass, "dimuon_mass_computed")
        return part
    except Exception:
        return part


def skim_for_dnn(input_root, out_root, feature_columns):
    """Main driver: read input parquet (dask_awkward), apply selection, add derived
    variables and write out per-njets parquet splits.
    """
    os.makedirs(out_root, exist_ok=True)

    specs = {
        # "SampleName": (category_tag, if DY then true else false)
        "ggh_powhegPS": ("notbtag", "False"),
        "vbf_powheg_dipole": ("notbtag", "False"),
    }

    for subdir, (tag, filter_func) in specs.items():
        pattern = os.path.join(input_root, subdir, "**", "*.parquet")
        print(f"Reading: {pattern}")
        additional_columns_for_selection = [
            "dimuon_mass",
            "nBtagLoose_nominal",
            "nBtagMedium_nominal",
            "gjj_mass",
        ]
        ddf = dak.from_parquet(pattern, columns=feature_columns + additional_columns_for_selection)

        # apply selection
        ddf = selection.applyRegionCatCuts(
            ddf,
            category=tag,
            region_name="h-peak",
            process=subdir,
            variation="nominal",
            do_vbf_filter_study=filter_func,
        )

        # add zero-jet derived variables partition-wise
        try:
            ddf = dak.map_partitions(_add_zero_jet_vars, ddf)
        except Exception as e:
            print(f"Warning: failed to add zero-jet derived variables: {e}")

        # split the ddf based on njets: 0, 1, 2, 3, 4+
        out_dir = os.path.join(out_root, subdir)
        print(f"Writing skim to: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        try:
            # full selection
            full_out = os.path.join(out_dir, "all_njets")
            os.makedirs(full_out, exist_ok=True)
            print(f"  -> writing full selection to {full_out}")
            ddf.to_parquet(full_out)

            # per-njets splits
            for n in range(0, 4):
                mask = ddf[ddf["njets_nominal"] == n]
                sub_out = os.path.join(out_dir, f"njets_{n}")
                os.makedirs(sub_out, exist_ok=True)
                try:
                    cnt = int(mask.count().compute())
                except Exception:
                    cnt = None
                print(f"  -> njets_{n}: writing {cnt if cnt is not None else 'unknown'} events to {sub_out}")
                mask.to_parquet(sub_out)

            # 4 or more
            mask4 = ddf[ddf["njets_nominal"] >= 4]
            sub_out4 = os.path.join(out_dir, "njets_4plus")
            os.makedirs(sub_out4, exist_ok=True)
            try:
                cnt4 = int(mask4.count().compute())
            except Exception:
                cnt4 = None
            print(f"  -> njets_4plus: writing {cnt4 if cnt4 is not None else 'unknown'} events to {sub_out4}")
            mask4.to_parquet(sub_out4)

        except Exception as e:
            print(f"Error while writing parquet for {subdir}: {e}")
            try:
                ddf.to_parquet(out_dir)
            except Exception as e2:
                print(f"Fallback write also failed: {e2}")

    print("Skim complete.")


if __name__ == "__main__":
    # example usage
    input_root = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_AK8jets/stage1_output/2016preVFP/f1_0/"
    out_root = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn_AK8jets/2016preVFP/"

    feature_columns = [
        'mu1_pt','mu1_eta','mu1_phi',
        'mu2_pt','mu2_eta','mu2_phi',
        'dimuon_pt','dimuon_pt_log','dimuon_rapidity','dimuon_phi_cs',
        'dimuon_ebe_mass_res','dimuon_ebe_mass_res_rel','dimuon_cos_theta_cs',
        'njets_nominal',
        'jet1_pt_nominal','jet1_eta_nominal','jet1_phi_nominal','jet1_qgl_nominal',
        'jet2_pt_nominal','jet2_eta_nominal','jet2_phi_nominal','jet2_qgl_nominal',
        'jet3_pt_nominal','jet3_eta_nominal','jet3_phi_nominal','jet3_qgl_nominal',
        'jet4_pt_nominal','jet4_eta_nominal','jet4_phi_nominal','jet4_qgl_nominal',
        'jj_mass_nominal','jj_mass_log_nominal','jj_dEta_nominal','jj_dPhi_nominal',
        'htsoft2_nominal','nsoftjets5_nominal',
        'rpt_nominal','mmj_min_dEta_nominal','mmj_min_dPhi_nominal',
        'll_zstar_log_nominal','pt_centrality_nominal','zeppenfeld_nominal',
        'nBtagLoose_nominal','nBtagMedium_nominal',
        'nfatJets_drmuon','MET_pt','year'
    ]

    # If using a Dask gateway, create client here (optional)
    try:
        from dask_gateway import Gateway

        gateway = Gateway(
            "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
            proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
        )
        cluster_info = gateway.list_clusters()[0]
        client = gateway.connect(cluster_info.name).get_client()
        print("Gateway Client created")
    except Exception:
        pass

    skim_for_dnn(input_root, out_root, feature_columns)
