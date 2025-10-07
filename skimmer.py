# skim_dnn.py
import os
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask_awkward as dak

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

def skim_for_dnn(input_root, out_root, feature_columns):
    """
    input_root: /.../stage1_output/2018/compacted/
    out_root:   /.../skimmed_for_dnn/
    feature_columns: list of columns to keep (your DNN features)
    """
    os.makedirs(out_root, exist_ok=True)

    specs = {
        # "SampleName": (category_tag, if DY then true else false)
        "ggh_powhegPS":        ("notbtag", "False"),
        "vbf_powheg_dipole":   ("notbtag", "False"),

        # backgrounds (union of ggh or vbf selections)
        "dy_VBF_filter":       ("notbtag", "True"),
        "dy_M-100To200_MiNNLO":("notbtag", "True"),
        "dy_M-50_MiNNLO":      ("notbtag", "True"),
        "ewk_lljj_mll50_mjj120":("notbtag", "False"),
        "ttjets_dl":           ("notbtag", "False"),
        "ttjets_sl":           ("notbtag", "False"),

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
        ddf = dak.from_parquet(pattern, columns=feature_columns+additional_columns_for_selection)

        # apply selection
        ddf = selection.applyRegionCatCuts(ddf,
                                        category=tag,
                                        region_name="h-peak",
                                        process=subdir,
                                        variation="nominal",
                                        do_vbf_filter_study=filter_func)

        out_dir = os.path.join(out_root, subdir)
        print(f"Writing skim to: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        ddf.to_parquet(out_dir)

    print("Skim complete.")


input_root = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2018/compacted/"
out_root   = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"

input_root = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2017/compacted/"
out_root = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2017/"

input_root = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2016preVFP/compacted/"
out_root = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2016preVFP/"

input_root = "/depot/cms/hmm/shar1172/hmm_ntuples/copperheadV1clean/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/stage1_output/2016postVFP/compacted/"
out_root = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2016postVFP/"

feature_columns = [
    # exactly your training features:
    'mu1_eta','mu1_pt_over_mass','mu2_eta','mu2_pt_over_mass','dimuon_pt','dimuon_pt_log',
    'dimuon_rapidity','dimuon_ebe_mass_res','dimuon_ebe_mass_res_rel','dimuon_cos_theta_cs',
    'dimuon_phi_cs','jet1_pt_nominal','jet1_eta_nominal','jet1_phi_nominal','jet2_pt_nominal',
    'jet2_eta_nominal','jet2_phi_nominal','jet1_qgl_nominal','jet2_qgl_nominal','njets_nominal',
    'jj_mass_nominal','jj_mass_log_nominal','jj_dEta_nominal','jj_dPhi_nominal','htsoft2_nominal',
    'nsoftjets5_nominal','rpt_nominal','mmj_min_dEta_nominal','mmj_min_dPhi_nominal',
    'll_zstar_log_nominal','pt_centrality_nominal','zeppenfeld_nominal',

    'nBtagLoose_nominal','nBtagMedium_nominal',  # for btag veto

    'year'
]

from dask_gateway import Gateway
gateway = Gateway(
    "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
)
cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
client = gateway.connect(cluster_info.name).get_client()
print("Gateway Client created")

skim_for_dnn(input_root, out_root, feature_columns)
