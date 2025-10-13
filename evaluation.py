#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
import pandas as pd
import awkward as ak
import dask_awkward as dak
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from rich import print
from typing import List, Optional

import selection  # your module

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

INPUT_DIR = "/depot/cms/hmm/shar1172/hmm_ntuples/skimmed_for_dnn/2018/"
OUTPUT_DIR = "./evaluation_results/"
FEATURES_JSON = (
    "/depot/cms/private/users/shar1172/HHWWyy_DNN_For_HMuMu/input_variables.json"
)
MODEL_DIR = "/depot/cms/users/shar1172/HHWWyy_DNN_For_HMuMu/outputs/Run2_nanoAODv12_UpdatedQGL_FixPUJetIDWgt/DNN_relativeEBEonly/"

SCALER_NPZ = f"{MODEL_DIR}/scaler.npz"
MODEL_PATH = f"{MODEL_DIR}/model.keras"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

# Step-1: Print the model summary
model.summary()

# Step-2: Plot the model architecture to a file
plot_model(model, to_file=f"{OUTPUT_DIR}/model_structure.pdf", show_shapes=True, )

# Step-3: Evaluate the model on a test dataset
# def load_features(features_json: str) -> List[str]:
#     with open(features_json, "r") as f:
#         features = json.load(f)
#     return features["input_features"]

# input_features = load_features(FEATURES_JSON)
# print(f"Input features: {input_features}")

scaler_data = np.load(SCALER_NPZ)
print(f"Scaler data keys: {list(scaler_data.keys())}")
scaler_mean = scaler_data["mean"]
scaler_scale = scaler_data["scale"]
scalar_var = scaler_data["var"]
scaler_n_features_in = scaler_data["features"]
print(f"Scaler mean: {scaler_mean}")
print(f"Scaler scale: {scaler_scale}")
print(f"Scaler var: {scalar_var}")
print(f"Scaler n_features_in: {scaler_n_features_in}")
