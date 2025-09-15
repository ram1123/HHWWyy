import os
import sys
import tempfile
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import json
import argparse

import dask.dataframe as dd
import numpy as np
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam
import uproot

from plotting.plotter_New import plot_correlation_matrix
from plotting.plotter_New import plot_training_progress
from plotting.plotter_New import plot_training_progress_from_csv
from plotting.plotter_New import plot_metrics
from plotting.plotter_New import plot_confusion_matrix_multiclass
from plotting.plotter_New import plot_roc_curve_multiclass
from plotting.plotter_New import plot_shap_values
from plotting.plotter_New import plot_overfitting
from plotting.plotter_New import plot_overfitting_per_class
from plotting.plotter_New import plot_overfitting_multiclass
from plotting.plotter_New import plot_classifier_output
from sklearn.preprocessing import StandardScaler

# Initialize StandardScaler
scaler = StandardScaler()

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Seed for reproducibility
np.random.seed(7)

CURRENT_DATETIME = datetime.now()


# Load data from ROOT files into a DataFrame
def load_data(inputPath, variables, num_events, csv_path, metadata_path):
    """
    Load data for ggH, VBF, and Background processes. Background mass is randomly sampled from the signal mass range.

    :param inputPath: Path to the ROOT files.
    :param variables: List of variables to extract from the ROOT files.
    :param num_events: Number of events to read.
    :param csv_path: Path to save/load the CSV file.
    :param metadata_path: Path to save/load the metadata (variable list).
    :return: Pandas DataFrame containing the dataset.
    """
    csv_exists = os.path.exists(csv_path)
    metadata_exists = os.path.exists(metadata_path)

    # Check if the CSV file exists and the variables are unchanged
    if csv_exists and metadata_exists:
        with open(metadata_path, 'r') as metadata_file:
            saved_metadata = json.load(metadata_file)

        saved_variables = saved_metadata.get('variables', [])
        if set(variables) == set(saved_variables):
            print(f"Loading data from existing CSV: {csv_path}")
            return pd.read_csv(csv_path)
        else:
            print("Variable list has changed. Reloading data from ROOT files.")
    else:
        print("CSV or metadata not found. Reading data from ROOT files.")

    # Read data from ROOT files
    keys = ['ggh', 'vbf', 'bkg']
    data = pd.DataFrame(columns=variables + ['target', 'process_ID', 'classweight', 'mass'])
    for key in keys:
        if key == 'ggh':
            fileNames = ["GluGluHToZZTo2L2Nu_M125_TuneCP5_13TeV_powheg2_minloHJJ_JHUGenV735_pythia8"]
            target = 0  # ggH
        elif key == 'vbf':
            fileNames = ["VBF_HToZZTo2L2Nu_M125_TuneCP5_withDipoleRecoil_13TeV_powheg2_JHUGenV735_pythia8"]
            target = 1  # VBF
        else:  # Background
            fileNames = ["ZZTo2L2Nu"]
            target = 2  # Background

        for filen in fileNames:
            process_ID = key.upper()
            full_path = os.path.join(inputPath, f"{filen}.root")
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                continue
            tree = uproot.open(full_path)["Events"]
            chunk_df = tree.arrays(variables, library="pd", entry_stop=num_events)
            chunk_df['target'] = target
            chunk_df['process_ID'] = process_ID
            chunk_df['classweight'] = 1.0

            data = pd.concat([data, chunk_df], ignore_index=True)

            print(f"Loaded {len(chunk_df)} events for process {key}")

    # Save DataFrame to CSV
    print(f"Saving DataFrame to CSV: {csv_path}")
    data.to_csv(csv_path, index=False)

    # Save variable list to metadata
    metadata = {"variables": variables}
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return data


def _read_proc_ddf(base, proc, variables):
    pat = os.path.join(base, proc, "*.parquet")
    print(f"Reading parquet files from: {pat}")
    # print
    return dd.read_parquet(pat, columns=variables)

def load_from_parquet_to_numpy(inputPath, variables, num_events):
    """
    Reads all parquet under inputPath/{ggh,vbf,bkg}/ recursively via Dask,
    takes up to num_events rows per process, returns (X, y) as numpy arrays.
    """
    specs = {
        "ggh_powhegPS": dict(target=0, process_ID="ggh"),
        "vbf_powheg_dipole": dict(target=1, process_ID="vbf"),

        "dy_VBF_filter": dict(target=2, process_ID="bkg"),
        "dy_M-100To200_MiNNLO": dict(target=2, process_ID="bkg"),
        "dy_M-50_MiNNLO": dict(target=2, process_ID="bkg"),

        "ewk_lljj_mll50_mjj120": dict(target=2, process_ID="bkg"),

        "ttjets_dl": dict(target=2, process_ID="bkg"),
        "ttjets_sl": dict(target=2, process_ID="bkg"),
    }
    parts = []
    for proc, meta in specs.items():
        ddf = _read_proc_ddf(inputPath, proc, variables)
        n_take = int(num_events) if (num_events and num_events > 0) else None
        print(f"Taking {n_take} rows for {proc}")
        df = ddf.head(n_take, compute=True) if n_take else ddf.compute()
        if df.empty:
            print(f"[warn] No rows for {proc} under {inputPath}")
            continue
        df = df.copy()
        df["target"] = meta["target"]
        df["process_ID"] = meta["process_ID"]
        df["classweight"] = 1.0
        parts.append(df)

    if not parts:
        raise RuntimeError("No data loaded from parquet. Check paths/variables.")
    df_all = pd.concat(parts, ignore_index=True)

    X = df_all[variables].to_numpy(dtype=np.float32)
    y = df_all["target"].to_numpy(dtype=np.int64)
    return X, y

def save_npz_dataset(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    print(f"[cache] saved: {path}")

def load_npz_dataset(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def save_scaler_npz(path, scaler, feature_names):
    np.savez_compressed(
        path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        var=scaler.var_.astype(np.float32),
        features=np.array(feature_names)
    )
    print(f"[cache] saved scaler: {path}")

def load_scaler_npz(path):
    z = np.load(path, allow_pickle=False)
    return dict(mean=z["mean"], scale=z["scale"], var=z["var"], features=z["features"])


# Ensure input data is numeric and clean
def preprocess_data(data, exclude_columns=[]):
    """
    Added exclude columns to avoid scaling the mass column, as
    it is a categorical variable and should not be scaled.
    """
    # Replace NaN or infinite values with a default (e.g., 0 or mean)
    data = data.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    data = data.fillna(0)  # Replace NaN with 0 (or use column mean if needed)
    if exclude_columns:
        cols_to_scale = [col for col in data.columns if col not in exclude_columns]
        # Scale only the columns present in cols_to_scale
        scaled_df = pd.DataFrame(scaler.fit_transform(data[cols_to_scale]), columns=cols_to_scale)
        # Reattach the excluded columns with their original values
        for col in exclude_columns:
            scaled_df[col] = data[col].values
        # Reorder columns to match the original order
        scaled_df = scaled_df[data.columns]
    else:
        scaled_df = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return scaled_df.astype('float32')

# Metrics for evaluation
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.CategoricalCrossentropy(name='crossentropy')
]

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    if epoch < 3:
        return 3e-4
    else:
        return float(lr * 0.98)

# Function to build the multi-class DNN model
def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
    model = Sequential([
        Dense(256, input_shape=(input_dim,), activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(3, activation="softmax")  # 3-class classification
    ])

    opt = Nadam(learning_rate=learn_rate, clipnorm=1.0)  # Gradient clipping to prevent exploding gradients

    # Compile the model
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=METRICS)

    return model


def build_parametric_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
    """
    Build a parametric multi-class DNN model with an additional mass input.
    """
    model = Sequential([
        Dense(256, input_dim=input_dim, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(3, activation="softmax")  # Softmax for multi-class classification
    ])
    model.compile(optimizer=Nadam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=METRICS)
    return model

# Train the model with early stopping
def train_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, output_dir, class_weight=None):
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(output_dir, 'training.log'))
    lr_scheduler = LearningRateScheduler(custom_learning_rate_scheduler)

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, csv_logger, lr_scheduler],
        class_weight=class_weight,
        verbose=1
    )
    return history


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, output_path, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a multi-class DNN for ggH/VBF/background classification.")
    parser.add_argument('--inputPath', required=True, help="Path to input ROOT files.")
    parser.add_argument('--output_dir', required=True, help="Directory to save outputs.")
    parser.add_argument('--retrain', action='store_true', help="Retrain the model.")
    parser.add_argument('--job_name', type=str, default="DNN", help="Job name.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_events', type=int, default=1000, help="Number of events to load.")
    parser.add_argument('--json', type=str, default='input_variables.json', help="Input variable JSON file.")
    parser.add_argument('--use_gateway', action='store_true', help="Use Dask Gateway for distributed computing.")

    args = parser.parse_args()

    # if args.use_gateway:
    #     from dask_gateway import Gateway
    #     gateway = Gateway(
    #         "http://dask-gateway-k8s.geddes.rcac.purdue.edu/",
    #         proxy_address="traefik-dask-gateway-k8s.cms.geddes.rcac.purdue.edu:8786",
    #     )
    #     cluster_info = gateway.list_clusters()[0]# get the first cluster by default. There only should be one anyways
    #     client = gateway.connect(cluster_info.name).get_client()
    #     print("Gateway Client created")
    # else: # use local cluster
    #     from dask.distributed import Client
    #     client = Client(n_workers=15,  threads_per_worker=1, processes=True, memory_limit='30 GiB')
    #     print("Local scale Client created")

    args.output_dir = os.path.join(args.output_dir, f"{args.job_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Create list of headers for dataset .csv
    input_var_jsonFile = open(args.json,'r')
    variable_list = json.load(input_var_jsonFile).items()
    variables = []
    for key,var in variable_list:
        variables.append(key)

    print(f"Variables: {variables}")

    # Define paths and parameters
    csv_path = os.path.join(args.output_dir, "output_dataframe.csv")
    metadata_path = os.path.join(args.output_dir, "variables_metadata.json")
    model_path = os.path.join(args.output_dir, "model.keras")

    # cache paths
    npz_cache = os.path.join(args.output_dir, "dataset_trainval.npz")
    scaler_cache = os.path.join(args.output_dir, "scaler.npz")

    # variables already built from JSON:
    feature_columns = [col for col in variables if col not in ['target','process_ID','classweight']]
    print(f"Feature columns: {feature_columns}")

    if os.path.exists(npz_cache) and os.path.exists(scaler_cache) and (not args.retrain):
        print(f"[cache] loading arrays from {npz_cache}")
        data = load_npz_dataset(npz_cache)
        X_train = data["X_train"]; X_val = data["X_val"]
        Y_train = data["Y_train"]; Y_val = data["Y_val"]
    else:
        # 1) parquet -> numpy
        X, y = load_from_parquet_to_numpy(args.inputPath, feature_columns, args.num_events)

        # 2) one-hot labels
        n_classes = 3
        Y = np.eye(n_classes, dtype=np.float32)[y]

        X = np.asarray(X, dtype=np.float32)
        X[~np.isfinite(X)] = 0.0  # Replace NaN and inf with 0

        # 3) train/val split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=7, stratify=y)

        # 4) scale (fit on train only), but DO NOT scale any special categorical column (if you add one later)
        X_train_df = pd.DataFrame(X_train, columns=feature_columns)
        X_val_df   = pd.DataFrame(X_val,   columns=feature_columns)
        X_train = scaler.fit_transform(X_train_df).astype(np.float32)
        X_val   = scaler.transform(X_val_df).astype(np.float32)

        # guard against zero-variance columns causing Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

        # 5) save arrays + scaler
        save_npz_dataset(npz_cache, X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val, features=np.array(feature_columns))
        save_scaler_npz(scaler_cache, scaler, feature_columns)

    print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    history = None
    # Check if the model already exists
    if os.path.exists(model_path) and (not args.retrain):
        print(f"Trained model already exists at {model_path}. Loading the model...")
        model = load_model(model_path)
    else:
        # Build model (input_dim is now len(feature_columns))
        model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)

        # Train model (simple multiclass DNN)
        history = train_model(
            model, X_train, Y_train, X_val, Y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir
        )

        # Evaluate and save model

        model.save(model_path)
        print(f"Saved model to: {model_path}")

    # Evaluate and plot on validation set (no mass filtering)
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(Y_val, axis=1)
    y_score = model.predict(X_val)

    # ROC Curve
    plot_roc_curve_multiclass(Y_val, y_score, plots_dir, labels=["ggh", "vbf", "bkg"], mass=None)

    # Classification Report
    plot_classifier_output(model, X_train, Y_train, X_val, Y_val, output_dir=plots_dir, mass=None)

    # Plot training progress
    plots_dir = os.path.join(args.output_dir, "plots")
    csv_log   = os.path.join(args.output_dir, "training.log")

    # only call the regular plot if we actually trained this run
    if history is not None and hasattr(history, "history"):
        plot_training_progress(history, plots_dir)
    else:
        # fallback to CSV log produced in a previous run
        plot_training_progress_from_csv(csv_log, plots_dir)


    plot_metrics(history, plots_dir, csv_log)
    plot_overfitting_multiclass(model = model, X_train=X_train, Y_train=Y_train, X_test=X_val, Y_test=Y_val, class_labels=["ggh", "vbf", "bkg"], output_dir=plots_dir)
    plot_overfitting_per_class(y_true=y_true, y_pred=y_pred, class_labels=["ggh", "vbf", "bkg"], output_dir=plots_dir)

    # Confusion Matrix
    plot_confusion_matrix_multiclass(Y_val, y_pred, plots_dir, labels=["ggh", "vbf", "bkg"], mass=None)

    plot_correlation_matrix(X_train, plots_dir, feature_columns)


if __name__ == "__main__":
    main()
