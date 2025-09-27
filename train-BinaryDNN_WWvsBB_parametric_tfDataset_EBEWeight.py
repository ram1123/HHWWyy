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
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Nadam

import keras_tuner as kt
from tensorflow.keras import regularizers

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
from sklearn.utils.class_weight import compute_class_weight

from rich import print

import random

# Seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Initialize StandardScaler
scaler = StandardScaler()

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


CURRENT_DATETIME = datetime.now()

# Metrics for evaluation
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=3),
    tf.keras.metrics.Precision(name="precision", top_k=1),
    tf.keras.metrics.Recall(name="recall", top_k=1),
    tf.keras.metrics.CategoricalCrossentropy(name="crossentropy"),
]


def make_ebe_weights(df, col="dimuon_ebe_mass_res", power=2, clip=(1e-6, None)):
    s = df[col].astype("float32").to_numpy()
    if clip[0] is not None:
        s = np.maximum(s, clip[0])
    if clip[1] is not None:
        s = np.minimum(s, clip[1])
    w = 1.0 / np.power(s, power)  # inverse-variance weighting
    w = w / np.mean(w)  # normalize to mean 1 (keeps loss scale stable)
    return w.astype("float32")


class BatchSizeTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = hp.Choice("batch_size", [128, 256, 512])

        model = self.hypermodel.build(hp)
        history = model.fit(*args, batch_size=bs, **kwargs)

        results = {k: v[-1] for k, v in history.history.items()}
        self.oracle.update_trial(trial.trial_id, results)
        # optional: save a snapshot of the model for this trial
        try:
            self.save_model(trial.trial_id, model)
        except NotImplementedError:
            pass  # if you decide not to implement save_model
        return results

    # only needed if you keep the save_model call above
    def save_model(self, trial_id, model, step=0):
        trial_dir = os.path.join(self.project_dir, f"trial_{trial_id}")
        os.makedirs(trial_dir, exist_ok=True)
        path = os.path.join(trial_dir, f"model_at_{step}.keras")
        model.save(path)


def build_model_from_hp_values(hp_values, input_dim, n_classes=3):
    # Turn a dict of fixed values into a HyperParameters object
    hp = kt.HyperParameters()
    for k, v in hp_values.items():
        hp.Fixed(k, v)
    return build_model_hp(hp, input_dim=input_dim, n_classes=n_classes)


def load_best_hp_json(outdir):
    hp_json = os.path.join(outdir, "best_hparams_bayes.json")
    if os.path.exists(hp_json):
        with open(hp_json, "r") as f:
            return json.load(f)
    return None


def hyperparam_scan(X_train, Y_train, X_val, Y_val, input_dim, output_dir,
                    trials=8, epochs=10, batch_size=256, class_weight=None):
    """
    Lightweight random search over a small grid. Returns (best_model, best_hist, best_cfg).
    """
    import itertools, random
    os.makedirs(output_dir, exist_ok=True)

    activations   = ['relu', 'gelu']
    dropouts      = [0.1, 0.2, 0.3]
    learn_rates   = [1e-4, 3e-4, 1e-3]
    widths        = [256, 384]
    depths        = [2, 3]  # number of hidden blocks before the final 64

    def make_model(cfg):
        act, dr, lr, width, depth = cfg
        layers = [Input(shape=(input_dim,))]
        for _ in range(depth):
            layers += [Dense(width, activation=act), BatchNormalization(), Dropout(dr)]
        layers += [Dense(64, activation=act), Dense(3, activation='softmax')]
        m = Sequential(layers)
        opt = Nadam(learning_rate=lr, clipnorm=1.0)
        m.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=METRICS)
        return m

    space = list(itertools.product(activations, dropouts, learn_rates, widths, depths))
    random.shuffle(space)
    space = space[:trials]

    best = None
    best_hist = None
    best_cfg = None

    for i, cfg in enumerate(space, 1):
        act, dr, lr, width, depth = cfg
        print(f"[scan {i}/{len(space)}] act={act} dr={dr} lr={lr} width={width} depth={depth}")

        model = make_model(cfg)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            CSVLogger(os.path.join(args.output_dir, 'training.log')),
            LearningRateScheduler(custom_learning_rate_scheduler),
            ModelCheckpoint(os.path.join(args.output_dir, 'model.best.keras'),
                            monitor='val_loss', save_best_only=True),
        ]

        hist = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=callbacks,
            # class_weight=class_weight,
        )

        # Use val_loss, with val_auc as tiebreaker
        val_loss = np.min(hist.history['val_loss'])
        val_auc  = np.max(hist.history.get('val_auc', [0.0]))
        score = (val_loss, -val_auc)

        if (best is None) or (score < best[0]):
            best = (score, model)
            best_hist = hist
            best_cfg = dict(activation=act, dropout=dr, learn_rate=lr, width=width, depth=depth)

    print("[scan] best:", best_cfg)
    # save best config for reproducibility
    with open(os.path.join(output_dir, "best_hparams.json"), "w") as f:
        json.dump(best_cfg, f, indent=2)
    return best[1], best_hist, best_cfg


def build_model_hp(hp, input_dim, n_classes=3):
    act = hp.Choice('activation', ['relu','gelu','swish'])              # ↑ added swish
    depth = hp.Int('depth', 2, 5)                                       # 2–5 blocks
    width = hp.Int('width', min_value=128, max_value=768, step=128)     # 128..768
    dropout = hp.Float('dropout', 0.05, 0.40, step=0.05)                # 5%..40%
    l2 = hp.Float('l2', 1e-6, 1e-3, sampling='log')                     # L2 reg
    lr = hp.Float('lr', 3e-5, 2e-3, sampling='log')                     # LR (log)
    final_width = hp.Choice('final_width', [32, 64, 96, 128])           # last hidden
    bs = hp.Choice('batch_size', [128, 256, 512])

    layers = [Input(shape=(input_dim,))]
    for _ in range(depth):
        layers += [
            Dense(width, activation=act, kernel_regularizer=regularizers.l2(l2)),
            BatchNormalization(),
            Dropout(dropout),
        ]
    layers += [
        Dense(final_width, activation=act, kernel_regularizer=regularizers.l2(l2)),
        Dense(n_classes, activation='softmax')
    ]
    model = Sequential(layers)

    opt = Nadam(learning_rate=lr, clipnorm=1.0)  # stable default
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=3),
            tf.keras.metrics.Precision(name="precision", top_k=1),
            tf.keras.metrics.Recall(name="recall", top_k=1),
            tf.keras.metrics.CategoricalCrossentropy(name="crossentropy"),
        ],
    )
    return model


def run_bayes_opt(X_train, Y_train, X_val, Y_val, input_dim, outdir, max_trials=40, executions_per_trial=1, epochs=30, class_weight=None):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        CSVLogger(os.path.join(args.output_dir, 'training.log')),
        LearningRateScheduler(custom_learning_rate_scheduler),
        ModelCheckpoint(os.path.join(args.output_dir, 'model.best.keras'),
                        monitor='val_loss', save_best_only=True),
    ]

    tuner = BatchSizeTuner(
        hypermodel=lambda hp: build_model_hp(hp, input_dim=input_dim, n_classes=Y_train.shape[1]),
        objective=kt.Objective('val_auc', direction='max'),
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=outdir,
        project_name='bayes_tuner',
        overwrite=True,
    )
    tuner.search(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        # class_weight=class_weight,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_bs = best_hp.get('batch_size')  # ← tuned batch size
    best_model = tuner.hypermodel.build(best_hp)
    history = best_model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=best_bs,
        callbacks=callbacks,
        verbose=1,
        # class_weight=class_weight,
    )

    # save artifacts
    with open(os.path.join(outdir, "best_hparams_bayes.json"), "w") as f:
        json.dump(best_hp.values, f, indent=2)
    best_model.save(os.path.join(outdir, "model_bayes_best.keras"))

    return best_model, history, best_hp

def _read_proc_ddf(base, proc, variables):
    pat = os.path.join(base, proc, "*.parquet")
    print(f"Reading parquet files from: {pat}")
    # print
    return dd.read_parquet(pat, columns=variables)

def load_from_parquet_to_numpy(inputPath, variables, num_events, weight_col='dimuon_ebe_mass_res'):
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
        ddf = _read_proc_ddf(inputPath, proc, variables + [weight_col])
        n_take = int(num_events) if (num_events and num_events > 0) else None
        df = ddf.head(n_take, compute=True) if n_take else ddf.compute()
        if df.empty:
            print(f"[warn] No rows for {proc} under {inputPath}")
            continue
        df = df.copy()
        df["target"] = meta["target"]
        df["process_ID"] = meta["process_ID"]
        parts.append(df)

    if not parts:
        raise RuntimeError("No data loaded from parquet. Check paths/variables.")
    df_all = pd.concat(parts, ignore_index=True)

    # features, labels, ebe weights
    X = df_all[variables].to_numpy(dtype=np.float32)
    y = df_all["target"].to_numpy(dtype=np.int64)
    w_ebe = make_ebe_weights(df_all, col=weight_col)

    return X, y, w_ebe

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


# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return float(lr)  # respect tuned lr
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


# Train the model with early stopping
def train_model(model, train_ds, val_ds, epochs, output_dir, class_weight=None):
    early_stopping = EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True)
    csv_logger = CSVLogger(os.path.join(output_dir, 'training.log'))
    lr_scheduler = LearningRateScheduler(custom_learning_rate_scheduler)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, csv_logger, lr_scheduler],
        # class_weight=class_weight,
        verbose=1
    )
    return history


def make_dataset(X, Y, W, batch_size, train=True, shuffle_buf=10000, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((X, Y, W))
    if train:
        ds = ds.shuffle(buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a multi-class DNN for ggH/VBF/background classification.")
    parser.add_argument('--inputPath', required=True, help="Base path containing parquet directories per process.")
    parser.add_argument('--output_dir', required=True, help="Directory to save outputs.")
    parser.add_argument('--retrain', action='store_true', help="Retrain the model.")
    parser.add_argument('--job_name', type=str, default="DNN", help="Job name.")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_events', type=int, default=1000, help="Number of events to load.")
    parser.add_argument('--json', type=str, default='input_variables.json', help="Input variable JSON file.")
    parser.add_argument('--use_gateway', action='store_true', help="Use Dask Gateway for distributed computing.")
    parser.add_argument('--bayes', action='store_true', help='Run Bayesian hyperparam optimization.')
    parser.add_argument('--max_trials', type=int, default=40, help='Bayesian max trials.')
    parser.add_argument('--executions_per_trial', type=int, default=1, help='KerasTuner executions per trial (average).')
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
    variables = []
    with open(args.json, "r") as f:
        variables = [k for k, _ in json.load(f).items()]

    print(f"Variables: {variables}")

    # Define paths and parameters
    model_path = os.path.join(args.output_dir, "model.keras")
    npz_cache = os.path.join(args.output_dir, "dataset_trainval.npz")
    scaler_cache = os.path.join(args.output_dir, "scaler.npz")

    # variables already built from JSON:
    feature_columns = [col for col in variables if col not in ['target','process_ID','classweight']]
    print(f"Feature columns: {feature_columns}")

    if os.path.exists(npz_cache) and os.path.exists(scaler_cache) and (not args.retrain):
        print(f"[cache] loading arrays from {npz_cache}")
        data = load_npz_dataset(npz_cache)
        X_train, X_val = data["X_train"], data["X_val"]
        Y_train, Y_val = data["Y_train"], data["Y_val"]
        y_train, y_val = data["y_train"], data["y_val"]
        Wtrain, Wval   = data["Wtrain"],  data["Wval"]

        class_ids = np.arange(Y_train.shape[1])
        class_weight_vals = compute_class_weight(
            class_weight='balanced',
            classes=class_ids,
            y=np.argmax(Y_train, axis=1),
        )
        class_weight = {int(c): float(w) for c, w in zip(class_ids, class_weight_vals)}
        print(f"class weights (from cache): {class_weight}")
    else:
        # 1) parquet -> numpy
        X, y, W_ebe = load_from_parquet_to_numpy(args.inputPath, feature_columns, args.num_events)

        # 2) one-hot labels
        n_classes = 3
        Y = np.eye(n_classes, dtype=np.float32)[y]

        X = np.asarray(X, dtype=np.float32)
        X[~np.isfinite(X)] = 0.0  # Replace NaN and inf with 0

        # 3) train/val split (keep y for stratify and to build class weights)
        X_train, X_val, Y_train, Y_val, y_train, y_val, Wtrain_ebe, Wval_ebe = train_test_split(
            X, Y, y, W_ebe, test_size=0.1, random_state=SEED, stratify=y
        )
        # class weights from y_train
        class_ids = np.arange(Y_train.shape[1])
        class_weight_vals = compute_class_weight(
            class_weight='balanced',
            classes=class_ids,
            y=y_train)
        cw_map = np.asarray([class_weight_vals[i] for i in class_ids], dtype=np.float32)

        Wtrain = Wtrain_ebe * cw_map[y_train]
        Wval   = Wval_ebe   * cw_map[y_val]

        # (optional) re-normalize to mean 1 to keep loss scale stable
        Wtrain = Wtrain / np.mean(Wtrain)
        Wval   = Wval   / np.mean(Wval)

        # 4) scale (fit on train only), but DO NOT scale any special categorical column (if you add one later)
        X_train_df = pd.DataFrame(X_train, columns=feature_columns)
        X_val_df   = pd.DataFrame(X_val,   columns=feature_columns)
        X_train = scaler.fit_transform(X_train_df).astype(np.float32)
        X_val   = scaler.transform(X_val_df).astype(np.float32)

        # guard against zero-variance columns causing Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

        # 5) save arrays + scaler
        save_npz_dataset(npz_cache, X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val,
                        y_train=y_train, y_val=y_val,
                        Wtrain=Wtrain, Wval=Wval,
                        features=np.array(feature_columns))
        save_scaler_npz(scaler_cache, scaler, feature_columns)

    print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Build datasets (always). Use tuned batch size if available.
    tuned_bs = None
    best_hp_vals = load_best_hp_json(args.output_dir)
    if best_hp_vals is not None:
        tuned_bs = int(best_hp_vals.get("batch_size", args.batch_size))
    bs = tuned_bs if tuned_bs is not None else args.batch_size

    train_ds = make_dataset(X_train, Y_train, Wtrain, batch_size=bs, train=True)
    val_ds   = make_dataset(X_val,   Y_val,   Wval,   batch_size=bs, train=False)

    history = None
    if os.path.exists(model_path) and (not args.retrain) and (not args.bayes):
        print(f"Trained model already exists at {model_path}. Loading the model...")
        model = load_model(model_path)
    else:
        if args.bayes:
            model, history, best_hp = run_bayes_opt(
                X_train, Y_train, X_val, Y_val,
                input_dim=X_train.shape[1],
                outdir=args.output_dir,
                max_trials=args.max_trials,
                executions_per_trial=args.executions_per_trial,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            print("Best HP:", best_hp.values)
            # also save a copy under the standard name
            model.save(model_path)
        else:
            if best_hp_vals is not None:
                print("[bayes] Using last saved best hyperparameters from JSON.")
                model = build_model_from_hp_values(best_hp_vals, input_dim=X_train.shape[1], n_classes=Y_train.shape[1])
                tuned_bs = int(best_hp_vals.get("batch_size", args.batch_size))
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    CSVLogger(os.path.join(args.output_dir, 'training.log')),
                    LearningRateScheduler(custom_learning_rate_scheduler),
                    ModelCheckpoint(os.path.join(args.output_dir, 'model.best.keras'),
                                    monitor='val_loss', save_best_only=True),
                ]

                # Train with datasets
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.epochs,
                    callbacks=callbacks,
                    # class_weight=class_weight,
                    verbose=1,
                )
                model.save(model_path)
                print(f"[bayes] Re-trained with best HP (bs={tuned_bs}). Saved to: {model_path}")
            else:
                # fallback: plain model
                model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)
                # Train model (simple multiclass DNN)
                history = train_model(
                    model, train_ds, val_ds,
                    epochs=args.epochs,
                    output_dir=args.output_dir,
                    # class_weight=class_weight,
                )
                # Evaluate and save model
                model.save(model_path)
                print(f"Saved model to: {model_path}")

    # Evaluate and plot on validation set (no mass filtering)
    y_prob = model.predict(X_val)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = np.argmax(Y_val, axis=1)

    print("="*40)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=['ggh','vbf','bkg'], digits=3))
    print("="*40)

    # ROC Curve
    plot_roc_curve_multiclass(Y_val, y_prob, plots_dir, labels=["ggh", "vbf", "bkg"], mass=None)

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

    # plot shap values
    # This is just commented out for now since it takes a long time to run
    # plot_shap_values(model, X_train, feature_columns, plots_dir)


if __name__ == "__main__":
    main()
