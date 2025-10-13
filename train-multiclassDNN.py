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
from tensorflow.keras.callbacks import ModelCheckpoint
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
# scaler = StandardScaler()

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


CURRENT_DATETIME = datetime.now()

# Metrics for evaluation
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=5),
    tf.keras.metrics.Precision(name="precision", top_k=1),
    tf.keras.metrics.Recall(name="recall", top_k=1),
    tf.keras.metrics.CategoricalCrossentropy(name="crossentropy"),
]


class BatchSizeTuner(kt.BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        bs = hp.Choice("batch_size", [512, 1024, 2048, 5120, 8192, 10240, 20480, 30720])
        print(f"[tuner] trial {trial.trial_id} using batch size: {bs}")
        # kwargs["batch_size"] = bs

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


def build_model_from_hp_values(hp_values, input_dim, n_classes=5):
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

def build_model_hp(hp, input_dim, n_classes=5):
    act = hp.Choice('activation', ['relu', 'gelu', 'swish'])
    depth = hp.Int('depth', 2, 9)
    width = hp.Int('width', min_value=128, max_value=2560, step=128)
    dropout = hp.Float('dropout', 0.05, 0.40, step=0.05)  # 5%..40%
    l2 = hp.Float('l2', 1e-6, 1e-3, sampling='log')  # L2 reg
    lr = hp.Float('lr', 3e-5, 2e-3, sampling='log')  # LR (log)
    bs = hp.Choice("batch_size", [512, 1024, 2048, 5120, 8192, 10240, 20480, 30720])

    # Below two are there for the pyramid shape, having decreasing widths
    decay_rate = hp.Float('decay_rate', 0.65, 0.95, step=0.05)  # width decay rate per layer
    min_width = hp.Int('min_width', 32, 256, step=32)  # minimum width

    print(f"[tuner] using batch size: {bs}, depth: {depth}, width: {width}, dropout: {dropout:.3f}, l2: {l2:.1e}, lr: {lr:.1e}, act: {act}, decay_rate: {decay_rate:.2f}, min_width: {min_width}")

    layers = [Input(shape=(input_dim,))]
    for i in range(depth):
        width_i = max(int(width * (decay_rate ** i)), min_width)
        print(f'Layer {i}: width {width_i}')
        layers += [
            Dense(width_i, activation=act, kernel_regularizer=regularizers.l2(l2)),
            BatchNormalization(),
            Dropout(dropout),
        ]
    layers += [Dense(n_classes, activation='softmax')]
    model = Sequential(layers)

    opt = Nadam(learning_rate=lr, clipnorm=1.0)  # stable default
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=5),
            tf.keras.metrics.Precision(name="precision", top_k=1),
            tf.keras.metrics.Recall(name="recall", top_k=1),
            tf.keras.metrics.CategoricalCrossentropy(name="crossentropy"),
        ],
    )
    return model


def run_bayes_opt(X_train, Y_train, X_val, Y_val, input_dim, outdir, max_trials=40, executions_per_trial=1, epochs=30, class_weight=None, batch_size=256):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        CSVLogger(os.path.join(outdir, 'training.log')),
        LearningRateScheduler(custom_learning_rate_scheduler),
        ModelCheckpoint(os.path.join(outdir, 'model.best.keras'),
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
        class_weight=class_weight,
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
        class_weight=class_weight,
    )

    # save artifacts
    with open(os.path.join(outdir, "best_hparams_bayes.json"), "w") as f:
        json.dump(best_hp.values, f, indent=2)
    best_model.save(os.path.join(outdir, "model_bayes_best.keras"))

    return best_model, history, best_hp

def _read_proc_ddf(base, proc, variables):
    # base is of type path
    base = Path(base)
    print(f"Reading process: {proc}")
    print(f"Reading path: {base}")
    # pat = os.path.join(base, "/**/", proc, "*.parquet")
    year = "**"
    year = 2017
    pat = str(base / year / proc / "*.parquet")
    print(f"Reading parquet files from: {pat}")
    # print
    return dd.read_parquet(pat, columns=variables)

def load_from_parquet_to_numpy(inputPath, variables, num_events):
    """
    Reads all parquet under inputPath/{ggh,vbf,bkg}/ recursively via Dask,
    takes up to num_events rows per process, returns (X, y) as numpy arrays.
    """
    specs = {
        "vbf_powheg_dipole": dict(target=0, process_ID="vbf"),
        "ggh_powhegPS": dict(target=1, process_ID="ggh"),

        "dy_VBF_filter": dict(target=2, process_ID="DY"),
        "dy_M-100To200_MiNNLO": dict(target=2, process_ID="DY"),
        "dy_M-50_MiNNLO": dict(target=2, process_ID="DY"),

        "ewk_lljj_mll50_mjj120": dict(target=3, process_ID="EWK"),

        "ttjets_dl": dict(target=4, process_ID="TOP"),
        "ttjets_sl": dict(target=4, process_ID="TOP"),
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

        # ---- 1) Add missingness indicators from jet multiplicity ----
        # These are the key, compact indicators:
        df["has_jet1"]  = (df["njets_nominal"] >= 1).astype("int8")
        df["has_jet2"]  = (df["njets_nominal"] >= 2).astype("int8")
        df["has_dijet"] = (df["njets_nominal"] >= 2).astype("int8")  # dijet defined iff >=2 jets

        # Optional (often helpful): one-hot jet multiplicity
        df["is_njet0"]  = (df["njets_nominal"] == 0).astype("int8")
        df["is_njet1"]  = (df["njets_nominal"] == 1).astype("int8")
        df["is_njet2p"] = (df["njets_nominal"] >= 2).astype("int8")


        parts.append(df)


    indicator_cols = ["has_jet1","has_jet2","has_dijet","is_njet0","is_njet1","is_njet2p"]

    if not parts:
        raise RuntimeError("No data loaded from parquet. Check paths/variables.")
    df_all = pd.concat(parts, ignore_index=True)

    variables += indicator_cols


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

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    warmup_epochs = 3
    if epoch < warmup_epochs:
        return float(lr)  # respect tuned lr
    return float(lr * 0.98)


# Function to build the multi-class DNN model
def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001, n_classes=5):
    model = Sequential([
        Dense(256, input_shape=(input_dim,), activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(128, activation=activation),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation=activation),
        Dense(n_classes, activation="softmax")  # n_classes classification
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
        class_weight=class_weight,
        verbose=1
    )
    return history


def make_dataset(X, Y, batch_size, train=True, shuffle_buf=10000, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
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

    # Step-0: Get the copy of this code and the input variables JSON into the output dir. The name of files should be appended with the current date and time.
    code_path = Path(__file__).resolve()
    code_copy_path = os.path.join(args.output_dir, f"{code_path.stem}_{CURRENT_DATETIME.strftime('%Y%m%d_%H%M%S')}{code_path.suffix}")
    os.system(f"cp {code_path} {code_copy_path}")
    print(f"Copied code to: {code_copy_path}")

    json_copy_path = os.path.join(args.output_dir, f"{Path(args.json).stem}_{CURRENT_DATETIME.strftime('%Y%m%d_%H%M%S')}{Path(args.json).suffix}")
    os.system(f"cp {args.json} {json_copy_path}")
    print(f"Copied JSON to: {json_copy_path}")

    # Keep some basic info about the run in a text file and store it to the output dir
    with open(os.path.join(args.output_dir, "command.txt"), "w") as f:
        f.write(f"Date: {CURRENT_DATETIME}\n")
        f.write("Command run:\n")
        f.write(" ".join(sys.argv) + "\n")
        f.write("\n")
    # save git patch
    with open(os.path.join(args.output_dir, "git_patch.diff"), "w") as f:
        f.write(os.popen("git diff").read())

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
    feature_columns = [col for col in variables if col not in ['target','process_ID']]
    print(f"Feature columns: {feature_columns}")

    n_classes = 5  # vbf, ggh, DY, EWK, TOP
    CLASS_LABELS = ["vbf", "ggh", "DY", "EWK", "TOP"]

    if os.path.exists(npz_cache) and os.path.exists(scaler_cache) and (not args.retrain):
        print(f"[cache] loading arrays from {npz_cache}")
        data = load_npz_dataset(npz_cache)
        X_train = data["X_train"]; X_val = data["X_val"]
        Y_train = data["Y_train"]; Y_val = data["Y_val"]

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
        X, y = load_from_parquet_to_numpy(args.inputPath, feature_columns, args.num_events)

        # 2) one-hot labels
        Y = np.eye(n_classes, dtype=np.float32)[y]

        # X = np.asarray(X, dtype=np.float32)
        # # Replace -999 or -99 with -9
        # # Replace NaN and inf with -9
        X[~np.isfinite(X)] = -9.0
        X[X == -999.0] = -9.0
        X[X == -99.0] = -9.0

        # is there any feature with value <= -9?
        for i, col in enumerate(feature_columns):
            if np.any(X[:, i] <= -9):
                print(f"[warn] Feature {col} has values <= -9. Consider checking for missing values.")

        print("X shape:", X.shape, "y shape:", y.shape, "Y shape:", Y.shape)
        print(Y)

        # plot the distribution for reach feature before scaling and after scaling and save it in one pdf file
        if 1:
            n_features = X.shape[1]
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            axes = axes.flatten()
            for i in range(n_features):
                axes[i].hist(X[:, i], bins=50, alpha=0.7, color='blue', label='Before Scaling')
                axes[i].set_title(f'Feature {feature_columns[i]}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            for i in range(n_features, len(axes)):
                fig.delaxes(axes[i])  # Remove unused subplots
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'feature_distributions_before_scaling.pdf'))
            plt.close()

        # 3) train/val split
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=SEED, stratify=y)

        # print("After split:")
        # print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
        # print("Y_train shape:", Y_train.shape, "Y_val shape:", Y_val.shape)
        # print(Y_train)

        # class weights
        class_ids = np.arange(Y_train.shape[1])
        class_weight_vals = compute_class_weight(
            class_weight='balanced',
            classes=class_ids,
            y=np.argmax(Y_train, axis=1),
        )
        class_weight = {int(c): float(w) for c, w in zip(class_ids, class_weight_vals)}
        print(f"class weights: {class_weight}")

        # 4) scale (fit on train only), but DO NOT scale any special categorical column (if you add one later)
        # X_train_df = pd.DataFrame(X_train, columns=feature_columns)
        # X_val_df   = pd.DataFrame(X_val,   columns=feature_columns)
        # X_train = scaler.fit_transform(X_train_df).astype(np.float32)
        # X_val   = scaler.transform(X_val_df).astype(np.float32)

        # # guard against zero-variance columns causing Inf
        # X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        # X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)

        # plot each feature after scaling and save it in one pdf file
        # if 1:
        #     n_features = X_train.shape[1]
        #     n_cols = 4
        #     n_rows = (n_features + n_cols - 1) // n_cols
        #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        #     axes = axes.flatten()
        #     for i in range(n_features):
        #         axes[i].hist(X_train[:, i], bins=50, alpha=0.7, color='green', label='After Scaling')
        #         axes[i].set_title(f'Feature {feature_columns[i]}')
        #         axes[i].set_xlabel('Value')
        #         axes[i].set_ylabel('Frequency')
        #         axes[i].legend()
        #     for i in range(n_features, len(axes)):
        #         fig.delaxes(axes[i])  # Remove unused subplots
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(args.output_dir, 'feature_distributions_after_scaling.pdf'))
        #     plt.close()
        #     #

        # sys.exit()
        # 5) save arrays + scaler
        save_npz_dataset(npz_cache, X_train=X_train, X_val=X_val, Y_train=Y_train, Y_val=Y_val, features=np.array(feature_columns))
        # save_scaler_npz(scaler_cache, scaler, feature_columns)

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

    train_ds = make_dataset(X_train, Y_train, batch_size=bs, train=True)
    val_ds   = make_dataset(X_val,   Y_val,   batch_size=bs, train=False)

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
                class_weight=class_weight,
            )
            print("Best HP:", best_hp.values)
            # also save a copy under the standard name
            model.save(model_path)
        else:
            if best_hp_vals is not None:
                print("[bayes] Using last saved best hyperparameters from JSON.")
                print(best_hp_vals)

                model = build_model_from_hp_values(best_hp_vals, input_dim=X_train.shape[1], n_classes=Y_train.shape[1])
                model.summary()

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
                    class_weight=class_weight,
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
                    class_weight=class_weight,
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
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=3))
    print("="*40)

    # ROC Curve
    plot_roc_curve_multiclass(Y_val, y_prob, plots_dir, labels=CLASS_LABELS, mass=None)

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
    # plot_overfitting_multiclass(model = model, X_train=X_train, Y_train=Y_train, X_test=X_val, Y_test=Y_val, class_labels=CLASS_LABELS, output_dir=plots_dir)
    plot_overfitting_per_class(y_true=y_true, y_pred=y_pred, class_labels=CLASS_LABELS, output_dir=plots_dir)

    # Confusion Matrix
    plot_confusion_matrix_multiclass(Y_val, y_pred, plots_dir, labels=CLASS_LABELS, mass=None)

    # plot_correlation_matrix(X_train, plots_dir, feature_columns)

    # plot shap values
    plot_shap_values(model, X_train[:11000], feature_columns, plots_dir, max_display=15)


if __name__ == "__main__":
    main()
