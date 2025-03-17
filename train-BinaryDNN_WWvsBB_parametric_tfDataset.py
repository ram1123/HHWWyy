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
from plotting.plotter import plotter

from plotting.plotter_New import plot_correlation_matrix
from plotting.plotter_New import plot_training_progress
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

class PerMassMetricsCallback(Callback):
    def __init__(self, model_reference, val_dataset, masses, output_dir, batch_size=32):
        """
        Callback to compute accuracy and loss for each mass point during training.

        :param model_reference: Reference to the trained model.
        :param val_dataset: tf.data.Dataset for validation (features + labels).
        :param masses: List of unique mass values.
        :param output_dir: Directory to save metric plots.
        :param batch_size: Batch size for evaluation.
        """
        super().__init__()
        self.model_reference = model_reference  # Store model reference
        # self.val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Ensure batched & optimized dataset
        # Instead of re-batching, create an unbatched copy for filtering.
        self.masses = masses
        self.val_dataset_unbatched = val_dataset.unbatch().prefetch(tf.data.AUTOTUNE)        self.output_dir = output_dir
        self.batch_size = batch_size
        self.history = {mass: {'accuracy': [], 'loss': []} for mass in masses}

    def filter_by_mass1(self, dataset, mass_value, batch_size=32):
        def filter_fn(x, y):
            tf.print("Mass Value in Sample:", x[-1])  # Debugging
            return tf.equal(x[-1], mass_value)  # Ensure x[-1] is the mass column
        # ✅ Fix: Extract mass from the last column (correct per-sample filtering)
        for f,l in dataset.take(1):
            tf.print("Mass Value in Sample:", f[-1])  # Debugging
        #filtered_dataset = dataset.filter(lambda x, y: tf.equal(x[:, -1], mass_value))
        filtered_dataset = dataset.filter(lambda x, y: tf.equal(tf.squeeze(x[-1]), mass_value))

        # ✅ Fix: Ensure consistent batch sizes (drop incomplete batches)
        filtered_dataset = filtered_dataset.batch(batch_size, drop_remainder=True)
        # Apply filtering sample-wise
        #filtered_dataset = dataset.filter(lambda x, y: tf.squeeze(filter_fn(x, y)))

        # Batch the dataset correctly, dropping incomplete batches
        #filtered_dataset = filtered_dataset.batch(batch_size, drop_remainder=True)

        return filtered_dataset

    def filter_by_mass2(self, dataset, mass_value, batch_size=32):
        def filter_fn(x, y):
            return tf.equal(tf.squeeze(x[-1]), tf.cast(mass_value, tf.float32))

        # ✅ Apply filtering sample-wise (ensuring scalar bool output)
        filtered_dataset = dataset.filter(lambda x, y: tf.reshape(filter_fn(x, y), []))

        # ✅ Ensure batch size consistency to avoid shape mismatches
        filtered_dataset = filtered_dataset.batch(batch_size, drop_remainder=True)

        return filtered_dataset

    def filter_by_mass(self, dataset, mass_value, batch_size=32):
        def filter_fn(x, y):
            # Extract the last column (mass feature) per individual sample
            mass_column = x[..., -1]  # Correctly extracts mass for each row
            return tf.equal(mass_column, tf.cast(mass_value, tf.float32))  # Element-wise comparison

        # ✅ Corrected: Apply filtering **per sample** and ensure scalar boolean output
        filtered_dataset = dataset.filter(lambda x, y: tf.equal(tf.squeeze(x[..., -1]), tf.cast(mass_value, tf.float32)))

        # ✅ Ensure all batches have the same shape to avoid "Cannot batch tensors with different shapes" errors
        filtered_dataset = filtered_dataset.batch(batch_size, drop_remainder=True)

        return filtered_dataset

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEvaluating per mass metrics after epoch {epoch + 1}...")

        for mass in self.masses:
            # mass_dataset = self.filter_by_mass(self.val_dataset, mass)
            mass_dataset = self.filter_by_mass(self.val_dataset_unbatched, mass)
            # Safely check if dataset is empty
            try:
                next(iter(mass_dataset.take(1)))  # Try fetching one batch
            except StopIteration:
                continue  # Skip this mass if no data

            metrics = self.model.evaluate(mass_dataset, verbose=0)
            self.history[mass]['loss'].append(metrics[0])
            self.history[mass]['accuracy'].append(metrics[1])



    def plot_metrics(self):
        """
        Generate and save accuracy vs. epoch and loss vs. epoch plots for each mass.
        """
        for mass in self.masses:
            if len(self.history[mass]['accuracy']) == 0:
                continue  # Skip if no data for this mass

            # Plot accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(self.history[mass]['accuracy'], label=f"Mass {mass}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy vs. Epoch for Mass {mass}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/accuracy_vs_epoch_mass_{mass}.png")
            plt.close()

            # Plot loss
            plt.figure(figsize=(8, 6))
            plt.plot(self.history[mass]['loss'], label=f"Mass {mass}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Loss vs. Epoch for Mass {mass}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/loss_vs_epoch_mass_{mass}.png")
            plt.close()

            print(f"Saved accuracy and loss plots for mass {mass}")

# Ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load data from ROOT files into a DataFrame
def load_data(inputPath, variables, num_events, csv_path, metadata_path, signal_masses):
    """
    Load data for ggH, VBF, and Background processes. Background mass is randomly sampled from the signal mass range.

    :param inputPath: Path to the ROOT files.
    :param variables: List of variables to extract from the ROOT files.
    :param num_events: Number of events to read.
    :param csv_path: Path to save/load the CSV file.
    :param metadata_path: Path to save/load the metadata (variable list).
    :param signal_masses: List of signal masses for ggH and VBF.
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
            fileNames = [f"GluGluHToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8" for mass in signal_masses]
            target = 0  # ggH
        elif key == 'vbf':
            fileNames = [f"VBF_HToZZTo2L2Nu_M{mass}_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8" for mass in signal_masses]
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

            mass = None
            # Assign mass for signals
            if key != 'bkg':
                mass = int(filen.split("_M")[1].split("_")[0])  # Extract mass from filename for signals
                chunk_df['mass'] = mass
            else:  # Assign mass for background randomly sampled from signal masses
                chunk_df['mass'] = np.random.choice(signal_masses, size=len(chunk_df))

            data = pd.concat([data, chunk_df], ignore_index=True)

            print(f"Loaded {len(chunk_df)} events for process {key} with mass {mass}")

    # Save DataFrame to CSV
    print(f"Saving DataFrame to CSV: {csv_path}")
    data.to_csv(csv_path, index=False)

    # Save variable list to metadata
    metadata = {"variables": variables}
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return data

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
]

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return 0.01
    else:
        return float(lr * tf.math.exp(-0.05 * (epoch - 10)))

# Build and compile a multi-class DNN model
#def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
#    model = Sequential([
#        Dense(256, input_dim=input_dim, activation=activation),
#        BatchNormalization(),
#        Dropout(dropout_rate),
#        Dense(128, activation=activation),
#        BatchNormalization(),
#        Dropout(dropout_rate),
#        Dense(64, activation=activation),
#        Dense(3, activation="softmax")  # Softmax for multi-class classification
#    ])
#    model.compile(optimizer=Nadam(learning_rate=learn_rate), loss='categorical_crossentropy', metrics=METRICS)
#    return model

# Function to build the model
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

    # Compile the model
    model.compile(optimizer=Nadam(learning_rate=learn_rate),
                  loss='categorical_crossentropy',
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

#def train_model_with_mass_metrics(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, output_dir, signal_masses, class_weight=None):
def train_model_with_mass_metrics(model, train_dataset, val_dataset, batch_size, epochs, output_dir, signal_masses, class_weight=None):
    """
    Train the model with accuracy and loss tracking for each mass point.

    :param model: The model to train.
    :param X_train: Training feature set (including the mass feature).
    :param Y_train: Training labels (one-hot encoded).
    :param X_val: Validation feature set (including the mass feature).
    :param Y_val: Validation labels (one-hot encoded).
    :param batch_size: Batch size for training.
    :param epochs: Number of epochs to train.
    :param output_dir: Directory to save plots and metrics.
    :param class_weight: Optional class weights for imbalanced data.
    :return: Training history and per mass metrics callback.
    """
    masses = np.unique(signal_masses)
    #per_mass_callback = PerMassMetricsCallback(model, X_val, Y_val, masses, output_dir)
    per_mass_callback =PerMassMetricsCallback(model, val_dataset, masses, output_dir)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[per_mass_callback],
        class_weight=class_weight,
        verbose=1
    )

    # Generate plots for per mass metrics
    per_mass_callback.plot_metrics()
    #per_mass_callback =1

    return history, per_mass_callback

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
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--num_events', type=int, default=1000, help="Number of events to load.")
    parser.add_argument('--json', type=str, default='input_variables.json', help="Input variable JSON file.")

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, f"{args.job_name}")
    ensure_directory_exists(args.output_dir)

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
    model_path = os.path.join(args.output_dir, "model.h5")

    # Load data
    signal_masses = [300, 400, 500, 1500, 2000, 3000]
    data = load_data(args.inputPath, variables, args.num_events, csv_path=csv_path, metadata_path=metadata_path, signal_masses=signal_masses)

    # print dataframe info
    #print(data.info())

    # print dataframe head
    #print(data.head())
    #print(data['mass'].unique())

    # Define feature columns explicitly (these are the features used for training)
    feature_columns = [col for col in variables if col not in ['target', 'process_ID', 'classweight', 'mass']]

    # add mass to the feature columns
    feature_columns.append('mass')

    print(f"Feature columns: {feature_columns}")

    # Extract only the features used for training
    X = data[feature_columns].values
    Y = pd.get_dummies(data['target']).values  # One-hot encoding for multi-class

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=7)

    # Debugging: Ensure 'mass' is in X
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    #print("Mass in validation data (last column of X_val):", X_val[:, -1])

    # Preprocess training and validation data
    X_train = preprocess_data(pd.DataFrame(X_train, columns=feature_columns), exclude_columns=["mass"]).values
    X_val = preprocess_data(pd.DataFrame(X_val, columns=feature_columns), exclude_columns=["mass"]).values
    # Do not scale Y since they are one values; We just need to ensure they are float32
    Y_train = pd.DataFrame(Y_train).astype('float32').values
    Y_val = pd.DataFrame(Y_val).astype('float32').values

    # Convert Pandas DataFrames to Tensors
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=tf.int32)  # Use int32 for classification
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    Y_val_tensor = tf.convert_to_tensor(Y_val, dtype=tf.int32)  # Use int32 for classification

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, Y_train_tensor))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_tensor, Y_val_tensor))

    BATCH_SIZE = 32  # Adjust as needed
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_directory_exists(plots_dir)

    # Check if the model already exists
    if os.path.exists(model_path) and (not args.retrain):
        print(f"Trained model already exists at {model_path}. Loading the model...")
        model = load_model(model_path)
    else:
        # Build model
        model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)

        # # Train model
        # history = train_model(
        #     model, X_train, Y_train, X_val, Y_val,
        #     batch_size=args.batch_size,
        #     epochs=args.epochs,
        #     output_dir=args.output_dir
        # )

        # Train model with per mass metrics
        history, per_mass_callback = train_model_with_mass_metrics(
            #model, X_train, Y_train, X_val, Y_val,
            model, train_dataset, val_dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=plots_dir,
            signal_masses=signal_masses
        )

        # Evaluate and save model
        model.save(model_path)
        print(f"Saved model to: {model_path}")
    for mass in signal_masses:
        print(f"Generating validation plots for mass: {mass}")

        # Filter data by mass
        mass_filter = X_val[:, -1] == mass  # The last column is the 'mass'
        X_val_mass = X_val[mass_filter]


        print(f"len(feature_columns) = {len(feature_columns)}")
        # Extract only the features used for training
        X_val_mass_features = X_val_mass[:, :len(feature_columns)]  # Use only the first N columns corresponding to feature_columns
        Y_val_mass = Y_val[mass_filter]

        if len(X_val_mass) == 0:
            print(f"No validation data found for mass: {mass}")
            continue

        # print feature_columns
        print(f"Feature columns: {feature_columns}")

        # 	2.	Validate X_train.shape and X_val_mass_features.shape to ensure they match input_dim=14.
        print("X_train shape:", X_train.shape)
        print("X_val_mass_features shape:", X_val_mass_features.shape)

        # Evaluate model
        y_pred_mass = np.argmax(model.predict(X_val_mass_features), axis=1)  # Exclude 'mass' from prediction input
        y_true_mass = np.argmax(Y_val_mass, axis=1)
        y_score_mass = model.predict(X_val_mass_features)

        # ROC Curve
        plot_roc_curve_multiclass(Y_val_mass, y_score_mass, plots_dir, labels=["ggH", "VBF", "Background"], mass=mass)

        # Confusion Matrix
        plot_confusion_matrix_multiclass(Y_val_mass, y_pred_mass, plots_dir, labels=["ggH", "VBF", "Background"], mass=mass)

        # Classification Report
        plot_classifier_output(model, X_train, Y_train, X_val_mass_features, Y_val_mass, output_dir=plots_dir, mass=mass)

if __name__ == "__main__":
    main()
