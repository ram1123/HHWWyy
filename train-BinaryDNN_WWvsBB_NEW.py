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

# Set TensorFlow and Matplotlib configurations
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
matplotlib.use('Agg')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Seed for reproducibility
np.random.seed(7)

CURRENT_DATETIME = datetime.now()

def GenerateGitPatchAndLog(logFileName,GitPatchName):
    #CMSSWDirPath = os.environ['CMSSW_BASE']
    #CMSSWRel = CMSSWDirPath.split("/")[-1]

    os.system('git diff > '+GitPatchName)

    outScript = open(logFileName,"w");
    #outScript.write('\nCMSSW Version used: '+CMSSWRel+'\n')
    #outScript.write('\nCurrent directory path: '+CMSSWDirPath+'\n')
    outScript.close()

    os.system('echo -e "\n\n============\n== Latest commit summary \n\n" >> '+logFileName )
    os.system("git log -1 --pretty=tformat:' Commit: %h %n Date: %ad %n Relative time: %ar %n Commit Message: %s' >> "+logFileName )
    os.system('echo -e "\n\n============\n" >> '+logFileName )
    os.system('git log -1 --format="SHA: %H" >> '+logFileName )

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    #out = commands.getoutput(eos_cmd)
    return

# Ensure directory exists
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load data from ROOT files into a DataFrame
def load_data(inputPath, variables, num_events, csv_path, metadata_path):
    """
    Save ROOT data to a CSV file if not already saved or if the variable list is updated.
    If conditions are met, load the data directly from the CSV file.

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
    data = pd.DataFrame(columns=variables)
    for key in keys:
        if key == 'ggh':
            fileNames = ["GluGluHToZZTo2L2Nu_M1000_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8"]
            target = 0  # ggH
        elif key == 'vbf':
            fileNames = ["VBF_HToZZTo2L2Nu_M1500_TuneCP5_13TeV_powheg2_JHUGenV7011_pythia8"]
            target = 1  # VBF
        else:
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

    # Save DataFrame to CSV
    print(f"Saving DataFrame to CSV: {csv_path}")
    data.to_csv(csv_path, index=False)

    # Save variable list to metadata
    metadata = {"variables": variables}
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    return data


# Ensure input data is numeric and clean
def preprocess_data(data):
    # Replace NaN or infinite values with a default (e.g., 0 or mean)
    data = data.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
    data = data.fillna(0)  # Replace NaN with 0 (or use column mean if needed)
    return data.astype('float32')  # Ensure all values are float32


# Metrics for evaluation
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# Custom learning rate scheduler
def custom_learning_rate_scheduler(epoch, lr):
    if epoch < 10:
        return 0.001
    else:
        return float(lr * tf.math.exp(-0.005 * (epoch - 10)))

# Build and compile a multi-class DNN model
def build_model(input_dim, activation='relu', dropout_rate=0.2, learn_rate=0.001):
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
    early_stopping = EarlyStopping(patience=21, monitor='val_loss', restore_best_weights=True)
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
    parser.add_argument('--job_name', type=str, default="DNN", help="Job name.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--learn_rate', type=float, default=0.0001, help="Learning rate.")
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
    data = load_data(args.inputPath, variables, args.num_events, csv_path=csv_path, metadata_path=metadata_path)

    X = data[variables].values
    Y = pd.get_dummies(data['target']).values  # One-hot encoding for multi-class

    # Split into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=7)

    # Preprocess training and validation data
    X_train = preprocess_data(pd.DataFrame(X_train)).values
    X_val = preprocess_data(pd.DataFrame(X_val)).values
    Y_train = preprocess_data(pd.DataFrame(Y_train)).values
    Y_val = preprocess_data(pd.DataFrame(Y_val)).values

    # Create plots directory
    plots_dir = os.path.join(args.output_dir, "plots")
    ensure_directory_exists(plots_dir)

    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Trained model already exists at {model_path}. Loading the model...")
        model = load_model(model_path)
    else:
        # Build model
        model = build_model(input_dim=X_train.shape[1], learn_rate=args.learn_rate)

        # Train model
        history = train_model(
            model, X_train, Y_train, X_val, Y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir
        )

        # Evaluate and save model
        model.save(model_path)
        print(f"Saved model to: {model_path}")

        # Training progress
        plot_training_progress(history, plots_dir)

        # Metrics
        plot_metrics(history, plots_dir)

    # Predictions for confusion matrix
    y_pred = np.argmax(model.predict(X_val), axis=1)
    y_true = np.argmax(Y_val, axis=1)

    labels = ["ggH", "VBF", "Background"]
    plot_confusion_matrix(y_true, y_pred, os.path.join(args.output_dir, "confusion_matrix.png"), labels)

    # Classification report
    print(classification_report(y_true, y_pred, target_names=labels))

    # Define feature columns by excluding non-feature variables
    feature_columns = [col for col in variables if col not in ['target', 'process_ID', 'classweight']]

    # Correlation matrix
    plot_correlation_matrix(pd.DataFrame(X_train, columns=feature_columns), plots_dir)



    # Confusion Matrix
    y_pred_val = np.argmax(model.predict(X_val), axis=1)
    plot_confusion_matrix_multiclass(Y_val, y_pred_val, plots_dir, labels=["ggH", "VBF", "Background"])

    # ROC Curve
    y_score_val = model.predict(X_val)
    plot_roc_curve_multiclass(Y_val, y_score_val, plots_dir, labels=["ggH", "VBF", "Background"])

    # SHAP values
    plot_shap_values(model, X_train[:10], feature_columns, plots_dir)

    # Overfitting diagnostic
    y_pred_train = np.argmax(model.predict(X_train), axis=1)
    plot_overfitting(Y_train, Y_val, y_pred_train, y_pred_val, plots_dir)

    # Plot overfitting per class
    plot_overfitting_per_class(y_true, y_pred, ["ggH", "VBF", "Background"], plots_dir)

    # Overfitting plots for multi-class classification
    class_labels = ["ggH", "VBF", "Background"]
    plot_overfitting_multiclass(model, X_train, Y_train, X_val, Y_val, class_labels=class_labels, output_dir=plots_dir)

    # Classifier output
    plot_classifier_output(model=model, X_train=X_train, Y_train=Y_train, X_test=X_val, Y_test=Y_val, output_dir=plots_dir)

    # Plot DNN network architecture
    tf.keras.utils.plot_model(model, to_file=os.path.join(plots_dir, "model.png"), show_shapes=True)

if __name__ == "__main__":
    main()
