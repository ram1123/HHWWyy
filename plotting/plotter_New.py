import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

import shap

# def plot_correlation_matrix(data, output_dir):
#     plt.figure(figsize=(12, 10))
#     corr_matrix = data.corr()
#     sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
#     plt.title("Correlation Matrix")
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "correlation_plot.png"))


def plot_correlation_matrix(data, output_dir, feature_names=None, max_rows=100_000):
    """
    Plots a correlation heatmap.
    - data: np.ndarray [n, d] or pandas.DataFrame with d features
    - output_dir: where to save PNG
    - feature_names: optional list of d names (used if data is ndarray)
    - max_rows: cap rows to speed up corr() on huge datasets
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        d = data.shape[1]
        if feature_names is None or len(feature_names) != d:
            feature_names = [f"f{i}" for i in range(d)]
        df = pd.DataFrame(data, columns=feature_names)
    else:
        df = data.copy()

    # keep only finite values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # row cap to speed up on very large samples
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=7)

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(0.6 * corr.shape[0] + 6, 0.6 * corr.shape[0] + 6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True,
                cbar_kws={'shrink': 0.8})
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    out = os.path.join(output_dir, "correlation_plot.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved correlation matrix to: {out}")


def plot_training_progress(history, output_dir):
    # Plot accuracy progression
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(output_dir, "DNN_acc_wrt_epoch.png"))

    # Plot loss progression
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(output_dir, "history_loss.png"))

import pandas as pd

def plot_training_progress_from_csv(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"[warn] no training log at {csv_path}; skipping training plots")
        return
    df = pd.read_csv(csv_path)
    plt.figure(); plt.plot(df['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in df: plt.plot(df['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy'); plt.ylabel('Accuracy'); plt.xlabel('Epoch')
    plt.legend(loc='upper left'); plt.savefig(os.path.join(output_dir, "DNN_acc_wrt_epoch.png")); plt.close()

    plt.figure(); plt.plot(df['loss'], label='Train Loss')
    if 'val_loss' in df: plt.plot(df['val_loss'], label='Validation Loss')
    plt.title('Model Loss'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend(loc='upper right'); plt.savefig(os.path.join(output_dir, "history_loss.png")); plt.close()


def plot_metrics(history, output_dir, csv_path=None):
    plt.figure(figsize=(12, 8))
    if history is not None:
        for metric in ['precision', 'recall', 'auc']:
            if metric in history.history:
                plt.plot(history.history[metric], label=f'Train {metric}')
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    elif csv_path is not None and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for metric in ['precision', 'recall', 'auc']:
            if metric in df.columns:
                plt.plot(df[metric], label=f'Train {metric}')
            if f'val_{metric}' in df.columns:
                plt.plot(df[f'val_{metric}'], label=f'Validation {metric}')
    else:
        print("No history or valid CSV log provided for plotting metrics.")
        return
    plt.title('Model Metrics')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_metrics.png"))


def plot_confusion_matrix_multiclass(y_true, y_pred, output_dir, labels, mass = None):
    """
    Plot a confusion matrix for multi-class classification or pMulti-class classification.

    :param y_true: Ground truth labels (integers).
    :param y_pred: Predicted labels (integers).
    :param output_dir: Directory to save the plot.
    :param labels: List of class labels.
    :param mass: Mass value for the plot, used in file naming.
    """
    # Convert one-hot to integers if needed
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)), normalize="true")

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    if mass is not None:
        plt.title(f"Confusion Matrix for Mass {mass}")
    else:
        plt.title(f"Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if mass is not None:
        output_path = os.path.join(output_dir, f"confusion_matrix_mass_{mass}.png")
    else:
        output_path = os.path.join(output_dir, "confusion_matrix_multiclass.png")
    plt.savefig(output_path)
    print(f"Saved confusion matrix to: {output_path}")
    plt.close()


def plot_roc_curve_multiclass(y_true, y_score, output_dir, labels, mass=None):
    y_true_b = label_binarize(y_true if y_true.ndim==1 else np.argmax(y_true,1),
                              classes=range(len(labels)))
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)  # <—
    plt.figure(figsize=(8,6))
    for i, label in enumerate(labels):
        if not np.isfinite(y_score[:, i]).all():
            continue
        if y_true_b[:, i].sum() == 0:  # no positives for this class
            continue
        fpr, tpr, _ = roc_curve(y_true_b[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'--',lw=2,color='gray')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(loc='lower right')
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "roc_curve_multiclass.png")); plt.close()


# def plot_roc_curve_multiclass(y_true, y_score, output_dir, labels, mass = None):
#     y_true_binarized = label_binarize(y_true, classes=range(len(labels)))
#     plt.figure(figsize=(8, 6))

#     # For each class, compute the ROC curve and AUC
#     for i, label in enumerate(labels):
#         fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f"Class {label} (AUC = {roc_auc:.2f})")

#     # Plot diagonal line
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--',  lw=2)

#     # Formatting the plot
#     if mass is not None:
#         plt.title(f"ROC Curve for Mass {mass}")
#     else:
#         plt.title("ROC Curve")
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.tight_layout()

#     # Save the plot
#     if mass is not None:
#         plot_path = os.path.join(output_dir, f"roc_curve_mass_{mass}.png")
#     else:
#         plot_path = os.path.join(output_dir, "roc_curve_multiclass.png")
#     plt.savefig(plot_path)
#     plt.close()
#     print(f"ROC curve saved  to: {plot_path}")

def plot_shap_values(model, X_sample, feature_columns, output_dir, max_display=15):
    """
    Plot SHAP values for multi-class model predictions.

    :param model: Trained model.
    :param X_sample: Subset of input features (Numpy array or DataFrame).
    :param feature_columns: List of feature column names.
    :param output_dir: Directory to save plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Verify data and feature alignment
    print("Feature names:", feature_columns)
    print("Number of feature names:", len(feature_columns))
    print("X_sample shape:", X_sample.shape)

    assert X_sample.shape[1] == len(feature_columns), (
        "Mismatch between the number of feature names and the input features!"
    )

    # Initialize the SHAP explainer
    explainer = shap.Explainer(model, X_sample)  # Use a subset for background data
    shap_values = explainer(X_sample)

    # Extract SHAP values for each class
    shap_values_array = shap_values.values  # Shape: (num_samples, num_features, num_classes)

    print("SHAP values shape:", shap_values_array.shape)
    num_classes = shap_values_array.shape[2]
    class_names = [f"Class {i}" for i in range(num_classes)]  # Modify this to reflect your class labels

    # Plot SHAP summary for each class
    for class_idx in range(num_classes):
        plt.figure()
        shap.summary_plot(
            shap_values_array[:, :, class_idx],  # SHAP values for this class
            X_sample,                           # Input features
            feature_names=feature_columns,      # Feature names
            show=False,
            plot_type='bar',
            max_display=max_display   # Only show top 15 features

        )

        # Save the plot
        class_label = class_names[class_idx].replace(" ", "_")  # Replace spaces with underscores
        output_path = f"{output_dir}/shap_summary_plot_class_{class_label}_top{max_display}.pdf"
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"SHAP summary plot saved for {class_label} at {output_path}")

def plot_overfitting(y_train, y_val, y_pred_train, y_pred_val, output_dir):
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_train, label="Train", kde=True, color="blue")
    sns.histplot(y_pred_val, label="Validation", kde=True, color="orange")
    plt.title('Overfitting Diagnostic')
    plt.xlabel('Predicted Class Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overfitting_plot_BinaryClassifier_Binary.png"))


def plot_overfitting_per_class(y_true, y_pred, class_labels, output_dir):
    """
    Plot overfitting for each class in terms of precision, recall, and accuracy.

    :param y_true: Ground truth labels (validation set)
    :param y_pred: Predicted labels
    :param class_labels: List of class names
    :param output_dir: Directory to save plots
    """
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    num_classes = len(class_labels)
    precisions = []
    recalls = []
    accuracies = []

    for i in range(num_classes):
        true_binary = (y_true == i).astype(int)
        pred_binary = (y_pred == i).astype(int)

        precision = precision_score(true_binary, pred_binary, zero_division=0)
        recall = recall_score(true_binary, pred_binary, zero_division=0)
        accuracy = accuracy_score(true_binary, pred_binary)

        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)

    # Plot precision, recall, and accuracy for each class
    metrics = {"Precision": precisions, "Recall": recalls, "Accuracy": accuracies}
    for metric_name, values in metrics.items():
        plt.figure(figsize=(8, 6))
        plt.bar(class_labels, values, color="skyblue")
        plt.ylim(0, 1)
        plt.title(f"{metric_name} by Class")
        plt.xlabel("Class")
        plt.ylabel(metric_name)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric_name}_per_class.png")
        plt.close()

    print(f"Saved per-class overfitting plots to {output_dir}")


def plot_overfitting_multiclass(model, X_train, Y_train, X_test, Y_test, class_labels, output_dir):
    """
    Generate overfitting plots for each class by comparing training and testing output distributions.

    :param model: Trained model.
    :param X_train: Training feature set.
    :param Y_train: Training labels (one-hot encoded or categorical).
    :param X_test: Testing feature set.
    :param Y_test: Testing labels (one-hot encoded or categorical).
    :param class_labels: List of class names.
    :param output_dir: Directory to save plots.
    """
    # Predict probabilities for train and test sets
    train_probs = model.predict(X_train)
    test_probs = model.predict(X_test)

    # Ensure one-hot encoding for labels
    Y_train = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
    Y_test = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test

    for class_idx, class_name in enumerate(class_labels):
        plt.figure(figsize=(8, 6))

        # Separate the predictions for the given class
        train_class_probs = train_probs[:, class_idx]
        test_class_probs = test_probs[:, class_idx]

        # Create histograms for training and testing
        train_hist, train_bins = np.histogram(train_class_probs[Y_train == class_idx], bins=20, density=True)
        test_hist, test_bins = np.histogram(test_class_probs[Y_test == class_idx], bins=20, density=True)

        train_bin_centers = 0.5 * (train_bins[1:] + train_bins[:-1])
        test_bin_centers = 0.5 * (test_bins[1:] + test_bins[:-1])

        # Plot training and testing distributions
        plt.errorbar(train_bin_centers, train_hist, yerr=np.sqrt(train_hist / len(train_class_probs)),
                     fmt='o', label=f"{class_name} training", color='red', alpha=0.6)
        plt.errorbar(test_bin_centers, test_hist, yerr=np.sqrt(test_hist / len(test_class_probs)),
                     fmt='o', label=f"{class_name} testing", color='blue', alpha=0.6)

        plt.title(f"Overfitting Analysis for {class_name}")
        plt.xlabel("Output Score")
        plt.ylabel("(1/N) dN/dX")
        plt.legend()
        plt.grid()

        # Save the plot
        plt.tight_layout()
        plt.savefig(f"{output_dir}/overfitting_{class_name}.png")
        plt.close()

    print(f"Overfitting plots saved to {output_dir}")


def plot_classifier_output(model, X_train, Y_train, X_test, Y_test, output_dir, mass=None):
    """
    Plot the classifier output for ggH vs Background and VBF vs Background for a specific mass.

    :param model: Trained model.
    :param X_train: Training feature set (including the mass feature).
    :param Y_train: Training labels (one-hot encoded or categorical).
    :param X_test: Testing feature set (including the mass feature).
    :param Y_test: Testing labels (one-hot encoded or categorical).
    :param output_dir: Directory to save plots.
    :param mass: Mass value for the plot, used in file naming and filtering.
    """
    # Filter by mass if specified
    if mass is not None:
        train_mass_filter = X_train[:, -1] == mass  # Assume 'mass' is the last column
        test_mass_filter = X_test[:, -1] == mass
        X_train = X_train[train_mass_filter]
        Y_train = Y_train[train_mass_filter]
        X_test = X_test[test_mass_filter]
        Y_test = Y_test[test_mass_filter]

        if len(X_test) == 0 or len(X_train) == 0:
            print(f"No data found for mass: {mass}. Skipping plot generation.")
            return

    # Predict probabilities for train and test sets
    train_probs = model.predict(X_train)
    test_probs = model.predict(X_test)

    # Ensure labels are one-hot encoded
    Y_train = np.argmax(Y_train, axis=1) if Y_train.ndim > 1 else Y_train
    Y_test = np.argmax(Y_test, axis=1) if Y_test.ndim > 1 else Y_test

    # Map class indices to desired comparisons
    comparisons = {
        "ggH vs Background": (1, 2),
        "VBF vs Background": (0, 2),
        "ggH vs VBF": (1, 0),
    }
    # Now, its 5 class: 0: VBF, 1: ggH, 2: DY, 3: EWK, 4: TOP
    #  so, vbf vs background = 0 vs (1+2+3+4)
    # comparisons = {
    #     "VBF vs Background 5Class": (0, 0 + 1 + 2 + 3 + 4),
    #     "ggH vs Background 5Class": (1, 1 + 2 + 3 + 4),
    # }

    for plot_title, (signal_idx, background_idx) in comparisons.items():
        plt.figure(figsize=(8, 6))

        # Extract classifier scores for the specific comparison
        train_signal_scores = train_probs[Y_train == signal_idx, signal_idx]
        train_background_scores = train_probs[Y_train == background_idx, signal_idx]
        test_signal_scores = test_probs[Y_test == signal_idx, signal_idx]
        test_background_scores = test_probs[Y_test == background_idx, signal_idx]

        # Plot histograms for signal and background
        bins = np.linspace(0, 1, 21)
        plt.hist(train_signal_scores, bins=bins, density=True, alpha=0.6, label="Signal (Train)", color='red')
        plt.hist(train_background_scores, bins=bins, density=True, alpha=0.6, label="Background (Train)", color='blue')
        plt.hist(test_signal_scores, bins=bins, density=True, histtype='step', linewidth=1.5, label="Signal (Test)", color='red', linestyle='dashed')
        plt.hist(test_background_scores, bins=bins, density=True, histtype='step', linewidth=1.5, label="Background (Test)", color='blue', linestyle='dashed')

        # Calculate separations
        train_sep = compute_separation(train_signal_scores, train_background_scores)
        test_sep = compute_separation(test_signal_scores, test_background_scores)
        plt.text(0.5, 0.8, f"Train Sep.: {train_sep:.4f}\nTest Sep.: {test_sep:.4f}", transform=plt.gca().transAxes)

        plt.xlabel("Classifier Output")
        plt.ylabel("(1/N) dN/dX")
        mass_label = f" (Mass: {mass})" if mass is not None else ""
        plt.title(f"Classifier Output: {plot_title}{mass_label}")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{output_dir}/classifier_output_{plot_title.replace(' ', '_').lower()}"
        if mass is not None:
            plot_filename += f"_mass_{mass}"
        plot_filename += ".png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"Saved {plot_title} plot to {plot_filename}")

def compute_separation(signal_scores, background_scores):
    """
    Compute the separation between signal and background scores.

    :param signal_scores: Array of signal classifier scores.
    :param background_scores: Array of background classifier scores.
    :return: Separation value.
    """
    hist_signal, bins = np.histogram(signal_scores, bins=50, range=(0, 1), density=True)
    hist_background, _ = np.histogram(background_scores, bins=50, range=(0, 1), density=True)
    bin_width = bins[1] - bins[0]
    sep = 0.5 * np.sum(((hist_signal - hist_background) ** 2) / (hist_signal + hist_background + 1e-10)) * bin_width
    return sep
