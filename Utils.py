import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
import os


def PrintScore(true_labels, pred_labels, fold=None, savePath='./output/', model_name="model"):
    """
    Calculate and print classification metrics
    """
    if fold is not None:
        print(f"\n=== Fold {fold} Results ===")
    else:
        print(f"\n=== Overall Results ===")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision_macro = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_labels, pred_labels)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # Save results to file
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    results_file = os.path.join(savePath, f"{model_name}_results{fold_str}.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"=== {model_name} Results ===\n")
        if fold is not None:
            f.write(f"Fold: {fold}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (macro): {precision_macro:.4f}\n")
        f.write(f"Recall (macro): {recall_macro:.4f}\n")
        f.write(f"F1-score (macro): {f1_macro:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(true_labels, pred_labels, 
                                    target_names=['W', 'N1', 'N2', 'N3', 'REM']))
    
    return accuracy, precision_macro, recall_macro, f1_macro, kappa


def ConfusionMatrix(true_labels, pred_labels, fold=None, classes=['W', 'N1', 'N2', 'N3', 'REM'], 
                   savePath='./output/', model_name="model"):
    """
    Generate and save confusion matrix
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} Confusion Matrix' + (f' - Fold {fold}' if fold is not None else ''))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    plt.savefig(os.path.join(savePath, f"{model_name}_confusion_matrix{fold_str}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate per-class metrics
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\nPer-class Accuracy:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {per_class_acc[i]:.4f}")
    
    return cm


def plot_training_history(history, fold=None, savePath='./output/', model_name="model"):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    plt.savefig(os.path.join(savePath, f"{model_name}_training_history{fold_str}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions(true_labels, pred_labels, pred_probs=None, fold=None, 
                    savePath='./output/', model_name="model"):
    """
    Save predictions to file
    """
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    fold_str = f"_fold_{fold}" if fold is not None else ""
    pred_file = os.path.join(savePath, f"{model_name}_predictions{fold_str}.npz")
    
    if pred_probs is not None:
        np.savez(pred_file, 
                true_labels=true_labels, 
                pred_labels=pred_labels, 
                pred_probs=pred_probs)
    else:
        np.savez(pred_file, 
                true_labels=true_labels, 
                pred_labels=pred_labels)


def calculate_overall_metrics(all_true, all_pred, classes=['W', 'N1', 'N2', 'N3', 'REM']):
    """
    Calculate overall metrics across all folds
    """
    print("\n" + "="*50)
    print("OVERALL CROSS-VALIDATION RESULTS")
    print("="*50)
    
    # Overall metrics
    overall_acc = accuracy_score(all_true, all_pred)
    overall_precision = precision_score(all_true, all_pred, average='macro', zero_division=0)
    overall_recall = recall_score(all_true, all_pred, average='macro', zero_division=0)
    overall_f1 = f1_score(all_true, all_pred, average='macro', zero_division=0)
    overall_kappa = cohen_kappa_score(all_true, all_pred)
    
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Overall Precision (macro): {overall_precision:.4f}")
    print(f"Overall Recall (macro): {overall_recall:.4f}")
    print(f"Overall F1-score (macro): {overall_f1:.4f}")
    print(f"Overall Cohen's Kappa: {overall_kappa:.4f}")
    
    # Per-class metrics
    per_class_precision = precision_score(all_true, all_pred, average=None, zero_division=0)
    per_class_recall = recall_score(all_true, all_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(all_true, all_pred, average=None, zero_division=0)
    
    print(f"\nPer-class Results:")
    print(f"{'Class':<5} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 40)
    for i, class_name in enumerate(classes):
        print(f"{class_name:<5} {per_class_precision[i]:<10.4f} {per_class_recall[i]:<10.4f} {per_class_f1[i]:<10.4f}")
    
    return overall_acc, overall_precision, overall_recall, overall_f1, overall_kappa
