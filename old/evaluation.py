"""
evaluation.py — Phase 1 Model Evaluation
=========================================
Evaluates the deployed U-Net anomaly detection model on the full test set.

Run from the project root:
    python notebooks/evaluation.py

Fixes from the original notebook:
  1. Correct model: unet_modified_model.h5 (not attention_unet)
  2. Correct scoring: top-5 pixels (matches main.py exactly)
  3. No data leakage: threshold computed on normal images only
  4. AUROC added: threshold-independent discriminative measure
  5. Precision-recall curve: finds optimal threshold automatically
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)

# ─────────────────────────────────────────────────────────────
# CONFIG — paths relative to project root
# ─────────────────────────────────────────────────────────────
DATASET_PATH = "ITD"
MODEL_PATH   = "model/unet_modified_model.h5"
OUTPUT_DIR   = "notebooks/eval_output"
IMG_SIZE     = (128, 128)
CROP         = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Loss functions (required to load model)
# ─────────────────────────────────────────────────────────────
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + ssim_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(
        vgg(preprocess_input(y_true * 255.0)) -
        vgg(preprocess_input(y_pred * 255.0))
    ))

def final_loss(y_true, y_pred):
    return combined_loss(y_true, y_pred) + 0.1 * perceptual_loss(y_true, y_pred)

# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
print("Loading model...")
model = load_model(
    MODEL_PATH,
    custom_objects={
        "final_loss":    final_loss,
        "ssim_loss":     ssim_loss,
        "combined_loss": combined_loss,
    }
)
print("Model loaded.\n")

# ─────────────────────────────────────────────────────────────
# Load test data
# Walks ITD/<typeXcamY>/test/good and ITD/<typeXcamY>/test/anomaly
# ─────────────────────────────────────────────────────────────
def load_test_images(base_dir, img_size=(128, 128)):
    images, labels, paths = [], [], []

    for folder in sorted(os.listdir(base_dir)):
        type_path = os.path.join(base_dir, folder)
        if not os.path.isdir(type_path):
            continue
        test_path = os.path.join(type_path, "test")
        if not os.path.exists(test_path):
            continue

        for label_name in ["good", "anomaly"]:
            class_path = os.path.join(test_path, label_name)
            if not os.path.exists(class_path):
                continue

            for img_file in sorted(os.listdir(class_path)):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size).astype(np.float32) / 255.0
                images.append(img)
                labels.append(0 if label_name == "good" else 1)
                paths.append(img_path)

    return np.array(images), np.array(labels), paths


print("Loading test data...")
X_test, y_true, img_paths = load_test_images(DATASET_PATH, IMG_SIZE)
n_normal  = (y_true == 0).sum()
n_anomaly = (y_true == 1).sum()
print(f"Total images : {len(X_test)}")
print(f"Normal       : {n_normal}")
print(f"Anomaly      : {n_anomaly}\n")

# ─────────────────────────────────────────────────────────────
# Compute anomaly scores for all images
# Uses EXACTLY the same scoring function as main.py:
#   score = mean of top-5 pixels in cropped error map
# ─────────────────────────────────────────────────────────────
print("Running inference (this will take a few minutes)...")
recons = model.predict(X_test, batch_size=16, verbose=1)

all_scores = []
for i in range(len(X_test)):
    img   = X_test[i]
    recon = recons[i]

    error_map   = np.mean((img - recon) ** 2, axis=2)
    error_map   = cv2.GaussianBlur(error_map, (5, 5), 0)
    h, w        = error_map.shape
    cropped     = error_map[CROP:h-CROP, CROP:w-CROP]
    score       = float(np.mean(np.sort(cropped.flatten())[-5:]))
    all_scores.append(score)

all_scores = np.array(all_scores)
print("Inference complete.\n")

# ─────────────────────────────────────────────────────────────
# AUROC — threshold-independent
# Most honest single measure of model discriminative power
# ─────────────────────────────────────────────────────────────
auroc = roc_auc_score(y_true, all_scores)
print(f"AUROC: {auroc:.4f}")

# ─────────────────────────────────────────────────────────────
# Threshold 1: deployed threshold from main.py
# ─────────────────────────────────────────────────────────────
DEPLOYED_THRESHOLD = 0.0015206437
y_pred_deployed    = (all_scores > DEPLOYED_THRESHOLD).astype(int)

print("\n===== DEPLOYED THRESHOLD ({:.6e}) =====".format(DEPLOYED_THRESHOLD))
print(f"Accuracy : {accuracy_score(y_true, y_pred_deployed):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_deployed):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_deployed):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred_deployed):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_deployed))

# ─────────────────────────────────────────────────────────────
# Threshold 2: computed from normal images only (no leakage)
# mean + 2*std of scores on normal images only
# ─────────────────────────────────────────────────────────────
normal_scores     = all_scores[y_true == 0]
computed_threshold = float(np.mean(normal_scores) + 2 * np.std(normal_scores))
y_pred_computed   = (all_scores > computed_threshold).astype(int)

print("\n===== COMPUTED THRESHOLD (mean+2std on normal) ({:.6e}) =====".format(
    computed_threshold))
print(f"Accuracy : {accuracy_score(y_true, y_pred_computed):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_computed):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_computed):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred_computed):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_computed))

# ─────────────────────────────────────────────────────────────
# Threshold 3: F1-optimal threshold
# Finds the threshold that maximises F1 across the test set
# ─────────────────────────────────────────────────────────────
precisions, recalls, thresholds_pr = precision_recall_curve(y_true, all_scores)
f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_idx   = np.argmax(f1_scores)
best_threshold = float(thresholds_pr[best_idx])
y_pred_optimal = (all_scores > best_threshold).astype(int)

print("\n===== F1-OPTIMAL THRESHOLD ({:.6e}) =====".format(best_threshold))
print(f"Accuracy : {accuracy_score(y_true, y_pred_optimal):.4f}")
print(f"Precision: {precision_score(y_true, y_pred_optimal):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred_optimal):.4f}")
print(f"F1 Score : {f1_score(y_true, y_pred_optimal):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_optimal))

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_true, all_scores)
axes[0].plot(fpr, tpr, color='steelblue', lw=2,
             label=f"AUROC = {auroc:.4f}")
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. Precision-Recall Curve
axes[1].plot(recalls, precisions, color='darkorange', lw=2)
axes[1].axvline(recalls[best_idx], color='red', linestyle='--', lw=1,
                label=f"Best F1={f1_scores[best_idx]:.3f} @ thr={best_threshold:.4e}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# 3. Score distribution
normal_sc  = all_scores[y_true == 0]
anomaly_sc = all_scores[y_true == 1]
axes[2].hist(normal_sc,  bins=60, alpha=0.6, color='green',
             label=f"Normal (n={len(normal_sc)})")
axes[2].hist(anomaly_sc, bins=60, alpha=0.6, color='red',
             label=f"Anomaly (n={len(anomaly_sc)})")
axes[2].axvline(DEPLOYED_THRESHOLD, color='blue', linestyle='--', lw=1.5,
                label=f"Deployed ({DEPLOYED_THRESHOLD:.4e})")
axes[2].axvline(best_threshold, color='black', linestyle='--', lw=1.5,
                label=f"F1-Optimal ({best_threshold:.4e})")
axes[2].set_xlabel("Anomaly Score")
axes[2].set_ylabel("Count")
axes[2].set_title("Score Distribution")
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3)

plt.suptitle("Phase 1 Evaluation — Textile Anomaly Detection", fontsize=13)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "evaluation.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlots saved → {plot_path}")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("EVALUATION SUMMARY")
print("="*55)
print(f"Total test images     : {len(X_test)}")
print(f"Normal / Anomaly      : {n_normal} / {n_anomaly}")
print(f"AUROC                 : {auroc:.4f}")
print(f"Deployed threshold    : {DEPLOYED_THRESHOLD:.6e}")
print(f"  → F1                : {f1_score(y_true, y_pred_deployed):.4f}")
print(f"  → Recall            : {recall_score(y_true, y_pred_deployed):.4f}")
print(f"  → Precision         : {precision_score(y_true, y_pred_deployed):.4f}")
print(f"F1-optimal threshold  : {best_threshold:.6e}")
print(f"  → F1                : {f1_score(y_true, y_pred_optimal):.4f}")
print(f"  → Recall            : {recall_score(y_true, y_pred_optimal):.4f}")
print(f"  → Precision         : {precision_score(y_true, y_pred_optimal):.4f}")
print("="*55)

if auroc >= 0.90:
    print("\nAUROC ≥ 0.90 — strong discriminative power.")
elif auroc >= 0.80:
    print("\nAUROC 0.80–0.90 — good discriminative power.")
elif auroc >= 0.70:
    print("\nAUROC 0.70–0.80 — moderate discriminative power.")
else:
    print("\nAUROC < 0.70 — weak discriminative power.")

f1_gap = f1_score(y_true, y_pred_optimal) - f1_score(y_true, y_pred_deployed)
if f1_gap > 0.1:
    print(f"F1 gap of {f1_gap:.3f} between deployed and optimal threshold.")
    print(f"Consider updating THRESHOLD in main.py to {best_threshold:.10f}")