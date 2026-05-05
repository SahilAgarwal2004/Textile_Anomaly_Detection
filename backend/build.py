"""
PHASE 2 — BUILD SCRIPT
======================
Run this ONCE across all fabric type folders.
It automatically finds every ITD/typeXcamY/test/anomaly/ folder.

Changes from previous version:
  - extract_patch now also returns geometric features from the mask
    (area, aspect_ratio, extent, solidity, equiv_diameter)
  - features are concatenated to embedding before clustering
    so cluster separation uses both visual AND shape information
  - geometric features saved in clusters.pkl for use at inference

Usage:
    python build.py

Edit the CONFIG block below if needed.
"""

import os
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH    = "model/unet_modified_model.h5"
ITD_ROOT      = "ITD"
OUTPUT_DIR    = "phase2_output"
CLUSTERS_PATH = "clusters.pkl"

IMG_HEIGHT      = 128
IMG_WIDTH       = 128
PATCH_SIZE      = 64
PADDING         = 20
MIN_DEFECT_AREA = 30
N_CLUSTERS      = 6
PIXEL_THRESHOLD = 0.2

# Weight of geometric features relative to embedding.
# 0.3 means geometry contributes 30% of clustering signal.
GEO_WEIGHT = 0.3
MIN_PATCHES_FOR_FABRIC_MODEL = 15

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Discover all anomaly folders automatically
# ─────────────────────────────────────────────────────────────
def find_anomaly_folders(itd_root):
    found = []
    if not os.path.isdir(itd_root):
        raise FileNotFoundError(f"ITD root not found: {itd_root}")
    for entry in sorted(os.listdir(itd_root)):
        candidate = os.path.join(itd_root, entry, "test", "anomaly")
        if os.path.isdir(candidate):
            images = [f for f in os.listdir(candidate)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if images:
                found.append((entry, candidate))
    return found


anomaly_folders = find_anomaly_folders(ITD_ROOT)
if not anomaly_folders:
    raise RuntimeError(f"No anomaly folders found under {ITD_ROOT}.")

print(f"Found {len(anomaly_folders)} anomaly folder(s):")
for name, path in anomaly_folders:
    n_imgs = len([f for f in os.listdir(path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f"  {name:15s}  →  {path}  ({n_imgs} images)")
print()

# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────
vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse + ssim_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
    y_true_vgg = vgg(preprocess_input(y_true * 255.0))
    y_pred_vgg = vgg(preprocess_input(y_pred * 255.0))
    return tf.reduce_mean(tf.square(y_true_vgg - y_pred_vgg))

def final_loss(y_true, y_pred):
    return combined_loss(y_true, y_pred) + 0.1 * perceptual_loss(y_true, y_pred)

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
print("Loading Phase 1 model...")
model = load_model(
    MODEL_PATH,
    custom_objects={
        "final_loss":    final_loss,
        "ssim_loss":     ssim_loss,
        "combined_loss": combined_loss,
    }
)
print("Phase 1 model loaded.\n")

print("Building embedder (EfficientNetB0)...")
base = EfficientNetB0(include_top=False, weights='imagenet',
                      input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
                      pooling='avg')
base.trainable = False
print("Embedder ready.\n")

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────
crop = 10

def phase1_outputs(img_bgr):
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
    img_norm    = img_resized / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)
    recon       = model.predict(img_input, verbose=0)[0]
    error_map   = np.mean((img_norm - recon) ** 2, axis=2)
    error_map   = cv2.GaussianBlur(error_map, (5, 5), 0)
    h, w        = error_map.shape
    cropped_err = error_map[crop:h-crop, crop:w-crop]
    norm_error  = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
    mask        = (norm_error > PIXEL_THRESHOLD).astype(np.uint8) * 255
    kernel      = np.ones((3, 3), np.uint8)
    mask        = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask        = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    flat        = cropped_err.flatten()
    score       = np.mean(np.sort(flat)[-300:])
    return norm_error, mask, img_norm, score


def compute_geometric_features(mask, stats, largest):
    """
    Computes 5 geometric features from the largest connected component.

    Returns a dict AND a numpy array of shape (5,):
      - area            : pixel count of defect region (normalized by image area)
      - aspect_ratio    : width / height of bounding box (>1 = wide, <1 = tall)
      - extent          : area / bounding_box_area  (how much bbox is filled)
      - solidity        : area / convex_hull_area   (how convex the shape is)
      - equiv_diameter  : diameter of circle with same area (normalized by image diagonal)

    Why these five:
      - area distinguishes small spots from large structural damage
      - aspect_ratio separates linear thread defects (high ratio) from spots (ratio ~1)
      - extent separates compact solid defects from irregular/fragmented ones
      - solidity separates smooth defects from jagged/frayed edges
      - equiv_diameter is a size measure independent of shape
    """
    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    w = stats[largest, cv2.CC_STAT_WIDTH]
    h = stats[largest, cv2.CC_STAT_HEIGHT]
    area = stats[largest, cv2.CC_STAT_AREA]

    image_area     = IMG_HEIGHT * IMG_WIDTH
    image_diagonal = np.sqrt(IMG_HEIGHT**2 + IMG_WIDTH**2)

    # Aspect ratio: clamp to avoid division by zero
    aspect_ratio = w / max(h, 1)

    # Extent: how much of the bounding box is filled
    bbox_area = max(w * h, 1)
    extent    = area / bbox_area

    # Solidity: area / convex hull area
    component_mask = (mask[y:y+h, x:x+w] > 0).astype(np.uint8)
    contours, _    = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull      = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        solidity  = min(1.0, area / max(hull_area, 1))
    else:
        solidity = 1.0

    # Equivalent diameter (normalized)
    equiv_diameter = np.sqrt(4 * area / np.pi) / image_diagonal

    geo_dict = {
        "area":           area / image_area,
        "aspect_ratio":   aspect_ratio,
        "extent":         float(extent),
        "solidity":       float(solidity),
        "equiv_diameter": float(equiv_diameter),
    }
    geo_array = np.array([
        geo_dict["area"],
        geo_dict["aspect_ratio"],
        geo_dict["extent"],
        geo_dict["solidity"],
        geo_dict["equiv_diameter"],
    ], dtype=np.float32)

    return geo_dict, geo_array


def extract_patch(img_norm, norm_error, mask):
    """
    Returns: patch_emb, patch_viz, geo_dict, geo_array
    All four are None if no valid region found.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, None, None, None
    valid = [i for i in range(1, num_labels)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_DEFECT_AREA]
    if not valid:
        return None, None, None, None

    largest = max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])

    # Geometric features from mask
    geo_dict, geo_array = compute_geometric_features(mask, stats, largest)

    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    w = stats[largest, cv2.CC_STAT_WIDTH]
    h = stats[largest, cv2.CC_STAT_HEIGHT]

    x1 = max(0,          x - PADDING)
    y1 = max(0,          y - PADDING)
    x2 = min(IMG_WIDTH,  x + w + PADDING)
    y2 = min(IMG_HEIGHT, y + h + PADDING)

    patch_orig     = img_norm[y1:y2, x1:x2]
    weight         = norm_error[y1:y2, x1:x2, np.newaxis]
    weight         = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
    patch_weighted = patch_orig * (0.5 + 0.5 * weight)
    patch_uint8    = (patch_weighted * 255).clip(0, 255).astype(np.uint8)
    patch_emb      = cv2.resize(patch_uint8, (PATCH_SIZE, PATCH_SIZE))
    patch_viz_raw  = (patch_orig * 255).clip(0, 255).astype(np.uint8)
    patch_viz      = cv2.resize(patch_viz_raw, (PATCH_SIZE, PATCH_SIZE))

    return patch_emb, patch_viz, geo_dict, geo_array


def embed_patch(patch_uint8):
    x = patch_uint8.astype(np.float32)
    x = eff_preprocess(x)
    x = np.expand_dims(x, axis=0)
    return base.predict(x, verbose=0)[0]   # (1280,)

# ─────────────────────────────────────────────────────────────
# STEP 1: Process ALL anomaly folders
# ─────────────────────────────────────────────────────────────
all_embeddings   = []
all_geo_arrays   = []    # (N, 5) geometric features
all_geo_dicts    = []    # list of dicts, saved for reference
all_patches_viz  = []
all_filenames    = []
all_fabric_types = []
skipped_total    = 0

for fabric_type, anomaly_dir in anomaly_folders:
    image_files = sorted([
        f for f in os.listdir(anomaly_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    print(f"Processing {fabric_type} ({len(image_files)} images)...")
    folder_collected = 0
    folder_skipped   = 0

    for fname in tqdm(image_files, desc=f"  {fabric_type}", leave=False):
        fpath   = os.path.join(anomaly_dir, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            folder_skipped += 1
            continue

        norm_error, mask, img_norm, score = phase1_outputs(img_bgr)

        patch_emb, patch_viz, geo_dict, geo_array = extract_patch(
            img_norm, norm_error, mask
        )
        if patch_emb is None:
            folder_skipped += 1
            continue

        embedding = embed_patch(patch_emb)
        all_embeddings.append(embedding)
        all_geo_arrays.append(geo_array)
        all_geo_dicts.append(geo_dict)
        all_patches_viz.append(patch_viz)
        all_filenames.append(f"{fabric_type}/{fname}")
        all_fabric_types.append(fabric_type)
        folder_collected += 1

    skipped_total += folder_skipped
    print(f"  → collected: {folder_collected}  |  skipped: {folder_skipped}")

print(f"\nTotal patches collected : {len(all_embeddings)}")
print(f"Total skipped           : {skipped_total}\n")

if len(all_embeddings) < N_CLUSTERS:
    raise ValueError(
        f"Only {len(all_embeddings)} valid patches but N_CLUSTERS={N_CLUSTERS}."
    )

# ─────────────────────────────────────────────────────────────
# STEP 2: Build combined feature vector
#
# Strategy:
#   1. L2-normalize embeddings  → unit sphere, shape (N, 1280)
#   2. StandardScale geo features → zero mean, unit variance, shape (N, 5)
#   3. Scale geo by GEO_WEIGHT  → controls how much shape influences clustering
#   4. Concatenate              → shape (N, 1285)
#   5. L2-normalize again       → KMeans works on cosine-like distances
#
# This means clustering uses both visual texture AND defect geometry.
# ─────────────────────────────────────────────────────────────
print("Building combined feature vectors...")
embeddings_array = np.array(all_embeddings)       # (N, 1280)
geo_array_all    = np.array(all_geo_arrays)        # (N, 5)

# Normalize embedding
embeddings_norm  = normalize(embeddings_array)     # (N, 1280)

# Scale geometric features to zero mean / unit variance
geo_scaler       = StandardScaler()
geo_scaled       = geo_scaler.fit_transform(geo_array_all)   # (N, 5)
geo_weighted     = geo_scaled * GEO_WEIGHT                    # (N, 5)

# Concatenate and re-normalize
combined         = np.concatenate([embeddings_norm, geo_weighted], axis=1)  # (N, 1285)
combined_norm    = normalize(combined)             # (N, 1285)

# ─────────────────────────────────────────────────────────────
# STEP 3: Cluster on combined features
# ─────────────────────────────────────────────────────────────
print(f"Clustering {len(all_embeddings)} patches into {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(combined_norm)
print("Clustering done.\n")
# ─────────────────────────────────────────────────────────────
# Calibrate confidence threshold
# For every training patch, compute its distance to its assigned
# centroid. Set threshold at 90th percentile of those distances.
# This replaces the arbitrary DISTANCE_THRESHOLD = 4.0 in infer.py
# with a value grounded in the actual data distribution.
# ─────────────────────────────────────────────────────────────
distances_to_assigned = np.array([
    kmeans.transform(combined_norm[i:i+1])[0][cluster_labels[i]]
    for i in range(len(combined_norm))
])
calibrated_threshold = float(np.percentile(distances_to_assigned, 90))
print(f"Calibrated confidence threshold (90th percentile): {calibrated_threshold:.4f}\n")

# ─────────────────────────────────────────────────────────────
# STEP 3b: Per-fabric cluster models
# For each fabric type with >= MIN_PATCHES_FOR_FABRIC_MODEL patches,
# fit a separate KMeans on only that fabric's patches.
# Fabric types below the threshold fall back to the global model.
# ─────────────────────────────────────────────────────────────
print("Building per-fabric cluster models...")
fabric_models = {}   # {fabric_type: {"kmeans": ..., "combined_norm": ...,
                     #                "cluster_label_map": ...,
                     #                "confidence_threshold": ...}}

for fabric_type, _ in anomaly_folders:
    # Get indices for this fabric type
    indices = [i for i, ft in enumerate(all_fabric_types) if ft == fabric_type]
    n       = len(indices)

    if n < MIN_PATCHES_FOR_FABRIC_MODEL:
        print(f"  {fabric_type:15s}: {n:3d} patches → using global model (below threshold)")
        continue

    # How many clusters for this fabric? Scale with patch count but cap at N_CLUSTERS.
    n_clusters_fabric = min(N_CLUSTERS, max(2, n // 5))

    fabric_combined = combined_norm[indices]   # subset of combined features

    km_fabric = KMeans(n_clusters=n_clusters_fabric, random_state=42, n_init=10)
    fabric_cluster_labels = km_fabric.fit_predict(fabric_combined)

    # Calibrate per-fabric confidence threshold
    fabric_distances = np.array([
        km_fabric.transform(fabric_combined[i:i+1])[0][fabric_cluster_labels[i]]
        for i in range(len(fabric_combined))
    ])
    fabric_threshold = float(np.percentile(fabric_distances, 90))

    fabric_models[fabric_type] = {
        "kmeans":               km_fabric,
        "combined_norm":        fabric_combined,
        "indices":              indices,          # indices into global arrays
        "n_clusters":           n_clusters_fabric,
        "cluster_label_map":    {i: f"Defect Type {i}"
                                 for i in range(n_clusters_fabric)},
        "confidence_threshold": fabric_threshold,
    }
    print(f"  {fabric_type:15s}: {n:3d} patches → "
          f"{n_clusters_fabric} clusters, threshold={fabric_threshold:.4f}")

print(f"\nPer-fabric models built: {len(fabric_models)} / {len(anomaly_folders)}")
fabric_fallback = [ft for ft, _ in anomaly_folders if ft not in fabric_models]
if fabric_fallback:
    print(f"Using global model for: {fabric_fallback}\n")


# ─────────────────────────────────────────────────────────────
# STEP 4: Cluster grid
# ─────────────────────────────────────────────────────────────
SAMPLES_PER_CLUSTER = 12
fabric_type_list    = sorted(set(all_fabric_types))

print("Generating cluster visualization grid...")
fig = plt.figure(figsize=(SAMPLES_PER_CLUSTER * 1.4, N_CLUSTERS * 2.4))
fig.suptitle(
    "CLUSTER GRID  —  Each row is one cluster\n"
    "Inspect each row and assign defect names using edit_labels.py",
    fontsize=11, y=1.01
)

for c in range(N_CLUSTERS):
    indices = np.where(cluster_labels == c)[0]
    count   = len(indices)

    types_in_cluster = [all_fabric_types[i] for i in indices]
    type_counts      = {t: types_in_cluster.count(t)
                        for t in fabric_type_list if t in types_in_cluster}
    type_str         = ", ".join(f"{t}:{n}" for t, n in sorted(type_counts.items()))

    # Mean geometric features for this cluster (for console summary)
    cluster_geo = geo_array_all[indices]
    mean_area   = cluster_geo[:, 0].mean()
    mean_ar     = cluster_geo[:, 1].mean()
    mean_sol    = cluster_geo[:, 3].mean()

    sample_idx = np.random.choice(
        indices, size=min(SAMPLES_PER_CLUSTER, count), replace=False
    )

    for col, idx in enumerate(sample_idx):
        ax = fig.add_subplot(
            N_CLUSTERS, SAMPLES_PER_CLUSTER,
            c * SAMPLES_PER_CLUSTER + col + 1
        )
        ax.imshow(cv2.cvtColor(all_patches_viz[idx], cv2.COLOR_BGR2RGB))
        ax.axis('off')
        if col == 0:
            ylabel = (f"Cluster {c}\n({count} imgs)\n"
                      f"ar={mean_ar:.1f} sol={mean_sol:.2f}\n{type_str}")
            ax.set_ylabel(ylabel, fontsize=6, rotation=0,
                          labelpad=95, va='center')

plt.tight_layout()
grid_path = os.path.join(OUTPUT_DIR, "cluster_grid.png")
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Cluster grid saved → {grid_path}\n")

# Console summary
print("Cluster distribution summary:")
print(f"  {'Cluster':>8}  {'N':>4}  {'mean_area':>10}  {'mean_ar':>8}  "
      f"{'mean_sol':>9}  fabric_types")
for c in range(N_CLUSTERS):
    indices    = np.where(cluster_labels == c)[0]
    cluster_geo = geo_array_all[indices]
    types_in_cluster = [all_fabric_types[i] for i in indices]
    type_counts = {t: types_in_cluster.count(t)
                   for t in fabric_type_list if t in types_in_cluster}
    type_str = ", ".join(f"{t}:{n}" for t, n in sorted(type_counts.items()))
    print(f"  {c:>8}  {len(indices):>4}  "
          f"{cluster_geo[:,0].mean():>10.4f}  "
          f"{cluster_geo[:,1].mean():>8.2f}  "
          f"{cluster_geo[:,3].mean():>9.3f}  {type_str}")

# ─────────────────────────────────────────────────────────────
# STEP 4b: Calibrate anomaly score threshold from good images
#
# For every fabric type that has a test/good/ folder, run Phase 1
# and collect the anomaly score (mean of top-300 cropped errors).
# The threshold is set as mean + 2*std of those good-image scores,
# matching the evaluation notebook approach exactly.
# This replaces the previously hardcoded THRESHOLD constant.
# ─────────────────────────────────────────────────────────────
print("Calibrating anomaly score threshold from good images...")
good_scores = []

# Discover good-image folders independently — scan ALL fabric type directories,
# not just those that also have anomaly images. This ensures every available
# good image contributes to the threshold, regardless of whether that fabric
# type has any anomaly images in the dataset.
good_folders = []
for entry in sorted(os.listdir(ITD_ROOT)):
    good_dir = os.path.join(ITD_ROOT, entry, "test", "good")
    if os.path.isdir(good_dir):
        good_files = [f for f in os.listdir(good_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if good_files:
            good_folders.append((entry, good_dir, good_files))

if not good_folders:
    raise RuntimeError(
        "No test/good/ folders found under ITD/. "
        "Cannot calibrate anomaly score threshold without good images. "
        "Check that ITD/<fabric>/test/good/ folders exist and are populated."
    )

print(f"Found {len(good_folders)} fabric type(s) with good images:")
for fabric_type, good_dir, good_files in good_folders:
    print(f"  {fabric_type:15s}  ->  {good_dir}  ({len(good_files)} images)")

for fabric_type, good_dir, good_files in good_folders:
    for fname in tqdm(good_files, desc=f"  good/{fabric_type}", leave=False):
        img_bgr = cv2.imread(os.path.join(good_dir, fname))
        if img_bgr is None:
            continue
        _, _, _, score = phase1_outputs(img_bgr)
        good_scores.append(score)

if len(good_scores) < 10:
    raise RuntimeError(
        f"Only {len(good_scores)} good images could be read across all fabric types. "
        "Need at least 10 to calibrate a reliable threshold."
    )

good_scores_arr = np.array(good_scores)
anomaly_score_threshold = float(good_scores_arr.mean() + 2 * good_scores_arr.std())
print(f"Good-image score  mean : {good_scores_arr.mean():.8f}")
print(f"Good-image score  std  : {good_scores_arr.std():.8f}")
print(f"Calibrated anomaly threshold (mean + 2·std): {anomaly_score_threshold:.8f}")
print(f"Calibrated from {len(good_scores)} good images across "
      f"{len(set(all_fabric_types))} fabric types.\n")

# ─────────────────────────────────────────────────────────────
# STEP 5: Save clusters.pkl
# ─────────────────────────────────────────────────────────────
CLUSTER_LABEL_MAP = {i: f"Defect Type {i}" for i in range(N_CLUSTERS)}

cluster_data = {
    "kmeans":             kmeans,
    "embeddings_norm":    embeddings_norm,    # (N, 1280) — pure embedding
    "combined_norm":      combined_norm,       # (N, 1285) — used for clustering
    "geo_arrays":         geo_array_all,       # (N, 5)    — raw geo features
    "geo_scaler":         geo_scaler,          # fitted StandardScaler
    "geo_weight":         GEO_WEIGHT,
    "cluster_labels":     cluster_labels,
    "filenames":          all_filenames,
    "fabric_types":       all_fabric_types,
    "cluster_label_map":  CLUSTER_LABEL_MAP,
    "n_clusters":         N_CLUSTERS,
    "patch_size":         PATCH_SIZE,
    "padding":            PADDING,
    "min_defect_area":    MIN_DEFECT_AREA,
    "confidence_threshold":      calibrated_threshold,
    "anomaly_score_threshold":   anomaly_score_threshold,   # adaptive Phase 1 threshold
    "fabric_models":               fabric_models,
    "min_patches_for_fabric_model": MIN_PATCHES_FOR_FABRIC_MODEL,
}

with open(CLUSTERS_PATH, 'wb') as f:
    pickle.dump(cluster_data, f)

print(f"\nclusters.pkl saved → {CLUSTERS_PATH}")
print("\n" + "=" * 60)
print("NEXT STEPS:")
print(f"  1. Open:  {grid_path}")
print(f"  2. Look at each row → decide defect names")
print(f"     Use the ar (aspect ratio) and sol (solidity) values to help.")
print(f"     High ar = elongated/linear. Low sol = irregular/frayed.")
print(f"  3. Run:   python edit_labels.py   (fill in your names)")
print(f"  4. Run:   python infer.py         (on any new image)")
print("=" * 60)