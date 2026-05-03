"""
main.py — Textile Anomaly Detection: Core Inference Engine
===========================================================
This is the single entry point for running the full pipeline
(Phase 1 detection + Phase 2 explanation) on any image.

Can be used in two ways:

  1. Standalone:
       python main.py --image ITD/type5cam2/test/anomaly/24135.png

  2. Imported by app.py:
       from main import load_models, run_inference

The function run_inference() returns a structured dict that
app.py can serialize to JSON and send to the frontend.
"""

import os
import sys
import argparse
import pickle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import normalize
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for both CLI and server
import matplotlib.pyplot as plt
import base64
import io

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_PATH    = "model/unet_modified_model.h5"
CLUSTERS_PATH = "clusters.pkl"
ITD_ROOT      = "ITD"

IMG_HEIGHT      = 128
IMG_WIDTH       = 128
THRESHOLD       = 0.0015206437
PIXEL_THRESHOLD = 0.2
crop            = 10

# ─────────────────────────────────────────────────────────────
# Loss functions (required to load the U-Net)
# ─────────────────────────────────────────────────────────────
_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
_vgg.trainable = False

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def combined_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred)) + ssim_loss(y_true, y_pred)

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(
        _vgg(preprocess_input(y_true * 255.0)) -
        _vgg(preprocess_input(y_pred * 255.0))
    ))

def final_loss(y_true, y_pred):
    return combined_loss(y_true, y_pred) + 0.1 * perceptual_loss(y_true, y_pred)

# ─────────────────────────────────────────────────────────────
# Model loading — called once at startup
# ─────────────────────────────────────────────────────────────
_unet        = None
_embedder    = None
_cluster_data = None

def load_models():
    """
    Loads all models and cluster data into module-level globals.
    Call this once at app startup. Safe to call multiple times.
    """
    global _unet, _embedder, _cluster_data

    if _unet is not None:
        return   # already loaded

    print("Loading Phase 1 model (U-Net)...")
    _unet = load_model(
        MODEL_PATH,
        custom_objects={
            "final_loss":    final_loss,
            "ssim_loss":     ssim_loss,
            "combined_loss": combined_loss,
        }
    )
    print("Phase 1 model loaded.")

    print("Loading Phase 2 cluster data...")
    with open(CLUSTERS_PATH, 'rb') as f:
        _cluster_data = pickle.load(f)
    print("Cluster data loaded.")

    patch_size = _cluster_data["patch_size"]
    print("Loading EfficientNetB0 embedder...")
    _embedder = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(patch_size, patch_size, 3),
        pooling='avg'
    )
    _embedder.trainable = False
    print("All models ready.\n")

# ─────────────────────────────────────────────────────────────
# Phase 1: reconstruction + error map + score + mask
# ─────────────────────────────────────────────────────────────
def _phase1(img_bgr):
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))
    img_norm    = img_resized / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)
    recon       = _unet.predict(img_input, verbose=0)[0]
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
    score       = float(np.mean(np.sort(flat)[-5:]))
    return norm_error, mask, img_norm, score, recon

# ─────────────────────────────────────────────────────────────
# Geometric features
# ─────────────────────────────────────────────────────────────
def _geometric_features(mask, stats, largest):
    x    = stats[largest, cv2.CC_STAT_LEFT]
    y    = stats[largest, cv2.CC_STAT_TOP]
    w    = stats[largest, cv2.CC_STAT_WIDTH]
    h    = stats[largest, cv2.CC_STAT_HEIGHT]
    area = stats[largest, cv2.CC_STAT_AREA]

    image_area     = IMG_HEIGHT * IMG_WIDTH
    image_diagonal = np.sqrt(IMG_HEIGHT**2 + IMG_WIDTH**2)
    aspect_ratio   = w / max(h, 1)
    extent         = area / max(w * h, 1)

    component_mask = (mask[y:y+h, x:x+w] > 0).astype(np.uint8)
    contours, _    = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
        solidity  = min(1.0, area / max(hull_area, 1))
    else:
        solidity = 1.0

    equiv_diameter = np.sqrt(4 * area / np.pi) / image_diagonal

    geo_dict = {
        "area":           float(area / image_area),
        "aspect_ratio":   float(aspect_ratio),
        "extent":         float(extent),
        "solidity":       float(solidity),
        "equiv_diameter": float(equiv_diameter),
    }
    geo_array = np.array(list(geo_dict.values()), dtype=np.float32)
    return geo_dict, geo_array

# ─────────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────────
def _extract_patch(img_norm, norm_error, mask):
    MIN_DEFECT_AREA = _cluster_data["min_defect_area"]
    PADDING         = _cluster_data["padding"]
    PATCH_SIZE      = _cluster_data["patch_size"]

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, None, None, None, None

    valid = [i for i in range(1, num_labels)
             if stats[i, cv2.CC_STAT_AREA] >= MIN_DEFECT_AREA]
    if not valid:
        return None, None, None, None, None

    largest          = max(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    geo_dict, geo_array = _geometric_features(mask, stats, largest)

    x  = stats[largest, cv2.CC_STAT_LEFT]
    y  = stats[largest, cv2.CC_STAT_TOP]
    w  = stats[largest, cv2.CC_STAT_WIDTH]
    h  = stats[largest, cv2.CC_STAT_HEIGHT]
    x1 = max(0,          x - PADDING)
    y1 = max(0,          y - PADDING)
    x2 = min(IMG_WIDTH,  x + w + PADDING)
    y2 = min(IMG_HEIGHT, y + h + PADDING)

    patch_orig     = img_norm[y1:y2, x1:x2]
    weight         = norm_error[y1:y2, x1:x2, np.newaxis]
    weight         = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
    patch_weighted = patch_orig * (0.5 + 0.5 * weight)
    patch_emb      = cv2.resize(
        (patch_weighted * 255).clip(0, 255).astype(np.uint8),
        (PATCH_SIZE, PATCH_SIZE)
    )
    patch_viz      = cv2.resize(
        (patch_orig * 255).clip(0, 255).astype(np.uint8),
        (PATCH_SIZE, PATCH_SIZE)
    )
    return patch_emb, patch_viz, geo_dict, geo_array, (x1, y1, x2, y2)

# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
def _embed(patch_uint8):
    x = eff_preprocess(patch_uint8.astype(np.float32))
    return _embedder.predict(np.expand_dims(x, 0), verbose=0)[0]

# ─────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────
def _detect_fabric_type(image_path):
    parts = image_path.replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("type") and "cam" in part:
            return part
    return None


def _classify(embedding, geo_array, fabric_type):
    cd           = _cluster_data
    geo_scaler   = cd["geo_scaler"]
    geo_weight   = cd["geo_weight"]
    fabric_models = cd["fabric_models"]

    emb_norm     = normalize(embedding.reshape(1, -1))
    geo_scaled   = geo_scaler.transform(geo_array.reshape(1, -1))
    geo_weighted = geo_scaled * geo_weight
    combined     = normalize(np.concatenate([emb_norm, geo_weighted], axis=1))

    if fabric_type and fabric_type in fabric_models:
        fm           = fabric_models[fabric_type]
        km           = fm["kmeans"]
        label_map    = fm["cluster_label_map"]
        threshold    = fm["confidence_threshold"]
        model_used   = f"per-fabric ({fabric_type})"
    else:
        km           = cd["kmeans"]
        label_map    = cd["cluster_label_map"]
        threshold    = cd["confidence_threshold"]
        model_used   = "global"

    distances    = km.transform(combined)[0]
    cluster_id   = int(np.argmin(distances))
    distance     = float(distances[cluster_id])
    defect_label = label_map.get(cluster_id, f"Type {cluster_id}")
    confidence   = "high" if distance <= threshold else "low"
    return cluster_id, defect_label, distance, confidence, model_used

# ─────────────────────────────────────────────────────────────
# Explanation text
# ─────────────────────────────────────────────────────────────
def _build_explanation(defect_label, geo_dict):
    ar  = geo_dict["aspect_ratio"]
    sol = geo_dict["solidity"]
    a   = geo_dict["area"]
    ed  = geo_dict["equiv_diameter"]

    size_str  = "small" if a < 0.005 else ("medium" if a < 0.02 else "large")
    shape_str = "elongated" if ar > 2.5 else ("narrow" if ar < 0.5 else "compact")
    edge_str  = ("irregular/frayed edges" if sol < 0.6
                 else ("smooth edges" if sol > 0.85
                       else "moderately irregular edges"))
    px_size   = int(ed * np.sqrt(IMG_HEIGHT**2 + IMG_WIDTH**2))

    return (f"{defect_label} — {size_str} {shape_str} defect, {edge_str} "
            f"(~{px_size}px, aspect ratio: {ar:.1f}, solidity: {sol:.2f})")

# ─────────────────────────────────────────────────────────────
# Similarity retrieval
# ─────────────────────────────────────────────────────────────
def _find_similar(embedding, image_path, top_k=3):
    cd              = _cluster_data
    emb_norm        = normalize(embedding.reshape(1, -1))
    similarities    = (cd["embeddings_norm"] @ emb_norm.T).flatten()

    parts       = image_path.replace("\\", "/").split("/")
    query_fname = parts[-4] + "/" + parts[-1] if len(parts) >= 4 else ""

    top_indices = [
        i for i in np.argsort(similarities)[::-1]
        if cd["filenames"][i] != query_fname
    ][:top_k]

    results = []
    for idx in top_indices:
        sim   = float(similarities[idx])
        fname = cd["filenames"][idx]
        p     = fname.split("/")
        fpath = os.path.join(ITD_ROOT, p[0], "test", "anomaly", p[1])
        img   = cv2.imread(fpath)
        if img is not None:
            img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)
        label = cd["cluster_label_map"].get(
            cd["cluster_labels"][cd["filenames"].index(fname)], "unknown"
        )
        results.append({
            "similarity": round(sim, 4),
            "filename":   fname,
            "label":      label,
            "image":      img,   # numpy RGB uint8 or None
        })
    return results

# ─────────────────────────────────────────────────────────────
# Image → base64 helper (for JSON serialization in app.py)
# ─────────────────────────────────────────────────────────────
def _img_to_b64(img_rgb_uint8):
    """Converts a numpy RGB uint8 image to a base64 PNG string."""
    if img_rgb_uint8 is None:
        return None
    _, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("utf-8")


def _fig_to_b64(fig):
    """Converts a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ─────────────────────────────────────────────────────────────
# MAIN INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────
def run_inference(image_path):
    """
    Runs the full Phase 1 + Phase 2 pipeline on a single image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict with keys:
          status          : "ok" or "error"
          prediction      : "NORMAL" or "ANOMALY"
          score           : float anomaly score
          defect_type     : str or None
          confidence      : "high" or "low" or None
          model_used      : "per-fabric (...)" or "global" or None
          explanation     : str human-readable description or None
          similar_patches : list of dicts (only when confidence == "low")
                            each dict: {similarity, filename, label, image_b64}
          images:
            original_b64      : base64 PNG of original image with bbox
            patch_b64         : base64 PNG of defect patch (or None)
            visualization_b64 : base64 PNG of full 5-panel figure
    """
    if _unet is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    # ── Read image ──
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"status": "error", "message": f"Cannot read image: {image_path}"}

    # ── Phase 1 ──
    norm_error, mask, img_norm, score, recon = _phase1(img_bgr)
    prediction = "ANOMALY" if score > THRESHOLD else "NORMAL"

    # ── Initialise result fields ──
    defect_type     = None
    confidence      = None
    model_used      = None
    explanation     = None
    similar_patches = []
    patch_viz       = None
    bbox            = None
    cluster_id      = None

    # ── Phase 2 (only if ANOMALY) ──
    if prediction == "ANOMALY":
        patch_emb, patch_viz, geo_dict, geo_array, bbox = _extract_patch(
            img_norm, norm_error, mask
        )

        if patch_emb is not None:
            embedding   = _embed(patch_emb)
            fabric_type = _detect_fabric_type(image_path)
            cluster_id, defect_type, distance, confidence, model_used = _classify(
                embedding, geo_array, fabric_type
            )
            explanation = _build_explanation(defect_type, geo_dict)

            if confidence == "low":
                raw_similar     = _find_similar(embedding, image_path, top_k=3)
                defect_type     = f"Uncertain (closest: {defect_type})"
                similar_patches = [
                    {
                        "similarity": s["similarity"],
                        "filename":   s["filename"],
                        "label":      s["label"],
                        "image_b64":  _img_to_b64(s["image"]),
                    }
                    for s in raw_similar
                ]
        else:
            defect_type = "Unlocalized"
            explanation = "Defect region too small or mask could not localize it."

    # ── Build visualization figure ──
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bbox = img_rgb.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 80, 80), 2)

    show_patch = prediction == "ANOMALY" and patch_viz is not None
    n_panels   = 5 if show_patch else 4

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))

    axes[0].imshow(img_bbox)
    axes[0].set_title("Original" + (" (+bbox)" if bbox is not None else ""))
    axes[0].axis("off")

    axes[1].imshow(recon.astype(np.float32))
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(norm_error.astype(np.float32), cmap="hot")
    axes[2].set_title("Error Map")
    axes[2].axis("off")

    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title(f"Mask ({prediction})")
    axes[3].axis("off")

    if show_patch:
        axes[4].imshow(cv2.cvtColor(patch_viz, cv2.COLOR_BGR2RGB))
        axes[4].set_title("Defect Patch")
        axes[4].axis("off")

    if prediction == "NORMAL":
        title = f"Score: {score:.6e}  |  Prediction: NORMAL"
    else:
        conf_str = f"  [confidence: {confidence}]" if confidence else ""
        title    = f"Score: {score:.6e}  |  Prediction: ANOMALY{conf_str}"

    fig.suptitle(title, fontsize=11, y=1.01)

    if explanation:
        fig.text(0.5, -0.02, explanation, ha="center", fontsize=9,
                 color="#222222", style="italic",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5",
                           edgecolor="#cccccc", alpha=0.9))

    plt.tight_layout()
    viz_b64 = _fig_to_b64(fig)
    plt.close(fig)

    # ── Patch image as base64 ──
    patch_b64 = None
    if patch_viz is not None:
        patch_b64 = _img_to_b64(cv2.cvtColor(patch_viz, cv2.COLOR_BGR2RGB))

    # ── Original with bbox as base64 ──
    original_b64 = _img_to_b64(img_bbox)

    # ── Print summary to console ──
    print(f"\n{'='*50}")
    print(f"Image     : {image_path}")
    print(f"Score     : {score:.6e}")
    print(f"Prediction: {prediction}")
    if prediction == "ANOMALY":
        print(f"Defect    : {defect_type}")
        print(f"Confidence: {confidence}")
        print(f"Model     : {model_used}")
        print(f"Reasoning : {explanation}")
        if similar_patches:
            print(f"Similar   : {[s['filename'] for s in similar_patches]}")
    print(f"{'='*50}\n")

    return {
        "status":          "ok",
        "prediction":      prediction,
        "score":           round(score, 10),
        "defect_type":     defect_type,
        "confidence":      confidence,
        "model_used":      model_used,
        "explanation":     explanation,
        "similar_patches": similar_patches,
        "images": {
            "original_b64":       original_b64,
            "patch_b64":          patch_b64,
            "visualization_b64":  viz_b64,
        }
    }

# ─────────────────────────────────────────────────────────────
# Standalone CLI usage
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Textile Anomaly Detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    load_models()
    result = run_inference(args.image)

    if result["status"] == "error":
        print(f"ERROR: {result['message']}")
        sys.exit(1)

    print(f"Prediction : {result['prediction']}")
    if result["prediction"] == "ANOMALY":
        print(f"Defect     : {result['defect_type']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Reasoning  : {result['explanation']}")
        if result["similar_patches"]:
            print("Similar known defects:")
            for s in result["similar_patches"]:
                print(f"  {s['filename']}  sim={s['similarity']:.3f}  label={s['label']}")