"""
INSPECT PER-FABRIC CLUSTERS
============================
Generates a separate cluster grid image for each fabric type
that has its own model. Use these grids to decide what to put
in FABRIC_LABEL_MAPS inside edit_labels.py.

Usage:
    python inspect_fabric_clusters.py

Output: phase2_output/clusters_<fabric_type>.png for each fabric model
"""

import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

CLUSTERS_PATH = "clusters.pkl"
OUTPUT_DIR    = "phase2_output"
ITD_ROOT      = "ITD"

with open(CLUSTERS_PATH, 'rb') as f:
    data = pickle.load(f)

fabric_models    = data.get("fabric_models", {})
all_patches_viz  = None   # we need to reload patches from disk since they aren't saved
all_filenames    = data["filenames"]
all_fabric_types = data["fabric_types"]

if not fabric_models:
    print("No per-fabric models found in clusters.pkl.")
    exit()

print(f"Found {len(fabric_models)} per-fabric models.\n")

SAMPLES_PER_CLUSTER = 8

for fabric_type, fm in fabric_models.items():
    indices       = fm["indices"]          # indices into global arrays
    km            = fm["kmeans"]
    n_clusters    = fm["n_clusters"]
    label_map     = fm["cluster_label_map"]
    fabric_combined = fm["combined_norm"]

    # Get cluster assignments for this fabric
    fabric_labels = km.predict(fabric_combined)

    # Load patch images from disk for this fabric
    patches = []
    for idx in indices:
        fname    = all_filenames[idx]
        parts    = fname.split("/")
        fullpath = os.path.join(ITD_ROOT, parts[0], "test", "anomaly", parts[1])
        img      = cv2.imread(fullpath)
        if img is not None:
            img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)
        patches.append(img)

    # Generate grid
    fig = plt.figure(figsize=(SAMPLES_PER_CLUSTER * 1.4, n_clusters * 2.4))
    fig.suptitle(
        f"PER-FABRIC CLUSTER GRID: {fabric_type}\n"
        f"{len(indices)} patches  |  {n_clusters} clusters\n"
        f"Fill in FABRIC_LABEL_MAPS['{fabric_type}'] in edit_labels.py",
        fontsize=10, y=1.02
    )

    for c in range(n_clusters):
        cluster_indices = np.where(fabric_labels == c)[0]
        count           = len(cluster_indices)
        current_label   = label_map.get(c, f"Defect Type {c}")

        sample_idx = np.random.choice(
            cluster_indices,
            size=min(SAMPLES_PER_CLUSTER, count),
            replace=False
        )

        for col, local_idx in enumerate(sample_idx):
            ax = fig.add_subplot(
                n_clusters, SAMPLES_PER_CLUSTER,
                c * SAMPLES_PER_CLUSTER + col + 1
            )
            img = patches[local_idx]
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor('#333333')
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(
                    f"Cluster {c}\n({count} imgs)\n[{current_label}]",
                    fontsize=7, rotation=0, labelpad=80, va='center'
                )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"clusters_{fabric_type}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {fabric_type}: saved → {out_path}")

print(f"\nAll per-fabric grids saved to {OUTPUT_DIR}/")
print("Open each grid, decide labels, fill FABRIC_LABEL_MAPS in edit_labels.py,")
print("then re-run edit_labels.py.")
