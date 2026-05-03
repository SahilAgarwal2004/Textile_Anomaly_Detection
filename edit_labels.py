"""
EDIT CLUSTER LABELS
===================
Run this AFTER inspecting cluster_grid.png and per-fabric grids.

Usage:
    python edit_labels.py
"""

import pickle

CLUSTERS_PATH = "clusters.pkl"

with open(CLUSTERS_PATH, 'rb') as f:
    data = pickle.load(f)

# ─────────────────────────────────────────────────────────────
# 1. GLOBAL MODEL LABELS
# ─────────────────────────────────────────────────────────────
GLOBAL_LABEL_MAP = {
    0: "thread_defect",
    1: "unknown_defect",
    2: "edge_fraying",
    3: "structural_damage",
    4: "unknown_defect",
    5: "spot_contamination",
}

# ─────────────────────────────────────────────────────────────
# 2. PER-FABRIC MODEL LABELS
# ─────────────────────────────────────────────────────────────
FABRIC_LABEL_MAPS = {
    "type1cam1": {
        0: "thread_defect",
        1: "structural_damage",
        2: "thread_loop",
        3: "spot_contamination",
        4: "edge_fraying",
        5: "spot_contamination_fine",
    },
    "type2cam2": {
        0: "thread_pull",
        1: "thread_defect",
        2: "spot_contamination",
    },
    "type3cam1": {
        0: "edge_damage",
        1: "structural_damage",
        2: "spot_contamination",
        3: "spot_contamination_fine",
        4: "unknown_defect",
        5: "hole",
    },
    "type5cam2": {
        0: "spot_contamination",
        1: "multi_spot",
        2: "large_spot",
        3: "contamination_cluster",
        4: "large_contamination",
        5: "micro_spot",
    },
    "type7cam2": {
        0: "edge_fraying",
        1: "spot_contamination",
        2: "thread_defect",
        3: "thread_pull",
        4: "thread_loop_curl",
        5: "structural_damage",
    },
    "type8cam1": {
        0: "spot_contamination",
        1: "thread_loop_curl",
        2: "edge_fraying",
        3: "structural_damage",
    },
    "type9cam2": {
        0: "color_shift",
        1: "spot_contamination",
        2: "spot_contamination_fine",
        3: "structural_damage",
        4: "spot_contamination",
        5: "unknown_defect",
    },
}

# ─────────────────────────────────────────────────────────────
# Apply global labels
# ─────────────────────────────────────────────────────────────
n_global = data["n_clusters"]
for i in range(n_global):
    if i not in GLOBAL_LABEL_MAP:
        GLOBAL_LABEL_MAP[i] = f"Defect Type {i}"

data["cluster_label_map"] = GLOBAL_LABEL_MAP

print("Global model labels applied:")
for k, v in sorted(GLOBAL_LABEL_MAP.items()):
    count = (data["cluster_labels"] == k).sum()
    print(f"  Cluster {k}: '{v}'  ({count} training images)")

# ─────────────────────────────────────────────────────────────
# Apply per-fabric labels
# ─────────────────────────────────────────────────────────────
print("\nPer-fabric model labels applied:")
fabric_models = data.get("fabric_models", {})

for fabric_type, fm in fabric_models.items():
    if fabric_type in FABRIC_LABEL_MAPS:
        label_map = FABRIC_LABEL_MAPS[fabric_type]
        n_fab     = fm["n_clusters"]
        for i in range(n_fab):
            if i not in label_map:
                label_map[i] = f"Defect Type {i}"
        fm["cluster_label_map"] = label_map
        print(f"  {fabric_type}: {n_fab} clusters → {list(label_map.values())}")
    else:
        print(f"  {fabric_type}: no label map provided, keeping placeholders")

data["fabric_models"] = fabric_models

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
with open(CLUSTERS_PATH, 'wb') as f:
    pickle.dump(data, f)

print(f"\nclusters.pkl updated successfully.")
