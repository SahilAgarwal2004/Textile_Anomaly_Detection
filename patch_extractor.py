"""
PHASE 2 - STEP 1: Patch Extraction
====================================
Extracts defect patches from anomaly masks using connected component analysis.

Input  : original image (H x W x 3), binary mask (H x W)
Output : list of cropped defect patches + metadata (bbox, area, centroid)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


# ─────────────────────────────────────────────
# Data container for a single defect patch
# ─────────────────────────────────────────────
@dataclass
class DefectPatch:
    patch_id: int                        # unique id within the image
    image_path: str                      # source image path (for traceability)
    crop: np.ndarray                     # cropped RGB patch (H x W x 3)
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in original image coords
    padded_bbox: Tuple[int, int, int, int]  # bbox after padding
    area: int                            # number of mask pixels in component
    centroid: Tuple[float, float]        # (cx, cy) in original image coords
    embedding: Optional[np.ndarray] = None   # filled later in Step 2
    cluster_id: Optional[int] = None         # filled later in Step 3
    cluster_label: Optional[str] = None      # filled later in Step 4


# ─────────────────────────────────────────────
# Core patch extractor
# ─────────────────────────────────────────────
class PatchExtractor:
    """
    Extracts defect patches from a binary anomaly mask.

    Parameters
    ----------
    min_area : int
        Minimum pixel area to keep a connected component.
        Tune this based on your image resolution (128x128 → start at 10).
    padding : int
        Pixels to pad around the tight bounding box before cropping.
    mask_threshold : int
        Binarization threshold for the mask (0–255).
        Use 127 for already-binary masks; lower if your mask is soft.
    morph_close_ksize : int
        Kernel size for morphological closing before component analysis.
        Helps merge nearby pixels into one component. Set to 0 to disable.
    """

    def __init__(
        self,
        min_area: int = 10,
        padding: int = 10,
        mask_threshold: int = 127,
        morph_close_ksize: int = 3,
    ):
        self.min_area = min_area
        self.padding = padding
        self.mask_threshold = mask_threshold
        self.morph_close_ksize = morph_close_ksize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_path: str = "unknown",
    ) -> List[DefectPatch]:
        """
        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3), uint8, RGB
        mask  : np.ndarray  shape (H, W),    uint8 or float
                            binary or soft mask (will be binarized internally)
        image_path : str    source path, stored for traceability

        Returns
        -------
        List[DefectPatch]  — one entry per defect region found
        """
        H, W = image.shape[:2]

        # 1. Binarize mask
        binary_mask = self._binarize(mask)

        # 2. Optional morphological closing to merge nearby blobs
        if self.morph_close_ksize > 0:
            binary_mask = self._morph_close(binary_mask, self.morph_close_ksize)

        # 3. Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        patches: List[DefectPatch] = []
        patch_id = 0

        # label 0 = background → skip
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])

            # 4. Filter small noise
            if area < self.min_area:
                continue

            # 5. Tight bounding box
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            # 6. Add padding (clamp to image bounds)
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(W, x + w + self.padding)
            y2 = min(H, y + h + self.padding)

            # 7. Crop from ORIGINAL image (not mask)
            crop = image[y1:y2, x1:x2].copy()

            cx, cy = float(centroids[label_idx][0]), float(centroids[label_idx][1])

            patch = DefectPatch(
                patch_id=patch_id,
                image_path=image_path,
                crop=crop,
                bbox=(x, y, w, h),
                padded_bbox=(x1, y1, x2 - x1, y2 - y1),
                area=area,
                centroid=(cx, cy),
            )
            patches.append(patch)
            patch_id += 1

        return patches

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _binarize(self, mask: np.ndarray) -> np.ndarray:
        """Convert mask to uint8 binary (0 or 255)."""
        if mask.dtype != np.uint8:
            # float mask in [0, 1] → scale to [0, 255]
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
        return binary

    def _morph_close(self, binary_mask: np.ndarray, ksize: int) -> np.ndarray:
        """Morphological closing: fills small gaps between nearby blobs."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)


# ─────────────────────────────────────────────
# Visualization helper
# ─────────────────────────────────────────────
def visualize_patches(
    image: np.ndarray,
    patches: List[DefectPatch],
    save_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Draw bounding boxes on the original image and display extracted patches.

    Returns the annotated image (numpy array, RGB).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    annotated = image.copy()

    # Draw padded bounding boxes on the original image
    for p in patches:
        x, y, w, h = p.padded_bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"#{p.patch_id}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    n_patches = len(patches)
    fig_cols = n_patches + 1  # +1 for annotated original
    fig, axes = plt.subplots(1, fig_cols, figsize=(4 * fig_cols, 4))

    if fig_cols == 1:
        axes = [axes]

    axes[0].imshow(annotated)
    axes[0].set_title("Detected Defects")
    axes[0].axis("off")

    for i, p in enumerate(patches):
        axes[i + 1].imshow(p.crop)
        axes[i + 1].set_title(
            f"Patch #{p.patch_id}\nArea: {p.area}px\n{p.crop.shape[1]}x{p.crop.shape[0]}"
        )
        axes[i + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[PatchExtractor] Saved visualization → {save_path}")

    if show:
        plt.show()

    plt.close()
    return annotated


# ─────────────────────────────────────────────
# Quick test / usage example
# ─────────────────────────────────────────────
def demo():
    """
    Demonstrates patch extraction on a synthetic image + mask.
    Replace `image` and `mask` with your actual outputs from Phase 1.
    """
    print("=== PatchExtractor Demo ===")

    # --- Synthetic data (replace with your real image + mask) ---
    H, W = 128, 128
    image = cv2.imread("ITD/type2cam2/test/anomaly/8047.png")

    # Simulate a fabric texture (slightly noisy blue-gray)
    image = (np.ones((H, W, 3)) * [50, 60, 70]).astype(np.uint8)
    image += np.random.randint(0, 15, (H, W, 3), dtype=np.uint8)

    # Synthetic mask: two white blobs simulating defects
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (30, 100), 6, 255, -1)   # small defect
    cv2.circle(mask, (90, 40), 10, 255, -1)   # larger defect

    # --- Extract patches ---
    extractor = PatchExtractor(min_area=10, padding=10)
    patches = extractor.extract(image, mask, image_path="demo_image.png")

    print(f"Found {len(patches)} defect patches:")
    for p in patches:
        print(
            f"  Patch #{p.patch_id} | area={p.area}px | "
            f"centroid=({p.centroid[0]:.1f}, {p.centroid[1]:.1f}) | "
            f"crop shape={p.crop.shape}"
        )

    # Visualize
    os.makedirs("outputs", exist_ok=True)
    visualize_patches(image, patches, save_path="outputs/demo_patches.png")
    print("Demo complete. Check outputs/demo_patches.png")


if __name__ == "__main__":
    demo()