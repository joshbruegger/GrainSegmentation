"""
Convert U-Net semantic predictions (H×W class indices) to per-instance label maps.

Two strategies:

- Connected components on the interior class (default, matches legacy AJI extraction).
- Marker-controlled watershed on the interior with the predicted boundary as an elevation
  ridge (opt-in). This can split touching interior blobs when seeds and geometry support it;
  it does not guarantee correct splitting when the model omits boundaries and yields a single
  distance peak.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

Connectivity = Literal[1, 2]


def _structure_for_connectivity(ndim: int, connectivity: Connectivity) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    return generate_binary_structure(ndim, connectivity)


def _relabel_sequential(labeled: np.ndarray) -> np.ndarray:
    """Map nonzero labels to 1..K; zero stays zero."""
    ids = sorted(x for x in np.unique(labeled) if x != 0)
    if not ids:
        return np.zeros_like(labeled)
    out = np.zeros_like(labeled)
    for new_id, old_id in enumerate(ids, start=1):
        out[labeled == old_id] = new_id
    return out


def _drop_small_components(labeled: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return labeled
    out = labeled.copy()
    max_id = int(labeled.max())
    for lid in range(1, max_id + 1):
        if (labeled == lid).sum() < min_area_px:
            out[labeled == lid] = 0
    return _relabel_sequential(out)


def semantic_to_instance_label_map(
    semantic: np.ndarray,
    *,
    interior_class: int = 1,
    connectivity: Connectivity = 1,
    min_area_px: int = 0,
) -> np.ndarray:
    """
    Instance IDs from connected components on ``semantic == interior_class``.

    Uses 4-neighborhood when ``connectivity=1`` and 8-neighborhood when ``connectivity=2``,
    matching :func:`scipy.ndimage.label` defaults for 2D when ``connectivity=1``.

    When ``min_area_px`` is 0, returns the raw ``label`` output (same as legacy
    :func:`evaluation.metrics.get_instances`).
    """
    if semantic.ndim != 2:
        raise ValueError(f"semantic must be 2D, got shape {semantic.shape}")
    interior = semantic == interior_class
    structure = _structure_for_connectivity(semantic.ndim, connectivity)
    labeled, _ = label(interior, structure=structure)
    if min_area_px > 0:
        return _drop_small_components(labeled, min_area_px)
    return labeled


def iter_instance_binary_masks(
    instance_label_map: np.ndarray,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(instance_id, mask)`` for each nonzero instance id."""
    for i in sorted(x for x in np.unique(instance_label_map) if x != 0):
        yield int(i), instance_label_map == i


def semantic_to_instance_label_map_watershed(
    semantic: np.ndarray,
    *,
    interior_class: int = 1,
    boundary_class: int = 2,
    min_distance: int = 1,
    footprint: np.ndarray | None = None,
    exclude_border: bool = False,
    boundary_dilate_iter: int = 0,
    ridge_level: float | None = None,
    watershed_connectivity: Connectivity = 1,
    min_area_px: int = 0,
) -> np.ndarray:
    """
    Marker-controlled watershed on interior pixels with boundary as a high ridge.

    ``ridge_level`` defaults to ``-(dt.min()) + dt.max() + 1.0`` on interior so boundaries
    sit above the inverted distance field. ``boundary_dilate_iter`` thickens the boundary
    mask before applying the ridge (0 keeps the raw predicted boundary).

    If no local maxima are found as markers, falls back to
    :func:`semantic_to_instance_label_map` (connected components).
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    if semantic.ndim != 2:
        raise ValueError(f"semantic must be 2D, got shape {semantic.shape}")
    interior = semantic == interior_class
    boundary = semantic == boundary_class
    if not np.any(interior):
        return np.zeros_like(semantic, dtype=np.int32)

    dt = distance_transform_edt(interior)
    if ridge_level is None:
        neg_dt = -dt[interior]
        ridge_level = float(-neg_dt.min() + dt.max() + 1.0)

    elev = np.full(semantic.shape, ridge_level, dtype=np.float64)
    elev[interior] = -dt[interior].astype(np.float64)

    bd = boundary
    if boundary_dilate_iter > 0:
        struct = _structure_for_connectivity(semantic.ndim, 2)
        bd = binary_dilation(
            boundary, structure=struct, iterations=boundary_dilate_iter
        )
    elev[bd] = ridge_level

    interior_labels = interior.astype(np.int32)
    coordinates = peak_local_max(
        dt,
        min_distance=min_distance,
        footprint=footprint,
        labels=interior_labels,
        exclude_border=exclude_border,
    )
    markers = np.zeros(semantic.shape, dtype=np.int32)
    if coordinates.size == 0:
        return semantic_to_instance_label_map(
            semantic,
            interior_class=interior_class,
            connectivity=1,
            min_area_px=min_area_px,
        )
    coord_arr = np.atleast_2d(np.asarray(coordinates, dtype=np.int64))
    if coord_arr.shape[-1] != 2:
        raise ValueError(
            f"peak_local_max must yield an array with shape (n_peaks, 2), got {coord_arr.shape}"
        )
    for i, (row, col) in enumerate(coord_arr):
        markers[int(row), int(col)] = i + 1

    ws_connectivity = max(1, int(watershed_connectivity))
    segmented = watershed(
        elev,
        markers,
        mask=interior,
        connectivity=ws_connectivity,
    ).astype(np.int32)

    segmented[interior & (segmented <= 0)] = 0
    if min_area_px > 0:
        segmented = _drop_small_components(segmented, min_area_px)
    return segmented
