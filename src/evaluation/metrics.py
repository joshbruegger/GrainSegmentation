import numpy as np
from scipy.ndimage import distance_transform_edt

from evaluation.instance_masks import semantic_to_instance_label_map


def compute_semantic_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3
):
    """
    Computes Mask IoU and Dice score for each class.
    y_true: 2D array of true labels
    y_pred: 2D array of predicted labels
    """
    metrics = {}
    for c in range(num_classes):
        true_c = y_true == c
        pred_c = y_pred == c

        intersection = np.logical_and(true_c, pred_c).sum()
        union = np.logical_or(true_c, pred_c).sum()

        iou = intersection / union if union > 0 else float("nan")

        sum_mask = true_c.sum() + pred_c.sum()
        dice = 2 * intersection / sum_mask if sum_mask > 0 else float("nan")

        metrics[f"iou_class_{c}"] = float(iou)
        metrics[f"dice_class_{c}"] = float(dice)

    return metrics


def compute_boundary_f1(
    y_true: np.ndarray, y_pred: np.ndarray, class_idx: int = 2, tolerance: float = 2.0
):
    """
    Computes Boundary F1 Score for a specific class (default: grain boundary = 2).
    A predicted boundary pixel is correct if it lies within `tolerance` distance of a true boundary pixel.
    """
    true_bnd = y_true == class_idx
    pred_bnd = y_pred == class_idx

    if true_bnd.sum() == 0 and pred_bnd.sum() == 0:
        return 1.0
    if true_bnd.sum() == 0 or pred_bnd.sum() == 0:
        return 0.0

    dist_to_true = distance_transform_edt(~true_bnd)
    dist_to_pred = distance_transform_edt(~pred_bnd)

    tp_pred = np.logical_and(pred_bnd, dist_to_true <= tolerance).sum()
    precision = tp_pred / pred_bnd.sum() if pred_bnd.sum() > 0 else 0.0

    tp_true = np.logical_and(true_bnd, dist_to_pred <= tolerance).sum()
    recall = tp_true / true_bnd.sum() if true_bnd.sum() > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)


def compute_boundary_iou(
    y_true: np.ndarray, y_pred: np.ndarray, class_idx: int = 2, tolerance: float = 2.0
):
    """
    Computes Boundary IoU.
    Dilates the boundaries of the specific class by `tolerance` and computes IoU.
    """
    true_bnd = y_true == class_idx
    pred_bnd = y_pred == class_idx

    if true_bnd.sum() == 0 and pred_bnd.sum() == 0:
        return 1.0
    if true_bnd.sum() == 0 or pred_bnd.sum() == 0:
        return 0.0

    thick_true_bnd = distance_transform_edt(~true_bnd) <= tolerance
    thick_pred_bnd = distance_transform_edt(~pred_bnd) <= tolerance

    inter = np.logical_and(thick_true_bnd, thick_pred_bnd).sum()
    union = np.logical_or(thick_true_bnd, thick_pred_bnd).sum()

    return float(inter / union) if union > 0 else 0.0


def get_instances(semantic_mask: np.ndarray, interior_class: int = 1):
    """
    Extracts instance masks from semantic masks by finding connected components of the interior class.
    """
    return semantic_to_instance_label_map(
        semantic_mask, interior_class=interior_class, connectivity=1, min_area_px=0
    )


def compute_aji(true_instances: np.ndarray, pred_instances: np.ndarray):
    """
    Computes Aggregated Jaccard Index (AJI).
    true_instances: 2D array of true instance labels (0 = background)
    pred_instances: 2D array of predicted instance labels (0 = background)
    """
    true_id_list = list(np.unique(true_instances))
    pred_id_list = list(np.unique(pred_instances))

    if 0 in true_id_list:
        true_id_list.remove(0)
    if 0 in pred_id_list:
        pred_id_list.remove(0)

    if not true_id_list and not pred_id_list:
        return 1.0
    if not true_id_list or not pred_id_list:
        return 0.0

    max_true = int(true_instances.max())
    max_pred = int(pred_instances.max())

    intersection_matrix = np.histogram2d(
        true_instances.flatten(),
        pred_instances.flatten(),
        bins=(max_true + 1, max_pred + 1),
        range=((0, max_true + 1), (0, max_pred + 1)),
    )[0]

    true_areas = intersection_matrix.sum(axis=1)
    pred_areas = intersection_matrix.sum(axis=0)

    overall_intersection = 0
    overall_union = 0

    unassigned_pred_ids = set(pred_id_list)

    for true_id in true_id_list:
        candidate_pred_ids = sorted(unassigned_pred_ids)
        if not candidate_pred_ids:
            overall_union += true_areas[true_id]
            continue

        intersections = np.array(
            [intersection_matrix[true_id, pred_id] for pred_id in candidate_pred_ids]
        )
        if intersections.sum() == 0:
            overall_union += true_areas[true_id]
            continue

        pred_areas_subset = np.array(
            [pred_areas[pred_id] for pred_id in candidate_pred_ids]
        )
        unions = true_areas[true_id] + pred_areas_subset - intersections

        ious = intersections / np.maximum(unions, 1)
        best_idx = np.argmax(ious)
        best_pred_id = candidate_pred_ids[best_idx]

        if ious[best_idx] > 0:
            overall_intersection += intersections[best_idx]
            overall_union += unions[best_idx]
            unassigned_pred_ids.remove(best_pred_id)
        else:
            overall_union += true_areas[true_id]

    for pred_id in unassigned_pred_ids:
        overall_union += pred_areas[pred_id]

    return float(overall_intersection / overall_union)
