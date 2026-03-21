import numpy as np

from evaluation.instance_masks import semantic_to_instance_label_map

# IoU thresholds for mP/mR/mF1@0.5:0.95 (COCO-style sweep).
IOU_THRESHOLDS_50_95 = tuple(np.arange(0.50, 1.0, 0.05))


def get_instances(semantic_mask: np.ndarray, interior_class: int = 1):
    """
    Extracts instance masks from semantic masks by finding connected components of the interior class.
    """
    return semantic_to_instance_label_map(
        semantic_mask, interior_class=interior_class, connectivity=1, min_area_px=0
    )


def _instance_ids(instance_map: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(instance_map) if x != 0)


def build_instance_iou_matrix(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    """
    Pairwise IoU between each GT instance and each predicted instance (0 = background).
    Rows: true_ids, columns: pred_ids.
    """
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)
    mat = np.zeros((nt, np_), dtype=np.float64)
    for i, tid in enumerate(true_ids):
        tm = true_instances == tid
        for j, pid in enumerate(pred_ids):
            pm = pred_instances == pid
            inter = np.logical_and(tm, pm).sum()
            union = np.logical_or(tm, pm).sum()
            mat[i, j] = float(inter) / float(union) if union > 0 else 0.0
    return mat, true_ids, pred_ids


def greedy_one_to_one_tp_count(iou_matrix: np.ndarray, iou_threshold: float) -> int:
    """
    Greedy maximum matching: sort candidate pairs by IoU descending, assign disjoint pairs.
    Returns the number of matched pairs (TP count).
    """
    if iou_matrix.size == 0:
        return 0
    nt, np_ = iou_matrix.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(nt):
        for j in range(np_):
            v = float(iou_matrix[i, j])
            if v >= iou_threshold:
                candidates.append((v, i, j))
    candidates.sort(key=lambda x: -x[0])
    used_row: set[int] = set()
    used_col: set[int] = set()
    tp = 0
    for _, i, j in candidates:
        if i in used_row or j in used_col:
            continue
        used_row.add(i)
        used_col.add(j)
        tp += 1
    return tp


def compute_instance_precision_recall_f1(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    iou_threshold: float,
) -> tuple[float, float, float]:
    """
    One-to-one instance matching at ``iou_threshold``; precision / recall / F1 from TP / FP / FN.
    """
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

    if nt == 0 and np_ == 0:
        return 1.0, 1.0, 1.0
    if nt == 0:
        return 0.0, 0.0, 0.0
    if np_ == 0:
        return 0.0, 0.0, 0.0

    iou_matrix, _, _ = build_instance_iou_matrix(true_instances, pred_instances)
    tp = greedy_one_to_one_tp_count(iou_matrix, iou_threshold)
    fp = np_ - tp
    fn = nt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def compute_instance_prf_mean_iou_sweep(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    thresholds: tuple[float, ...] = IOU_THRESHOLDS_50_95,
) -> tuple[float, float, float]:
    """
    Mean precision, recall, and F1 over ``thresholds`` (default 0.50, 0.55, ..., 0.95).
    """
    if not thresholds:
        return float("nan"), float("nan"), float("nan")
    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for t in thresholds:
        p, r, f = compute_instance_precision_recall_f1(
            true_instances, pred_instances, t
        )
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))


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
