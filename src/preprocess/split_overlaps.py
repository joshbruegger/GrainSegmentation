import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd

from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, split, unary_union
from shapely.prepared import prep
from shapely import shortest_line, set_precision

import pygeoops
from shapelysmooth import chaikin_smooth
import networkx as nx
from tqdm import tqdm

HARD_SNAP_TOL = 1e-6
CENTERLINE_SMOOTH_ITERS = 3


def _as_polygon_parts(geom) -> List[Polygon]:
    """Normalize Polygon/MultiPolygon/GeometryCollection to polygon parts."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts: List[Polygon] = []
        for part in geom.geoms:
            parts.extend(_as_polygon_parts(part))
        return parts
    return []


def _centerline(poly: Polygon) -> Optional[LineString]:
    # Simplify the polygon a bit to speed up pygeoops centerline computation
    poly_simp = poly.simplify(0.5, preserve_topology=True)
    if (
        poly_simp.is_empty
        or not hasattr(poly_simp, "geom_type")
        or "Polygon" not in poly_simp.geom_type
    ):
        poly_simp = poly

    # Use pygeoops to compute a centerline, extending it to boundary.
    line = pygeoops.centerline(poly_simp, extend=True)
    if line is None or line.is_empty:
        return None
    if line.geom_type == "MultiLineString":
        line = max(line.geoms, key=lambda g: g.length)
    if line.geom_type != "LineString":
        return None
    return line


def _smooth_centerline(line: LineString) -> LineString:
    if line.is_empty:
        return line
    smoothed = chaikin_smooth(line, CENTERLINE_SMOOTH_ITERS, keep_ends=True)
    if isinstance(smoothed, LineString):
        return smoothed
    if hasattr(smoothed, "geom_type") and smoothed.geom_type == "LineString":
        return smoothed
    return line


def _force_endpoints_to_intersection(line: LineString, intersection_geom) -> LineString:
    # Snap the line's endpoints onto the polygon-boundary intersection geometry.
    if line.is_empty:
        return line
    if intersection_geom is None or intersection_geom.is_empty:
        return line
    coords = list(line.coords)
    if len(coords) < 2:
        return line
    start = Point(coords[0])
    end = Point(coords[-1])
    snapped_start = nearest_points(start, intersection_geom)[1]
    snapped_end = nearest_points(end, intersection_geom)[1]
    if snapped_start.is_empty or snapped_end.is_empty:
        return line
    if snapped_start.equals(snapped_end):
        return line
    coords[0] = (snapped_start.x, snapped_start.y)
    coords[-1] = (snapped_end.x, snapped_end.y)

    return LineString(coords)


def _split_by_centerline(
    poly: Polygon, boundary_intersection
) -> Tuple[List[Polygon], Optional[LineString]]:
    # Split the overlap polygon using the centerline after endpoint snapping.
    line = _centerline(poly)
    if line is None:
        return [poly], None
    line = _force_endpoints_to_intersection(line, boundary_intersection)
    line = _smooth_centerline(line)
    # line = snap(line, boundary_intersection, HARD_SNAP_TOL)

    direct_parts = [p for p in split(poly, line).geoms if not p.is_empty]
    # direct_parts = _validate_split_pieces(direct_parts, poly)
    if len(direct_parts) >= 2:
        return direct_parts, line
    return [poly], line


def _assign_piece(pieces_list: List[Polygon], transect: LineString):
    intercepted_pieces = []
    for piece in pieces_list:
        # 1. Get the intersection between the piece's boundary and the transect
        boundary_intersection = piece.boundary.intersection(
            transect, grid_size=HARD_SNAP_TOL
        )

        if (
            boundary_intersection is None
            or boundary_intersection.is_empty
            or boundary_intersection.geom_type == "Point"
        ):
            continue

        if boundary_intersection.geom_type in (
            "MultiPoint",
            "GeometryCollection",
        ):
            num_intersections = sum(
                1 for geom in boundary_intersection.geoms if geom.geom_type == "Point"
            )
            if num_intersections < 2:
                continue

        intercepted_pieces.append(piece)

    if len(intercepted_pieces) == 0:
        return None
    return intercepted_pieces


def _assign_halves(
    pieces_list: List[Polygon],
    exclusive_a,
    exclusive_b,
    centerline: Optional[LineString],
) -> Optional[Tuple[List[Polygon], List[Polygon], List[Point], List[LineString]]]:

    if centerline is None or centerline.is_empty or len(pieces_list) < 2:
        return None

    midpoints = []
    transect_a = []
    transect_b = []

    # To avoid problems in which the centerline is not fully contained in the pieces_list,
    # we clip it to the pieces_list. We get a midpoint for each piece.
    clipped_centerline = centerline.intersection(unary_union(pieces_list))

    if clipped_centerline is None or clipped_centerline.is_empty:
        midpoints.append(centerline.interpolate(0.5, normalized=True))

    elif clipped_centerline.geom_type == "MultiLineString":
        midpoints.extend(
            line.interpolate(0.5, normalized=True) for line in clipped_centerline.geoms
        )
    elif clipped_centerline.geom_type == "LineString":
        midpoints.append(clipped_centerline.interpolate(0.5, normalized=True))
    elif clipped_centerline.geom_type == "GeometryCollection":
        midpoints.extend(
            geom.interpolate(0.5, normalized=True)
            for geom in clipped_centerline.geoms
            if geom is not None and not geom.is_empty and geom.geom_type == "LineString"
        )

    if len(midpoints) == 0:
        return None

    # For each midpoint, we get the transects by getting the shortest line
    # between the midpoint and the exclusive_a and exclusive_b boundaries
    # We extend the lines by a small amount to avoid problems with precision.
    for midpoint in midpoints:
        a = shortest_line(midpoint, exclusive_a.boundary)
        a = pygeoops.extend_line_by_distance(a, HARD_SNAP_TOL * 2, HARD_SNAP_TOL * 2)

        b = shortest_line(midpoint, exclusive_b.boundary)
        b = pygeoops.extend_line_by_distance(b, HARD_SNAP_TOL * 2, HARD_SNAP_TOL * 2)

        transect_a.append(a)
        transect_b.append(b)

    # Assign pieces to sides based on their intersection with the transects
    pieces_a = []
    pieces_b = []
    for transect in transect_a:
        hit_pieces = _assign_piece(pieces_list, transect)
        if hit_pieces is not None:
            pieces_a.extend(hit_pieces)
    for transect in transect_b:
        hit_pieces = _assign_piece(pieces_list, transect)
        if hit_pieces is not None:
            pieces_b.extend(hit_pieces)

    # Resolve conflicts between pieces assigned to both sides
    # Assign the piece to the side closest to its representative point
    for piece in pieces_list:
        in_a = any(piece is p for p in pieces_a)
        in_b = any(piece is p for p in pieces_b)
        if in_a and in_b:
            p_rep = piece.representative_point()
            if p_rep.distance(exclusive_a) < p_rep.distance(exclusive_b):
                pieces_b.remove(piece)
            else:
                pieces_a.remove(piece)

    # Simple heuristic to assign unassigned pieces to the other side
    # If everything fails, add piece to unassigned
    a_union = unary_union(pieces_a)
    b_union = unary_union(pieces_b)
    unassigned = []
    for piece in pieces_list:
        if piece not in pieces_a and piece not in pieces_b:
            if a_union.touches(piece) or a_union.overlaps(piece):
                pieces_b.append(piece)
            elif b_union.touches(piece) or b_union.overlaps(piece):
                pieces_a.append(piece)
            else:
                unassigned.append(piece)

    return pieces_a, pieces_b, unassigned, midpoints, transect_a + transect_b


def _split_overlap(
    poly_a: Polygon,
    poly_b: Polygon,
    min_area: float = 0.0,
    pair_i: Optional[int] = None,
    pair_j: Optional[int] = None,
) -> Tuple[
    Polygon,
    Polygon,
    bool,
    List[LineString],
    List[Polygon],
    List[dict],
    List[Point],
    List[LineString],
]:
    # Make sure the polygons are valid
    poly_a = pygeoops.make_valid(poly_a, only_if_invalid=True)
    poly_b = pygeoops.make_valid(poly_b, only_if_invalid=True)

    if getattr(poly_a, "geom_type", "") == "GeometryCollection":
        parts = _as_polygon_parts(poly_a)
        poly_a = unary_union(parts) if parts else poly_a

    if getattr(poly_b, "geom_type", "") == "GeometryCollection":
        parts = _as_polygon_parts(poly_b)
        poly_b = unary_union(parts) if parts else poly_b

    if poly_a is None or poly_a.is_empty or poly_b is None or poly_b.is_empty:
        print(f"Polygons {pair_i} and {pair_j} are invalid or empty, skipping")
        return poly_a, poly_b, False, [], [], [], [], []

    # Compute the overlap to be split and reassigned.
    overlap = poly_a.intersection(poly_b)
    if overlap.is_empty:
        return poly_a, poly_b, False, [], [], [], [], []

    overlap_parts = _as_polygon_parts(overlap)
    if not overlap_parts:
        return poly_a, poly_b, False, [], [], [], [], []

    min_area = max(float(min_area or 0.0), 0.0)
    smallest_is_a = poly_a.area <= poly_b.area
    small_overlap_only = min_area > 0.0 and all(
        part.area < min_area for part in overlap_parts
    )

    # Areas exclusive to each polygon are used to anchor assignments.
    exclusive_a = pygeoops.make_valid(poly_a.difference(overlap), only_if_invalid=True)
    exclusive_b = pygeoops.make_valid(poly_b.difference(overlap), only_if_invalid=True)
    context = ""
    if pair_i is not None and pair_j is not None:
        context = f" for polygon pair ({pair_i}, {pair_j})"
    if (
        exclusive_a is None
        or exclusive_b is None
        or exclusive_a.is_empty
        or exclusive_b.is_empty
    ):
        if not small_overlap_only:
            raise ValueError(
                "Exclusive geometry/boundry missing for overlap between polygons; "
                f"check for full containment or identical geometries{context}."
            )
        exclusive_a = None
        exclusive_b = None
    boundary_a = poly_a.boundary
    boundary_b = poly_b.boundary
    if boundary_a is None or boundary_b is None:
        print(f"DEBUG: poly_a={type(poly_a)}, boundary_a={type(boundary_a)}")
        print(f"DEBUG: poly_b={type(poly_b)}, boundary_b={type(boundary_b)}")
    boundary_intersection = boundary_a.intersection(boundary_b)

    a_parts: List[Polygon] = []
    b_parts: List[Polygon] = []
    centerlines: List[LineString] = []
    split_pieces: List[dict] = []
    midpoints: List[Point] = []
    transects: List[LineString] = []

    def _append_split_piece(overlap_id: int, assigned: str, geometry: Polygon) -> None:
        split_pieces.append(
            {"overlap_id": overlap_id, "assigned": assigned, "geometry": geometry}
        )

    for part_idx, part in enumerate(overlap_parts):
        if part.is_empty:
            continue
        if not part.is_valid:
            part = part.buffer(0)
        if part.is_empty:
            continue
        if min_area > 0.0 and part.area < min_area:
            if smallest_is_a:
                a_parts.append(part)
                _append_split_piece(part_idx, "A", part)
            else:
                b_parts.append(part)
                _append_split_piece(part_idx, "B", part)
            continue
        pieces, line = _split_by_centerline(part, boundary_intersection)
        if line is not None and not line.is_empty:
            centerlines.append(line)
        if len(pieces) < 2:
            for piece in pieces:
                _append_split_piece(part_idx, "U", piece)
            continue
        pieces_list = [piece for piece in pieces if not piece.is_empty]
        if not pieces_list:
            continue
        transect_assignment = _assign_halves(
            pieces_list, exclusive_a, exclusive_b, line
        )

        if transect_assignment is None:
            for piece in pieces_list:
                _append_split_piece(part_idx, "U", piece)
            continue

        assigned_a, assigned_b, unassigned, new_midpoints, new_transects = (
            transect_assignment
        )

        if new_midpoints is not None and len(new_midpoints) > 0:
            midpoints.extend(new_midpoints)
        if new_transects is not None and len(new_transects) > 0:
            transects.extend(new_transects)

        for piece in unassigned:
            _append_split_piece(part_idx, "U", piece)
        for piece in assigned_a:
            a_parts.append(piece)
            _append_split_piece(part_idx, "A", piece)
        for piece in assigned_b:
            b_parts.append(piece)
            _append_split_piece(part_idx, "B", piece)

    if not a_parts and not b_parts:
        return (
            poly_a,
            poly_b,
            False,
            centerlines,
            overlap_parts,
            split_pieces,
            midpoints,
            transects,
        )

    overlap_a = unary_union(a_parts) if a_parts else None
    overlap_b = unary_union(b_parts) if b_parts else None

    overlap_a = pygeoops.make_valid(overlap_a, only_if_invalid=True)
    overlap_b = pygeoops.make_valid(overlap_b, only_if_invalid=True)

    base_a = poly_a
    base_b = poly_b

    def _apply_assignment(src_a, src_b, keep_a, keep_b):
        new_a = src_a.difference(keep_b) if keep_b is not None else src_a
        new_b = src_b.difference(keep_a) if keep_a is not None else src_b

        new_a = pygeoops.make_valid(new_a, only_if_invalid=True)
        new_b = pygeoops.make_valid(new_b, only_if_invalid=True)

        new_a = set_precision(new_a, grid_size=HARD_SNAP_TOL)
        new_b = set_precision(new_b, grid_size=HARD_SNAP_TOL)
        return new_a, new_b

    poly_a, poly_b = _apply_assignment(base_a, base_b, overlap_a, overlap_b)

    return (
        poly_a,
        poly_b,
        True,
        centerlines,
        overlap_parts,
        split_pieces,
        midpoints,
        transects,
    )


def _build_candidates(geoms: List[Polygon], sindex) -> List[List[int]]:
    # Build per-geometry candidate indices with spatial index compatibility.
    candidates = [[] for _ in range(len(geoms))]
    if hasattr(sindex, "query_bulk"):
        pair_ix = sindex.query_bulk(geoms)
        for left, right in zip(pair_ix[0], pair_ix[1]):
            if right > left:
                candidates[left].append(int(right))
        return candidates
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        hits = None
        if hasattr(sindex, "query"):
            try:
                hits = sindex.query(geom, predicate="intersects")
            except TypeError:
                try:
                    hits = sindex.query(geom)
                except Exception:
                    hits = None
        if hits is None:
            try:
                hits = sindex.intersection(geom.bounds)
            except Exception:
                hits = []
        for j in hits:
            if int(j) > i:
                candidates[i].append(int(j))
    return candidates


def _build_overlap_components(
    geoms: List[Polygon], candidates: List[List[int]]
) -> List[List[int]]:
    """
    Build a deterministic list of connected components of overlapping polygons.
    Returns a list of components, where each component is a sorted list of polygon indices.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(geoms)))

    # Add edges only for actual intersections
    for i, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        prepared = prep(geom)
        for j in candidates[i]:
            if j <= i:
                continue
            other = geoms[j]
            if other is None or other.is_empty:
                continue
            if prepared.intersects(other):
                G.add_edge(i, j)

    # Extract components and sort them deterministically
    components = []
    for comp in nx.connected_components(G):
        sorted_comp = sorted(list(comp))
        components.append(sorted_comp)

    # Sort components by their first element to ensure deterministic order
    components.sort(key=lambda c: c[0])
    return components


def _process_component(
    component_indices: List[int],
    local_geoms: dict,
    comp_candidates: dict,
    min_area: float = 0.0,
) -> Tuple[
    List[Polygon], int, List[dict], List[dict], List[dict], List[dict], List[dict]
]:
    """
    Process a single connected component of overlapping polygons.
    Returns the updated geometries for the component, and debug records.
    """
    changed = 0
    centerline_records: List[dict] = []
    overlap_records: List[dict] = []
    split_records: List[dict] = []
    midpoint_records: List[dict] = []
    transect_records: List[dict] = []

    # Iterate through all pairs in the component
    for i_idx in range(len(component_indices)):
        i = component_indices[i_idx]
        geom = local_geoms[i]
        if geom is None or geom.is_empty:
            continue

        prepared = prep(geom)
        valid_js = sorted(comp_candidates.get(i, []))
        for j in valid_js:
            other = local_geoms[j]

            if other is None or other.is_empty:
                continue

            if not prepared.intersects(other):
                continue

            (
                new_i,
                new_j,
                did_change,
                centerlines,
                overlap_parts,
                split_pieces,
                midpoints,
                transects,
            ) = _split_overlap(
                geom,
                other,
                min_area=min_area,
                pair_i=i,
                pair_j=j,
            )

            if overlap_parts:
                for part_idx, part in enumerate(overlap_parts):
                    overlap_records.append(
                        {
                            "pair_i": i,
                            "pair_j": j,
                            "overlap_id": part_idx,
                            "geometry": part,
                        }
                    )
            if split_pieces:
                for record in split_pieces:
                    split_records.append(
                        {
                            "pair_i": i,
                            "pair_j": j,
                            "overlap_id": record["overlap_id"],
                            "assigned": record["assigned"],
                            "geometry": record["geometry"],
                        }
                    )
            if did_change:
                local_geoms[i] = new_i
                local_geoms[j] = new_j
                geom = new_i
                prepared = prep(geom)  # Update prepared geometry
                changed += 1

            for line in centerlines:
                centerline_records.append({"pair_i": i, "pair_j": j, "geometry": line})
            for pt in midpoints:
                midpoint_records.append({"pair_i": i, "pair_j": j, "geometry": pt})
            for tr in transects:
                transect_records.append({"pair_i": i, "pair_j": j, "geometry": tr})

    # Return the updated geometries in the same order as component_indices
    updated_geoms = [local_geoms[idx] for idx in component_indices]
    return (
        updated_geoms,
        changed,
        centerline_records,
        overlap_records,
        split_records,
        midpoint_records,
        transect_records,
    )


def _process_component_worker(
    args: Tuple[List[int], dict, dict, float, bool],
) -> Tuple[
    List[int],
    List[Polygon],
    int,
    List[dict],
    List[dict],
    List[dict],
    List[dict],
    List[dict],
]:
    """
    Worker function for processing a component in a separate process.
    Unpacks arguments and calls _process_component.
    """
    comp_indices, local_geoms, comp_candidates, min_area, debug_layers = args
    updated_geoms, changed, centerlines, overlaps, splits, midpoints, transects = (
        _process_component(comp_indices, local_geoms, comp_candidates, min_area)
    )

    if not debug_layers:
        centerlines = []
        overlaps = []
        splits = []
        midpoints = []
        transects = []

    return (
        comp_indices,
        updated_geoms,
        changed,
        centerlines,
        overlaps,
        splits,
        midpoints,
        transects,
    )


def resolve_overlaps(
    geoms: List[Polygon],
    min_area: float = 0.0,
    workers: int = 1,
    min_component_size_for_parallel: int = 10,
    debug_layers: bool = False,
) -> Tuple[
    List[Polygon], int, List[dict], List[dict], List[dict], List[dict], List[dict]
]:
    # Convert all geometries to a fixed precision grid to avoid topological errors
    geoms = set_precision(geoms, grid_size=HARD_SNAP_TOL, mode="valid_output").tolist()

    # Iterate through pairs with a spatial index to resolve overlaps in-place.
    changed = 0
    centerline_records: List[dict] = []
    overlap_records: List[dict] = []
    split_records: List[dict] = []
    midpoint_records: List[dict] = []
    transect_records: List[dict] = []
    sindex = gpd.GeoSeries(geoms).sindex
    candidates = _build_candidates(geoms, sindex)

    components = _build_overlap_components(geoms, candidates)

    # Filter out components with size < 2
    components = [comp for comp in components if len(comp) >= 2]

    if not components:
        return (
            geoms,
            changed,
            centerline_records,
            overlap_records,
            split_records,
            midpoint_records,
            transect_records,
        )

    if workers <= 1:
        # Sequential processing
        components_iter = tqdm(components, desc="Resolving overlaps sequentially")
        for comp_indices in components_iter:
            local_geoms = {idx: geoms[idx] for idx in comp_indices}
            comp_candidates = {
                idx: [j for j in candidates[idx] if j in local_geoms]
                for idx in comp_indices
            }
            (
                updated_geoms,
                comp_changed,
                comp_centerlines,
                comp_overlaps,
                comp_splits,
                comp_midpoints,
                comp_transects,
            ) = _process_component(
                comp_indices, local_geoms, comp_candidates, min_area=min_area
            )

            if comp_changed > 0:
                changed += comp_changed
                if debug_layers:
                    centerline_records.extend(comp_centerlines)
                    overlap_records.extend(comp_overlaps)
                    split_records.extend(comp_splits)
                    midpoint_records.extend(comp_midpoints)
                    transect_records.extend(comp_transects)

                # Update global geometries
                for idx, new_geom in zip(comp_indices, updated_geoms):
                    geoms[idx] = new_geom
    else:
        # Parallel processing
        large_components = []
        small_components = []

        for comp in components:
            if len(comp) >= min_component_size_for_parallel:
                large_components.append(comp)
            else:
                small_components.append(comp)

        # Process small components sequentially to avoid overhead
        if small_components:
            small_iter = tqdm(small_components, desc="Resolving small components")
            for comp_indices in small_iter:
                local_geoms = {idx: geoms[idx] for idx in comp_indices}
                comp_candidates = {
                    idx: [j for j in candidates[idx] if j in local_geoms]
                    for idx in comp_indices
                }
                (
                    updated_geoms,
                    comp_changed,
                    comp_centerlines,
                    comp_overlaps,
                    comp_splits,
                    comp_midpoints,
                    comp_transects,
                ) = _process_component(
                    comp_indices, local_geoms, comp_candidates, min_area=min_area
                )

                if comp_changed > 0:
                    changed += comp_changed
                    if debug_layers:
                        centerline_records.extend(comp_centerlines)
                        overlap_records.extend(comp_overlaps)
                        split_records.extend(comp_splits)
                        midpoint_records.extend(comp_midpoints)
                        transect_records.extend(comp_transects)

                    for idx, new_geom in zip(comp_indices, updated_geoms):
                        geoms[idx] = new_geom

        # Process large components in parallel
        if large_components:
            print(
                f"Processing {len(large_components)} large components with {workers} workers..."
            )
            # Prepare arguments for workers. We only send the required geometries to minimize pickling.
            worker_args = []
            for comp_indices in large_components:
                # We send a dict of exactly the required geometries
                local_geoms = {idx: geoms[idx] for idx in comp_indices}
                comp_candidates = {
                    idx: [j for j in candidates[idx] if j in local_geoms]
                    for idx in comp_indices
                }
                worker_args.append(
                    (comp_indices, local_geoms, comp_candidates, min_area, debug_layers)
                )

            results = []
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_component_worker, arg): i
                    for i, arg in enumerate(worker_args)
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Resolving large components",
                ):
                    results.append((futures[future], future.result()))

            # Sort results by original component order to ensure determinism
            results.sort(key=lambda x: x[0])

            for _, result in results:
                (
                    comp_indices,
                    updated_geoms,
                    comp_changed,
                    comp_centerlines,
                    comp_overlaps,
                    comp_splits,
                    comp_midpoints,
                    comp_transects,
                ) = result
                if comp_changed > 0:
                    changed += comp_changed
                    if debug_layers:
                        centerline_records.extend(comp_centerlines)
                        overlap_records.extend(comp_overlaps)
                        split_records.extend(comp_splits)
                        midpoint_records.extend(comp_midpoints)
                        transect_records.extend(comp_transects)

                    for idx, new_geom in zip(comp_indices, updated_geoms):
                        geoms[idx] = new_geom

    return (
        geoms,
        changed,
        centerline_records,
        overlap_records,
        split_records,
        midpoint_records,
        transect_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split overlapping polygons by centerline so overlaps become shared boundaries."
        )
    )
    parser.add_argument("--input", required=True, help="Input GeoPackage path")
    parser.add_argument("--output", required=True, help="Output GeoPackage path")
    parser.add_argument(
        "--layer",
        default=None,
        help="Input layer name (default: first layer)",
    )
    parser.add_argument(
        "--output-layer",
        default=None,
        help="Output layer name (default: input layer name)",
    )
    parser.add_argument(
        "--debug-layers",
        action="store_true",
        help="Write debug layers (centerlines, overlaps, split overlaps)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help=(
            "Minimum overlap area to split by centerline; smaller overlaps are "
            "assigned to the smaller polygon (default: 0.0)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel execution (default: 1).",
    )
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=10,
        help="Minimum component size to process in parallel (default: 10).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    gdf = gpd.read_file(input_path, layer=args.layer)
    if gdf.empty:
        raise SystemExit("No features found in input.")

    out_layer = args.output_layer or args.layer or input_path.stem

    geoms = list(gdf.geometry)

    (
        geoms,
        overlap_count,
        centerline_records,
        overlap_records,
        split_records,
        midpoint_records,
        transect_records,
    ) = resolve_overlaps(
        geoms,
        min_area=args.min_area,
        workers=args.workers,
        min_component_size_for_parallel=args.min_component_size,
        debug_layers=args.debug_layers,
    )

    out_gdf = gdf.copy()
    out_gdf.geometry = geoms

    # Filter out any that became fully empty/None
    out_gdf = out_gdf[out_gdf.geometry.notnull() & ~out_gdf.geometry.is_empty]

    # Explode multipolygons into separate features so the output is strictly Polygon
    out_gdf = out_gdf.explode(index_parts=False, ignore_index=True)

    # Filter again just to be sure we only have Polygons
    out_gdf = out_gdf[out_gdf.geometry.type == "Polygon"]

    out_gdf.to_file(output_path, layer=out_layer, driver="GPKG")

    def _write_debug_layer(records: List[dict], columns: List[str], layer: str) -> None:
        """Write a debug layer with a stable schema even if empty."""
        if records:
            gdf_layer = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
        else:
            gdf_layer = gpd.GeoDataFrame(
                columns=columns,
                geometry="geometry",
                crs=gdf.crs,
            )
        gdf_layer.to_file(output_path, layer=layer, driver="GPKG")

    if args.debug_layers:
        _write_debug_layer(
            centerline_records,
            ["pair_i", "pair_j", "geometry"],
            f"{out_layer}_centerlines",
        )
        _write_debug_layer(
            overlap_records,
            ["pair_i", "pair_j", "overlap_id", "geometry"],
            f"{out_layer}_overlaps",
        )
        _write_debug_layer(
            split_records,
            [
                "pair_i",
                "pair_j",
                "overlap_id",
                "assigned",
                "geometry",
            ],
            f"{out_layer}_split_overlaps",
        )
        _write_debug_layer(
            midpoint_records,
            ["pair_i", "pair_j", "geometry"],
            f"{out_layer}_midpoints",
        )
        _write_debug_layer(
            transect_records,
            ["pair_i", "pair_j", "geometry"],
            f"{out_layer}_transects",
        )

    print(
        f"Wrote {len(out_gdf)} features to {output_path} "
        f"(resolved {overlap_count} overlaps)."
    )


if __name__ == "__main__":
    main()
