import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, snap, split, unary_union
from shapely.prepared import prep
from shapely import shortest_line

import pygeoops
from shapelysmooth import chaikin_smooth
from tqdm import tqdm

HARD_SNAP_TOL = 1e-6
EPS = 1e-9
CENTERLINE_SMOOTH_ITERS = 5


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


def _has_holes(geom) -> bool:
    """Return True if any polygon part has interior rings."""
    if geom is None or geom.is_empty:
        return False
    return any(poly.interiors for poly in _as_polygon_parts(geom))


def _centerline(poly: Polygon) -> Optional[LineString]:
    # Use pygeoops to compute a centerline, extending it to boundary.
    line = pygeoops.centerline(poly, extend=True)
    if line.is_empty:
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


def _auto_snap_centerline(
    line: LineString, poly: Polygon, base_tolerance: float
) -> Tuple[LineString, float]:
    # Compute a safe snap tolerance from endpoint-to-boundary distances.
    if line.is_empty:
        return line, base_tolerance
    coords = list(line.coords)
    if len(coords) < 2:
        return line, base_tolerance
    boundary = poly.boundary
    start = Point(coords[0])
    end = Point(coords[-1])
    dist_start = start.distance(boundary)
    dist_end = end.distance(boundary)
    needed = max(dist_start, dist_end)
    tol = max(base_tolerance, needed) * 1.001
    if tol > 0:
        line = snap(line, boundary, tol)
    return line, tol


def _dedupe_polygons(pieces: List[Polygon], tol: float = 1e-8) -> List[Polygon]:
    """Remove near-duplicate polygon parts by geometric equality."""
    unique: List[Polygon] = []
    for piece in pieces:
        is_dup = False
        for existing in unique:
            if piece.equals_exact(existing, tol):
                is_dup = True
                break
        if not is_dup:
            unique.append(piece)
    return unique


def _validate_split_pieces(pieces: List[Polygon], original: Polygon) -> List[Polygon]:
    # Filter/repair invalid or empty parts from the split.
    cleaned: List[Polygon] = []
    for piece in pieces:
        if piece.is_empty or piece.area <= 0:
            continue
        if not piece.is_valid:
            piece = piece.buffer(0)
        if not piece.is_empty and piece.area > 0:
            cleaned.append(piece)
    cleaned = _dedupe_polygons(cleaned)
    if len(cleaned) < 2:
        return cleaned
    union = unary_union(cleaned)
    if union.is_empty:
        return [original]
    sum_area = sum(p.area for p in cleaned)
    if union.area > 0 and sum_area > union.area * 1.01:
        return [original]
    return cleaned


def _split_by_centerline(
    poly: Polygon, boundary_intersection, snap_tolerance: float
) -> Tuple[List[Polygon], Optional[LineString]]:
    # Split the overlap polygon using the centerline after endpoint snapping.
    line = _centerline(poly)
    if line is None:
        return [poly], None
    line = _smooth_centerline(line)
    line = _force_endpoints_to_intersection(line, boundary_intersection)
    split_line, _ = _auto_snap_centerline(
        line, poly, max(snap_tolerance, HARD_SNAP_TOL)
    )
    line = split_line

    direct_parts = [p for p in split(poly, split_line).geoms if not p.is_empty]
    direct_parts = _validate_split_pieces(direct_parts, poly)
    if len(direct_parts) >= 2:
        return direct_parts, line
    return [poly], line


def _shared_edge_length(
    piece: Polygon, exclusive_geom, exclusive_boundary=None
) -> float:
    # Shared edge length between split piece and exclusive area boundary.
    if exclusive_boundary is None:
        if exclusive_geom is None or exclusive_geom.is_empty:
            return 0.0
        exclusive_boundary = exclusive_geom.boundary
    if exclusive_boundary is None or exclusive_boundary.is_empty:
        return 0.0
    return piece.boundary.intersection(exclusive_boundary).length


def _midpoint_perpendicular(
    line: Optional[LineString],
) -> Optional[Tuple[Point, Tuple[float, float]]]:
    # Compute a unit perpendicular vector at the centerline midpoint by
    # estimating the local tangent direction around the midpoint.
    if line is None or line.is_empty:
        return None
    length = line.length
    if length <= 0:
        return None
    midpoint = line.interpolate(0.5, normalized=True)
    mid_dist = line.project(midpoint)
    # Sample two nearby points around the midpoint to estimate direction.
    delta = min(max(length * 0.01, length * 1e-6), length / 2)
    p0 = line.interpolate(max(mid_dist - delta, 0.0))
    p1 = line.interpolate(min(mid_dist + delta, length))
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    # Fall back to overall line direction if the local segment is degenerate.
    if abs(dx) + abs(dy) <= 0:
        coords = list(line.coords)
        if len(coords) >= 2:
            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
    norm = math.hypot(dx, dy)
    if norm <= 0:
        return None
    # Rotate by 90 degrees to get perpendicular direction.
    perp = (-dy / norm, dx / norm)
    return midpoint, perp


def _ray_intersection_distance(
    midpoint: Point,
    ray: LineString,
    boundary,
) -> Optional[float]:
    # Use the provided ray and see where it first hits a boundary.
    if boundary is None or boundary.is_empty:
        return None
    if not ray.intersects(boundary):
        return None
    intersection = ray.intersection(boundary)
    if intersection.is_empty:
        return None
    # Pick the closest intersection point to the midpoint.
    hit = nearest_points(midpoint, intersection)[1]
    dist = midpoint.distance(hit)
    # Ignore near-zero distances to avoid self-hits at the midpoint.
    if dist <= HARD_SNAP_TOL:
        return None
    return dist


def _assign_halves_by_transect(
    pieces_list: List[Polygon],
    exclusive_a,
    exclusive_b,
    centerline: Optional[LineString],
) -> Optional[Tuple[List[Polygon], List[Polygon]]]:

    if centerline is None or centerline.is_empty or len(pieces_list) < 2:
        return None
    midpoint = centerline.interpolate(0.5, normalized=True)
    boundary_a = exclusive_a.boundary
    boundary_b = exclusive_b.boundary

    # shortest line to boundary a from midpoint
    transect = shortest_line(midpoint, boundary_a)
    if transect is None or transect.is_empty:
        transect = shortest_line(midpoint, boundary_b)

    # Choose the split piece most intersected by the transect.
    best_length = 0.0
    best_index = None
    for idx, piece in enumerate(pieces_list):
        if not piece.intersects(transect):
            length = 0.0
        else:
            intersection = piece.intersection(transect)
            length = intersection.length if not intersection.is_empty else 0.0
        if length > best_length + EPS:
            best_length = length
            best_index = idx

    if best_index is None or best_length <= 0:
        return None
    hit_piece = pieces_list[best_index]
    if transect.intersects(boundary_a):
        pieces_a = [hit_piece]
        pieces_b = [p for idx, p in enumerate(pieces_list) if idx != best_index]
    else:
        pieces_b = [hit_piece]
        pieces_a = [p for idx, p in enumerate(pieces_list) if idx != best_index]
    return pieces_a, pieces_b


def _split_overlap(
    poly_a: Polygon,
    poly_b: Polygon,
    snap_tolerance: float,
    pair_i: Optional[int] = None,
    pair_j: Optional[int] = None,
) -> Tuple[
    Polygon,
    Polygon,
    bool,
    List[LineString],
    List[Polygon],
    List[dict],
]:
    # Compute the overlap to be split and reassigned.
    overlap = poly_a.intersection(poly_b)
    if overlap.is_empty:
        return poly_a, poly_b, False, [], [], []

    overlap_parts = _as_polygon_parts(overlap)
    if not overlap_parts:
        return poly_a, poly_b, False, [], [], []

    # Areas exclusive to each polygon are used to anchor assignments.
    exclusive_a = poly_a.difference(overlap)
    exclusive_b = poly_b.difference(overlap)
    context = ""
    if pair_i is not None and pair_j is not None:
        context = f" for polygon pair ({pair_i}, {pair_j})"
    if (
        exclusive_a is None
        or exclusive_b is None
        or exclusive_a.is_empty
        or exclusive_b.is_empty
    ):
        raise ValueError(
            "Exclusive geometry/boundry missing for overlap between polygons; "
            f"check for full containment or identical geometries{context}."
        )
    boundary_a = poly_a.boundary
    boundary_b = poly_b.boundary
    boundary_intersection = boundary_a.intersection(boundary_b)

    a_parts: List[Polygon] = []
    b_parts: List[Polygon] = []
    centerlines: List[LineString] = []
    split_pieces: List[dict] = []

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
        pieces, line = _split_by_centerline(part, boundary_intersection, snap_tolerance)
        if line is not None and not line.is_empty:
            centerlines.append(line)
        if len(pieces) < 2:
            for piece in pieces:
                _append_split_piece(part_idx, "U", piece)
            continue
        pieces_list = [piece for piece in pieces if not piece.is_empty]
        if not pieces_list:
            continue
        transect_assignment = _assign_halves_by_transect(
            pieces_list, exclusive_a, exclusive_b, line
        )
        if transect_assignment is None:
            for piece in pieces_list:
                _append_split_piece(part_idx, "U", piece)
            continue
        assigned_a, assigned_b = transect_assignment
        for piece in assigned_a:
            a_parts.append(piece)
            _append_split_piece(part_idx, "A", piece)
        for piece in assigned_b:
            b_parts.append(piece)
            _append_split_piece(part_idx, "B", piece)

    if not a_parts and not b_parts:
        return poly_a, poly_b, False, centerlines, overlap_parts, split_pieces

    overlap_a = unary_union(a_parts) if a_parts else None
    overlap_b = unary_union(b_parts) if b_parts else None

    # Apply assignment; if holes appear, swap assignment once.
    base_a = poly_a
    base_b = poly_b

    def _apply_assignment(
        src_a: Polygon,
        src_b: Polygon,
        keep_a,
        keep_b,
    ) -> Tuple[Polygon, Polygon]:
        new_a = src_a.difference(keep_b) if keep_b is not None else src_a
        new_b = src_b.difference(keep_a) if keep_a is not None else src_b
        return new_a.buffer(0), new_b.buffer(0)

    poly_a, poly_b = _apply_assignment(base_a, base_b, overlap_a, overlap_b)
    if _has_holes(poly_a) or _has_holes(poly_b):
        swapped_a, swapped_b = _apply_assignment(base_a, base_b, overlap_b, overlap_a)
        poly_a, poly_b = swapped_a, swapped_b
        swap = {"A": "B", "B": "A"}
        split_pieces = [
            {**record, "assigned": swap.get(record["assigned"], record["assigned"])}
            for record in split_pieces
        ]

    return poly_a, poly_b, True, centerlines, overlap_parts, split_pieces


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


def resolve_overlaps(
    geoms: List[Polygon],
    snap_tolerance: float,
) -> Tuple[List[Polygon], int, List[dict], List[dict], List[dict]]:
    # Iterate through pairs with a spatial index to resolve overlaps in-place.
    changed = 0
    centerline_records: List[dict] = []
    overlap_records: List[dict] = []
    split_records: List[dict] = []
    sindex = gpd.GeoSeries(geoms).sindex
    candidates = _build_candidates(geoms, sindex)
    indices = range(len(geoms))
    indices = tqdm(indices, total=len(geoms), desc="Resolving overlaps")
    for i in indices:
        geom = geoms[i]
        if geom is None or geom.is_empty:
            continue
        prepared = prep(geom)
        candidate_js = candidates[i]
        for j in candidate_js:
            if j <= i:
                continue
            other = geoms[j]
            if other is None or other.is_empty:
                continue
            if not prepared.intersects(other):
                continue
            new_i, new_j, did_change, centerlines, overlap_parts, split_pieces = (
                _split_overlap(
                    geom,
                    other,
                    snap_tolerance=snap_tolerance,
                    pair_i=i,
                    pair_j=j,
                )
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
                geoms[i] = new_i
                geoms[j] = new_j
                geom = new_i
                changed += 1
                for line in centerlines:
                    centerline_records.append(
                        {"pair_i": i, "pair_j": j, "geometry": line}
                    )
    return geoms, changed, centerline_records, overlap_records, split_records


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
        "--snap-tolerance",
        type=float,
        default=0.0,
        help="Snap centerline to polygons by this tolerance",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    gdf = gpd.read_file(input_path, layer=args.layer)
    if gdf.empty:
        raise SystemExit("No features found in input.")

    # Fail early if the input already contains holes.
    print("Checking for holes in input polygons...")
    hole_indices: List[int] = []
    for idx, geom in gdf.geometry.items():
        if _has_holes(geom):
            hole_indices.append(int(idx))
    if hole_indices:
        sample = ", ".join(str(x) for x in hole_indices[:10])
        raise SystemExit(
            "Input polygons contain holes; fix before processing. "
            f"Found {len(hole_indices)} features (sample indices: {sample})."
        )

    out_layer = args.output_layer or args.layer or input_path.stem

    geoms = list(gdf.geometry)

    geoms, overlap_count, centerline_records, overlap_records, split_records = (
        resolve_overlaps(geoms, snap_tolerance=args.snap_tolerance)
    )

    out_gdf = gdf.copy()
    out_gdf.geometry = geoms
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

    print(
        f"Wrote {len(out_gdf)} features to {output_path} "
        f"(resolved {overlap_count} overlaps)."
    )


if __name__ == "__main__":
    main()
