import argparse
from pathlib import Path
from typing import List

import cv2
import geopandas as gpd
import numpy as np
from rasterio.features import shapes
import tifffile
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

Image.MAX_IMAGE_PIXELS = None


def _load_raster(path: str | Path) -> np.ndarray:
    raster_path = Path(path)
    suffix = raster_path.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        raster = tifffile.imread(raster_path)
    else:
        with Image.open(raster_path) as img:
            raster = np.array(img)

    if raster.ndim != 2:
        raise ValueError(f"Expected a 2D semantic raster, got shape {raster.shape}")

    return raster.astype(np.int32, copy=False)


def _binary_mask(mask: np.ndarray, class_value: int) -> np.ndarray:
    return (mask == class_value).astype(np.uint8)


def _component_masks(binary_mask: np.ndarray) -> List[np.ndarray]:
    binary = (binary_mask > 0).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(binary, connectivity=8)

    components: List[np.ndarray] = []
    for component_id in range(1, component_count):
        component = (labels == component_id).astype(np.uint8)
        if component.any():
            components.append(component)
    return components


def _component_to_polygon(component_mask: np.ndarray, flip_y: bool = True):
    if not component_mask.any():
        return Polygon()

    polygons = []
    for geom_mapping, value in shapes(
        component_mask.astype(np.uint8),
        mask=component_mask.astype(bool),
        connectivity=8,
    ):
        if int(value) != 1:
            continue
        polygon = shape(geom_mapping)
        if polygon.is_empty:
            continue
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        polygons.append(polygon)

    if not polygons:
        return Polygon()

    merged = polygons[0] if len(polygons) == 1 else unary_union(polygons)
    if flip_y:
        merged = affinity.scale(merged, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
    return merged


def _build_gdf(polygons: List[dict], class_value: int) -> gpd.GeoDataFrame:
    if not polygons:
        return gpd.GeoDataFrame(
            columns=["component_id", "class_value", "pixel_area", "geometry"],
            geometry="geometry",
        )

    gdf = gpd.GeoDataFrame(polygons, geometry="geometry")

    gdf["class_value"] = int(class_value)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["component_id", "class_value", "pixel_area", "geometry"],
            geometry="geometry",
        )

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.type == "Polygon"]
    return gdf


def raster_to_gdf(
    raster: np.ndarray,
    class_value: int = 1,
    min_area: int = 0,
    flip_y: bool = True,
) -> gpd.GeoDataFrame:
    binary = _binary_mask(raster, class_value=class_value)
    components = _component_masks(binary)

    records = []
    for component_id, component in enumerate(components, start=1):
        pixel_area = int(component.sum())
        if pixel_area < min_area:
            continue

        geometry = _component_to_polygon(component, flip_y=flip_y)
        if geometry.is_empty:
            continue

        records.append(
            {
                "component_id": component_id,
                "class_value": int(class_value),
                "pixel_area": pixel_area,
                "geometry": geometry,
            }
        )

    return _build_gdf(records, class_value=class_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a semantic prediction raster into polygon features."
    )
    parser.add_argument("-i", "--input", required=True, help="Input semantic raster")
    parser.add_argument(
        "-o", "--output", required=True, help="Output GeoPackage path (.gpkg)"
    )
    parser.add_argument(
        "--output-layer",
        default=None,
        help="Output layer name (default: output file stem)",
    )
    parser.add_argument(
        "--class-value",
        type=int,
        default=1,
        help="Semantic class value to polygonize (default: 1)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="Minimum connected-component area in pixels (default: 0)",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Do not flip Y coordinates into the repo's negative-Y convention",
    )
    args = parser.parse_args()

    raster = _load_raster(args.input)
    gdf = raster_to_gdf(
        raster,
        class_value=args.class_value,
        min_area=args.min_area,
        flip_y=not args.no_flip_y,
    )

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".gpkg":
        output_path = output_path.with_suffix(".gpkg")

    layer_name = args.output_layer or output_path.stem
    gdf.to_file(output_path, layer=layer_name, driver="GPKG")
    print(f"Wrote {len(gdf)} polygons to {output_path}")


if __name__ == "__main__":
    main()
