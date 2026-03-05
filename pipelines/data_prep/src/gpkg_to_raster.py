import argparse
import numpy as np
import cv2
import geopandas as gpd
from PIL import Image
import tifffile
import sys
import os

Image.MAX_IMAGE_PIXELS = None


def process_polygons(gdf, height, width, boundary_width, flip_y):
    mask = np.zeros((height, width), dtype=np.uint8)

    # We need to extract outer rings and inner rings (holes)
    def _get_rings(polygon):
        rings = [list(polygon.exterior.coords)]
        for interior in polygon.interiors:
            rings.append(list(interior.coords))
        return rings

    def _draw_poly(m, poly, val):
        if poly.geom_type == "Polygon":
            rings = _get_rings(poly)
            pts = [np.array(r, dtype=np.int32) for r in rings]
            if flip_y:
                for p in pts:
                    p[:, 1] = -p[:, 1]
            cv2.fillPoly(m, [pts[0]], val)
            for hole in pts[1:]:
                cv2.fillPoly(m, [hole], 0)  # clear holes
        elif poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                _draw_poly(m, p, val)

    # 1. Rasterize full polygons as boundary (class 2)
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        _draw_poly(mask, geom, 2)

    # 2. Rasterize inner polygons as interior (class 1)
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue

        inner_geom = geom.buffer(-boundary_width)
        if not inner_geom.is_empty:
            _draw_poly(mask, inner_geom, 1)

    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPKG polygons to 3-class raster masks."
    )
    parser.add_argument("-i", "--input", required=True, help="Input GPKG file")
    parser.add_argument(
        "-r", "--reference", required=True, help="Reference TIFF file for dimensions"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output raster file path (PNG or TIFF)"
    )
    parser.add_argument(
        "--boundary-width",
        type=float,
        default=3.0,
        help="Width of the interior boundary in pixels",
    )
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Do not flip Y coordinates if they are negative",
    )

    args = parser.parse_args()

    # Read reference image to get dimensions
    try:
        with Image.open(args.reference) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error reading reference image: {e}")
        sys.exit(1)

    # Read GPKG
    try:
        gdf = gpd.read_file(args.input)
    except Exception as e:
        print(f"Error reading GPKG file: {e}")
        sys.exit(1)

    # Process
    mask = process_polygons(gdf, height, width, args.boundary_width, not args.no_flip_y)

    # Save output
    output_path = args.output
    if not (
        output_path.lower().endswith(".tif") or output_path.lower().endswith(".tiff")
    ):
        output_path = os.path.splitext(output_path)[0] + ".tif"

    try:
        tifffile.imwrite(output_path, mask, compression="deflate")
        print(f"Saved mask to {output_path}")
    except Exception as e:
        print(f"Error saving output image: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
