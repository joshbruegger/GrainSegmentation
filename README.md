[Hippocratic License HL3-FULL](https://firstdonoharm.dev/version/3/0/full.html)

# GrainSegmentation

**Objective:** Compare instance grain segmentation performance across four microscopy input variants using both U-Net and YOLO segmentation pipelines.

The four input variants are:

- `PPL`
- `PPL+AllPPX`
- `PPL+PPXblend`
- `PPLPPXblend`

For each variant, the project aims to:

1. Tune YOLO hyperparameters.
2. Tune U-Net hyperparameters.
3. Train the final YOLO model.
4. Train the final U-Net model.
5. Tune watershed postprocessing for the U-Net model.
6. Compare connected-components and watershed-based instance extraction for U-Net outputs and choose the better postprocessing strategy.
7. Evaluate YOLO and U-Net instance segmentation on a held-out test image.
8. Compare results across all models and variants.

## Dataset

The dataset consists of one partially labelled high-resolution sandstone thin-section captured as aligned microscopy images:

- 1 Plane-Polarized Light image (`PPL`)
- 6 Cross-Polarized Light images (`PPX1` to `PPX6`)

Labels were obtained by running SAM2 on the PPL image using a sliding-window approach and then manually refining the result in QGIS. The final labels were quality-checked by fixing invalid geometries, removing holes and fully-contained features, smoothing, and finally buffering polygon boundaries by 5px.

Because the manually refined polygons often contained overlaps, especially after smoothing and buffering, an automated preprocessing step (`split_overlaps.py`) was developed to produce strictly non-overlapping grain masks where previously-overlapping polygons touch exactly at a shared boundary.

The algorithm resolves overlaps by:
1. Identifying connected components of overlapping polygons using a spatial index and bounding-box intersection graph.
2. For each overlapping pair, computing the exact intersection polygon.
3. Calculate a topological centerline that splits the overlap in two halves using voronoi polygons (handled by `pygeoops`).
4. Smoothing the centerline (using Taubin and Chaikin smoothing algorithms) to remove Voronoi-originated zigzags.
5. Snapping the centerline endpoints exactly to the outer boundaries of the intersecting grains.
6. Splitting the overlap along this smoothed centerline
7. Assigning the resulting halves to the original adjacent polygons based on a ray-cast heuristic by building lines from the midpoint of the centerline to the exclusive areas of the polygons. 

These refined polygons are then rasterized into three semantic classes:

- background
- grain interior
- grain boundary

The grain boundary class is represented as a 3 pixel-wide explicit boundary band so that both semantic and instance-separation quality can be evaluated.

Two composite variants are also derived:

- `PPLPPXblend`: a single composite input of all images using screen blending
- `PPL+PPXblend`: a two-input variant using `PPL` plus a screen-blended PPX composite

### Train/Validation Split and Patch Extraction

To process the large high-resolution thin-section image, the training data is spatially split and patchified:

1. **Spatial Tiling:** The image is divided into large 4096×4096 spatial tiles. 
2. **Coverage Stratification:** To ensure balanced sets, the grain coverage (percentage of grain pixels) is computed for each tile. Tiles with less than 10% coverage are assigned strictly to the training set. The remaining eligible tiles are binned by coverage and split using stratified sampling: 80% to the training set and 20% to the validation set.
3. **Patch Extraction:** The tiles are then densely cropped into 1024×1024 patches with a 50% overlap. For each time, the corresponding polygon annotation is also saved.


## Research Pipeline

### 1. Input Variants

The project compares four input configurations:

- `PPL`: single-input baseline
- `PPLPPXblend`: single blended composite input
- `PPL+PPXblend`: two-input PPL + PPX-blend configuration
- `PPL+AllPPX`: seven-input configuration using PPL and all PPX images

### 2. Model Families

Two segmentation model families are evaluated:

- **U-Net** for semantic segmentation followed by instance extraction
- **YOLO segmentation** for direct instance segmentation

### 3. U-Net Workflow

For each input variant, the U-Net workflow is:

1. Tune model hyperparameters to achieve best validaton on the training dataset.
2. Train the final U-Net model using the selected hyperparameters.
3. Generate semantic predictions on the training dataset.
4. Convert semantic predictions to instances using:

   - connected components (`CC`)
   - watershed

5. Tune watershed hyperparameters for each U-Net variant to achieve best aji on training image.
6. Compare `CC` versus tuned `watershed` on the **training dataset** to decide which postprocessing strategy should be used for final U-Net evaluation.
7. Evaluate the selected U-Net pipeline on the held-out test image.

### 4. YOLO Workflow

For each input variant, the YOLO workflow is:

1. Tune YOLO hyperparameters to achieve best validaton on the training dataset.
2. Train the final YOLO segmentation model.
3. Evaluate instance segmentation performance on the held-out test image.

### 5. Metrics and Comparison

Evaluation metrics include:

- **AJI (Aggregated Jaccard Index):** An instance-aware metric specifically designed for microscopy and cell segmentation. AJI directly penalizes under-segmentation (merged grains) and over-segmentation (split grains) at the pixel level. It provides a holistic view of both detection and boundary adherence without relying on confidence thresholds.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. It indicates how many of the segmented grains are actually grains.
- **Recall:** The ratio of correctly predicted positive observations to all observations in actual class. It measures how many of the actual grains were successfully segmented.
- **F1 Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both false positives and false negatives.

For the YOLO models, **COCO-style mask AP (Average Precision)** was also used. It decouples detection performance from spatial accuracy by averaging across multiple IoU thresholds (AP@0.5:0.95). It allows for the creation of precision-recall curves to understand model confidence. It is generally less sensitive to the topological structure of boundaries (like fused touching instances) than AJI. AP was not calculated for U-net models as they don't produce the required confidence scores for each prediction.


## Interpretation Notes

Because the final test set is a single held-out image, final evaluation should be interpreted descriptively rather than inferentially. The held-out result is useful for practical comparison and model selection, but it does not support strong statistical claims about generalization.

## Paper Structure

- **Introduction & motivation:** grain segmentation in sandstone thin sections and why multi-modal microscopy may help
- **Related work:** thin-section segmentation and multi-modal microscopy fusion
- **Dataset & labeling:** imaging setup, annotation workflow, overlap removal, raster-mask generation
- **Input variants:** PPL baseline, PPX composites, and multi-input variants
- **Methods:** U-Net pipeline, YOLO pipeline, and U-Net postprocessing with CC and watershed
- **Experiments:** per-variant tuning, training, postprocessing selection on the training data, and held-out evaluation
- **Results:** quantitative comparison across models and variants, plus qualitative overlays
- **Discussion & limitations:** single-thin-section dataset, descriptive held-out testing, and model failure modes
- **Conclusion & future work**
