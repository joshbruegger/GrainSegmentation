[Hippocratic License HL3-FULL](https://firstdonoharm.dev/version/3/0/full.html)

# GrainSegmentation

**Objective:** Segment grains in sandstone thin-section microscopy comparing three U-net model inputs: PPL-only baseline, PPL + all PPX images, and PPL + screen-blended PPX composite, single PPL+PPX composite.

## Dataset

The dataset consists of one partially-labelled high-resolution thin-section divided in 7 aligned images, 1 using Plane-Polarized Light (PPL) and 6 using Cross-Polarized Light (PPX) at different angles.

Labels were obtained by running SAM2 on the PPL image using a sliding-window approach and then manually perfecting the labels using QGIS. The final hand-labelled masks were then quality checked by fixing invalid geometries, removing holes and fully-contained features and were then smoothed and buffered by 5 pixels. As the resulting labels contained overlaps, a script was developed to produce non-overlapping grain masks by splitting the overlap at its centre so that previously-overlapping polygons touch along it. Polygon masks were then converted into raster masks with the classes background, grain interior, and grain boundary; the grain boundary were computed to be 3-pixel wide.

For the PPL + screen-blended PPX model, the 6 PPX images were combined into a single 3-channel composite using screen blending (per-pixel `1 - Π(1 - I_k)` on normalized `[0,1]` channels). The same process, but for all images was used to obtained the PPL_PPX screen-blend composite.

### Modeling & Training Pipeline

- **Baseline U-Net:** A consistent U-Net backbone is used across all models for fair comparisons.
- **Multi-input Data Loader:** Supports flexible input channels (`num_inputs` ∈ {1, 2, 7}) via the wrapper script
- **Online Random Patches:** Random patch sampling per epoch (e.g., sample N random coordinates per image/region instead of deterministic stride) handles large training images efficiently.

### 3. Cross-Validation & Hyperparameter Optimization

- **Spatial K-fold:** Splits the large training image using grid regions and grain-coverage stratification to ensure similar grain pixels per fold.
- **Bayesian Optimization Loop:** A training runner evaluates candidate hyperparameter sets (base filters, learning rate, batch size, and dropout rate) across K folds to maximize mean validation metrics.

### 4. Evaluation

- **Metrics:** Recommended evaluation metrics for this project are:
    - **Boundary IoU:** Best primary boundary metric for this task because the labels include an explicit thin grain-boundary class and boundary quality is central to downstream grain separation.
    - **Boundary F1:** Useful companion metric with a pixel tolerance to summarize how often predicted boundaries fall close to the annotated boundaries.
    - **Interior IoU / Dice:** Still needed to measure overall grain-region overlap, since strong boundary scores alone do not guarantee good interior coverage.
    - **Aggregated Jaccard Index (AJI):** A good instance-aware metric for microscopy-style segmentation because it penalizes grain merges and splits more directly than semantic overlap metrics.
    - **Optional diagnostics:** Boundary precision/recall, Adapted Rand error, and Variation of Information can help separate false-split and false-merge failure modes when comparing models.

    Pixel accuracy should be treated as secondary because the background class can dominate it. Panoptic Quality (PQ) is also lower priority for this project because microscopy literature has shown that IoU-thresholded panoptic metrics can be hard to interpret for small, boundary-sensitive objects.
- **Final Test:** Models are trained with tuned hyperparameters on the full training image and evaluated once on a held-out section of the thin-section to measure generalization.
- **Ablations:** Comparisons between PPL-only, PPL+PPX blend, PPL + all-PPX, and PPL + screen-blend highlight differences in boundary quality and grain separation.



## Paper Structure

- **Introduction & motivation:** Why grain boundary segmentation is useful, previous research using PPL, how the addition of PPX can improve grain boundary segmentation.
- **Related work:** Thin-section segmentation and multi-modal microscopy fusion.
- **Dataset & labeling:** Imaging setup, labeling protocol, de-overlap processing, train/val/test split design.
- **Method:** U-Net architecture, input configurations, screen-blend composite, online patch sampling.
- **Experiments:** Spatial K-fold protocol, Bayesian hyperparameter search, metrics.
- **Results:** Quantitative tables/plots + qualitative overlays.
- **Discussion & limitations:** Generalization to new thin-sections, failure modes.
- **Conclusion & future work.**

