# Codebase Index:

This document provides a detailed overview of the source code files in the **Behavior-Aware Trajectory Prediction System**. The codebase is modular, with distinct components for data handling, lane extraction, prediction logic, and visualization.

## 1. Core System:

### `main.py`:
The main entry point for the application. It orchestrates the entire pipeline by:
*   Initializing all subsystems (Data Loader, Lane Manager, Predictor, Visualizer).
*   Loading the video and tracking data.
*   Running the main processing loop frame-by-frame.
*   Generating the final output video with visualizations.

### `config.py`:
Central configuration file containing all system parameters and constants. It includes:
*   **File Paths:** Locations for video, tracking data, and output directories.
*   **Prediction Parameters:** Settings for history window (4s) and prediction horizon (2s).
*   **Algorithm Thresholds:** Parameters for social potential fields, game theory weights, and lane extraction tolerances.
*   **Visualization Settings:** Colors, alpha values, and drawing options.

## 2. Data Management:

### `track_data_loader.py`:
Responsible for ingesting and organizing vehicle tracking data.
*   **Data Loading:** Parses CSV or JSON files containing pre-computed vehicle detections.
*   **Track Management:** Organizes detections by `track_id` and `frame_id` for efficient retrieval.
*   **History Retrieval:** Provides methods to fetch the past trajectory of any vehicle at a given frame.

### `road_mask_manager.py`:
Manages the definition and representation of the drivable road area.
*   **Polygon Loading:** Reads road boundary annotations from JSON files.
*   **Rasterization:** Converts vector polygons into a continuous raster mask (Traffic Flow Potential Surface).
*   **Lane Embedding:** Embeds extracted emerging lanes into the road mask using Gaussian splatting to create flow corridors.
*   **Constraint Checking:** Provides methods to check if a point or path lies within the drivable area.

## 3. Lane Analysis:

### `emerging_lane_extractor.py`:
Implements the logic for "learning" lanes from historical traffic data.
*   **Trajectory Clustering:** Groups similar vehicle tracks using Hausdorff distance.
*   **Lane Smoothing:** Applies Savitzky-Golay filtering and B-Spline fitting to create smooth lane centerlines.
*   **Confidence Scoring:** Assigns scores to lanes based on the volume of supporting traffic.
*   **GeoJSON Export:** Saves the extracted lanes and connections to a standard geospatial format.

### `lane_manager.py`:
Manages the collection of extracted lanes during runtime.
*   **Spatial Indexing:** Allows for efficient querying of lanes near a specific point.
*   **Path Planning:** Provides utilities to generate candidate paths that follow the geometry of the nearest lanes.
*   **Connectivity:** Manages connections between lane segments to allow for longer-range path planning.

## 4. Prediction Logic:

### `trajectory_predictor.py`:
The core engine that generates future trajectories for vehicles.
*   **Candidate Generation:** Creates a diverse set of potential paths using kinematic models and lane geometry.
*   **Scoring:** Evaluates each candidate path based on a multi-objective function (Flow alignment, Social safety, Kinematic feasibility).
*   **Selection:** Selects the top-k most likely trajectories for the final output.

### `social_potential_field.py`:
Implements the "Social Force" model for collision avoidance.
*   **Potential Calculation:** Computes a repulsive scalar field around each vehicle.
*   **Risk Assessment:** Calculates the collision risk of a candidate path by integrating the potential field along it.
*   **Gradient Visualization:** Generates the heatmap overlay showing high-risk zones.

### `game_theory_predictor.py`:
Models cooperative interactions between vehicles.
*   **Interaction Detection:** Identifies conflict scenarios like intersections or merging.
*   **Payoff Matrix:** Evaluates strategies (Yield vs. Pass) based on safety, efficiency, and right-of-way rules.
*   **Velocity Adjustment:** Modifies the speed profile of predicted paths to reflect cooperative decisions (like slowing down to yield).

## 5. Visualization & Evaluation:

### `visualizer.py`:
Handles all rendering and graphical output.
*   **Overlays:** Draws the Traffic Flow Potential Surface (heatmap) and Road Mask.
*   **Trajectory Rendering:** Draws predicted paths as semi-transparent splines with arrowheads.
*   **Vehicle Annotation:** Renders Oriented Bounding Boxes (OBBs), tracking IDs, and history trails.
*   **Info Display:** Adds frame counters and legend information to the video.

### `evaluate_system.py`:
A standalone script for quantitative evaluation of the system.
*   **Test Harness:** Loads a dataset and runs the predictor on a set of test cases.
*   **Batch Processing:** Iterates through frames and vehicles to collect prediction samples.
*   **Reporting:** Aggregates results and prints summary statistics for validation and test splits.

### `evaluation_metrics.py`:
Defines the mathematical metrics used to judge performance.
*   **minADE@k:** Minimum Average Displacement Error (mean Euclidean distance).
*   **minFDE@k:** Minimum Final Displacement Error (endpoint distance).
*   **Miss Rate:** Percentage of predictions exceeding error thresholds.
*   **Norm FDE:** FDE normalized by trajectory length.
*   **APD:** Average Pairwise Distance for measuring diversity.
*   **NLL:** Negative Log Likelihood for probabilistic evaluation.
*   **Collision Rate:** Percentage of predictions that intersect with other agents.
*   **Off-Road Rate:** Percentage of predictions that leave the drivable area.

## 6. Dependencies:

### `requirements.txt`:
Lists the external Python libraries required to run the project.
*   **Core:** `numpy`, `opencv-python`, `scipy`, `shapely`.
*   **Optional:** `geojson` (for lane export).
