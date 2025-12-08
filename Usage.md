# Usage Guide:

This document provides instructions on how to configure, run, and evaluate the **Behavior-Aware Trajectory Prediction System**.

## 1. Configuration:

Before running the system, you must configure the input paths and system parameters in `config.py`.

Open `config.py` and update the following variables to match your data location:

```python
# config.py

# Input Paths
VIDEO_PATH = "path/to/your/video.mp4"
TRACKING_DATA_PATH = "path/to/your/tracks.csv"
ROAD_MASK_PATH = "path/to/your/road_mask.json"

# System Parameters
PAST_HISTORY_SECONDS = 2.0      # Duration of observation window
FUTURE_PREDICTION_SECONDS = 4.0 # Duration of prediction horizon
```

You can also tune algorithmic parameters such as `SOCIAL_POTENTIAL_RADIUS_CRITICAL` or `YIELD_ANGLE_THRESHOLD` in this file to adjust the behavior of the predictor.

## 2. Running the Prediction Engine:

To run the full trajectory prediction pipeline, use the `main.py` script. This will process the video, extract emerging lanes, and generate visualizations.

### Basic Usage:

```bash
python main.py
```

### Limiting Frames:

For testing or debugging, you can limit the number of frames processed using the `--frames` argument:

```bash
python main.py --frames 500
```
*This command will process only the first 500 frames.*

### Output:

The system generates the following outputs:
*   **Visualization Video:** Saved to `output/predictions/trajectory_prediction_output.mp4`. This video shows the original footage overlaid with.
    *   **Heatmap:** The Traffic Flow Potential Surface.
    *   **Splines:** Predicted trajectories for each vehicle.
    *   **OBBs:** Oriented Bounding Boxes for vehicles.
*   **Emerging Lanes:** Extracted lane geometry saved as `emerging_lanes.geojson` in `output/emerging_lanes/`.

## 3. System Evaluation:

To quantitatively assess the performance of the model, use the `evaluate_system.py` script. This module calculates metrics such as Average Displacement Error (ADE), Final Displacement Error (FDE), and Collision Rates.

### Running Evaluation:

```bash
python evaluate_system.py --split val
# or
python evaluate_system.py --split test
```

### Evaluating on Custom Datasets (e.g., 10Hz):

You can specify a different dataset directory and frame rate using the `--dataset` and `--fps` arguments. This is useful for evaluating on downsampled versions of the dataset.

```bash
python evaluate_system.py --split val --dataset ChunkedProjectPrayagBEVDataset10Hz --fps 10.0
```

### Evaluation Metrics:

The script reports the following metrics:

*   **minADE@k:** Minimum Average Displacement Error at k=1 and k=4. Measures the average Euclidean distance between the best predicted path and the ground truth.
*   **minFDE@k:** Minimum Final Displacement Error at k=1 and k=4. Measures the distance between the predicted endpoint and the actual endpoint.
*   **Miss Rate:** The percentage of cases where the best prediction's final error exceeds a threshold (10px and 20px).
*   **Norm FDE:** Final Displacement Error normalized by the length of the ground truth trajectory.
*   **APD (Diversity):** Average Pairwise Distance between the top predicted trajectories, measuring the diversity of the predictions.
*   **NLL:** Negative Log Likelihood, evaluating the probabilistic quality of the predictions.
*   **Collision Rate:** The percentage of predicted trajectories that result in a collision with another agent.
*   **Off-Road Rate:** The percentage of predicted trajectories that exit the drivable area defined by the road mask.

## 4. Troubleshooting:

*   **"Video not found"**: Ensure the `VIDEO_PATH` in `config.py` is absolute or correctly relative to the project root.
*   **"Tracking data covers only X frames"**: This warning appears if your CSV file is shorter than the video. The system will stop processing when tracking data runs out.
*   **Low FPS**: The visualization and potential field calculation can be computationally intensive. Reduce `NUM_PREDICTION_PATHS` in `config.py` to improve performance.
