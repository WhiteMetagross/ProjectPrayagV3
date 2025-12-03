# Installation and Setup:

This guide details the steps to set up the **Behavior-Aware Trajectory Prediction System** on your local machine.

## 1. Prerequisites:

Ensure you have the following software installed:

*   **Python 3.8** or higher.
*   **pip** (Python package installer).
*   **Git** (for cloning the repository).

## 2. Installation:

### 2.1. Clone the Repository:

Open your terminal or command prompt and clone the project repository:

```bash
git clone https://github.com/WhiteMetagross/PrayagProjectv3.git
cd PrayagProjectv3
```

### 2.2. Create a Virtual Environment (Recommended):

It is best practice to use a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3. Install Dependencies:

Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
*   `numpy`: For vectorized mathematical operations.
*   `opencv-python`: For video processing and visualization.
*   `scipy`: For spatial distance metrics (Hausdorff distance) and signal processing.
*   `shapely`: For geometric operations and polygon manipulation.

## 3. Data Setup:

The system requires specific input data to function. Ensure your project directory has the following structure for data files. You may need to create these directories if they do not exist.

### 3.1. Directory Structure:

```text
PrayagProjectv3/
├── config.py               # Configuration file
├── main.py                 # Main execution script
├── ...
├── output/                 # Generated outputs
│   ├── emerging_lanes/     # Extracted lane data (GeoJSON)
│   └── predictions/        # Output videos
└── [Your Data Directory]/  # External or internal data folder
    ├── video.mp4           # Source traffic video
    ├── tracks.csv          # Pre-computed vehicle tracks
    └── road_mask.json      # Drivable area polygon
```

### 3.2. Input Files:

1.  **Video File (`.mp4`):** The raw footage of the traffic scene.
2.  **Tracking Data (`.csv`):** A CSV file containing pre-computed vehicle trajectories. The system expects columns for `frame_id`, `track_id`, `x`, `y`, etc.
3.  **Road Mask (`.json`):** A JSON file defining the polygon coordinates of the drivable road area. This is used to constrain predictions and calculate off-road metrics.

*Note: The paths to these files must be configured in `config.py` before running the system.*
