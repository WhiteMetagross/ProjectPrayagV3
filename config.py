"""
Configuration for Behavior-Aware Trajectory Prediction System
Based on: "Behavior-Aware Trajectory Prediction in Unstructured Traffic"
"""
import os

# ==================== PATH CONFIGURATION ====================
# Data source paths
TRACKING_DATA_PATH = r"C:\Users\Xeron\OneDrive\Documents\Programs\PrayagProjectv1.5\output\intermediate_files\DJI_0912"
ROAD_MASK_PATH = r"C:\Users\Xeron\Videos\CIRAerialDroneInianIndtersectionsVideoes\DJI_0912_road_annotation.json"
VIDEO_PATH = r"C:\Users\Xeron\OneDrive\Documents\Programs\PrayagProjectv1.5\ProjectPrayagTopDownDataset\CIRAerialDroneIndianIntersectionsVideoes\DJI_0912.mp4"

# Output paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LANES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "emerging_lanes")
PREDICTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "predictions")

# ==================== TRAJECTORY PREDICTION PARAMETERS ====================
# Time horizon parameters (seconds)
PAST_HISTORY_SECONDS = 4.0  # Use past 3-5 seconds for prediction (use 4 seconds)
FUTURE_PREDICTION_SECONDS = 2.0  # Predict 2 seconds into future

# FPS - will be auto-detected from video
DEFAULT_FPS = 30.0

# ==================== SOCIAL POTENTIAL FIELD PARAMETERS ====================
# Social potential radius bands (in pixels)
SOCIAL_POTENTIAL_RADIUS_CRITICAL = 30.0   # Critical zone - immediate collision risk
SOCIAL_POTENTIAL_RADIUS_HIGH = 60.0       # High risk zone
SOCIAL_POTENTIAL_RADIUS_MEDIUM = 100.0    # Medium risk zone
SOCIAL_POTENTIAL_RADIUS_LOW = 150.0       # Low risk zone - awareness zone

# Potential field decay factors (exponential decay)
POTENTIAL_DECAY_CRITICAL = 1.0
POTENTIAL_DECAY_HIGH = 0.7
POTENTIAL_DECAY_MEDIUM = 0.4
POTENTIAL_DECAY_LOW = 0.15

# Heatmap colors for social potential visualization (BGR format for OpenCV)
HEATMAP_COLORS = [
    (0, 0, 255),      # Critical - Red
    (0, 128, 255),    # High - Orange
    (0, 255, 255),    # Medium - Yellow
    (0, 255, 0),      # Low - Green
]

# ==================== GAME THEORY COOPERATIVE PARAMETERS ====================
# Priority thresholds for cooperative behavior
YIELD_ANGLE_THRESHOLD = 45.0              # Angle difference to consider yielding (degrees)
RIGHT_OF_WAY_DISTANCE = 80.0              # Distance threshold for right-of-way decisions
COOPERATIVE_PREDICTION_HORIZON = 1.5      # Time horizon for cooperative predictions (seconds)

# Vehicle priority scoring weights
SPEED_PRIORITY_WEIGHT = 0.3
DIRECTION_PRIORITY_WEIGHT = 0.4
DISTANCE_PRIORITY_WEIGHT = 0.3

# ==================== LANE EXTRACTION PARAMETERS ====================
MIN_TRACK_DURATION_SECONDS = 2.5          # Minimum track duration for valid lane
MIN_TRACK_POINTS = 10                     # Minimum number of points in a track
HAUSDORFF_THRESHOLD = 20.0                # Threshold for merging similar tracks
ENDPOINT_SNAP_TOLERANCE = 15.0            # Tolerance for snapping endpoints
SMOOTHING_WINDOW_SIZE = 5                 # Window size for track smoothing
SIMPLIFY_TOLERANCE = 2.0                  # Tolerance for simplifying lane geometry

# ==================== TRAJECTORY PREDICTION ENGINE PARAMETERS ====================
# Motion model parameters
VELOCITY_SMOOTHING_WINDOW = 5             # Frames to smooth velocity calculation
ACCELERATION_SMOOTHING_WINDOW = 3         # Frames to smooth acceleration

# Realistic physics constraints (pixels per second^2)
MAX_ACCELERATION = 50.0                   # Max acceleration magnitude (px/s^2)
MAX_DECELERATION = 80.0                   # Max braking deceleration (px/s^2)
MAX_LATERAL_ACCELERATION = 40.0           # Max turning acceleration (px/s^2)
MIN_STOPPING_DISTANCE = 10.0              # Minimum distance to stop (pixels)

# Path prediction parameters
NUM_PREDICTION_PATHS = 4                  # Number of top paths to display
PATH_SAMPLING_RATE = 10                   # Points per predicted path
MIN_PATH_PROBABILITY = 0.15               # Minimum probability for valid path
LANE_ALIGNMENT_THRESHOLD = 0.4            # Minimum alignment with lane direction

# Road constraint parameters
ROAD_BOUNDARY_MARGIN = 10.0               # Margin from road edge (pixels)
OFF_ROAD_PENALTY = 100.0                  # Penalty for off-road paths

# ==================== VISUALIZATION PARAMETERS ====================
# Path visualization colors (BGR format) - for top 4 paths
PATH_COLORS = [
    (0, 255, 0),      # Best path - Green
    (255, 255, 0),    # 2nd best - Cyan
    (255, 165, 0),    # 3rd best - Orange/Yellow
    (255, 0, 255),    # 4th best - Magenta
]

# Arrow visualization
ARROW_LENGTH = 15
ARROW_ANGLE = 0.4  # Radians (~23 degrees)
PATH_LINE_THICKNESS = 2

# Road mask overlay
ROAD_MASK_ALPHA = 0.3
ROAD_MASK_COLOR = (100, 150, 100)  # Greenish tint for road

# Social potential overlay
SOCIAL_POTENTIAL_ALPHA = 0.45  # Increased for better visibility

# Trail visualization
TRAIL_LENGTH = 50                         # Number of past points to display
TRAIL_THICKNESS = 2

# ==================== DEBUG AND LOGGING ====================
DEBUG_MODE = False
VERBOSE_OUTPUT = True
SAVE_INTERMEDIATE_RESULTS = True
