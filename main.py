"""
Behavior-Aware Trajectory Prediction System
Main entry point for the trajectory prediction pipeline.

Based on: "Behavior-Aware Trajectory Prediction in Unstructured Traffic"

This system performs deterministic, mathematical trajectory prediction using:
- Social potential fields for collision avoidance
- Game theory for cooperative behavior modeling
- Lane-based path planning from emerging lanes
- Road mask constraints
"""
import os
import sys
import cv2
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple

import config
from track_data_loader import TrackDataLoader
from road_mask_manager import RoadMaskManager
from emerging_lane_extractor import EmergingLaneExtractor
from lane_manager import LaneManager
from social_potential_field import SocialPotentialField
from game_theory_predictor import GameTheoryPredictor
from trajectory_predictor import TrajectoryPredictor
from visualizer import Visualizer


class BehaviorAwarePredictor:
    """
    Main class for behavior-aware trajectory prediction system.
    """
    
    def __init__(self):
        """Initialize the prediction system."""
        self.track_loader = None
        self.road_mask_manager = None
        self.lane_manager = None
        self.lane_extractor = None
        self.social_field = None
        self.game_theory = None
        self.trajectory_predictor = None
        self.visualizer = None
        
        self.fps = config.DEFAULT_FPS
        self.frame_width = 0
        self.frame_height = 0
        
        # Tracking state
        self.vehicle_histories = {}  # id -> deque of positions
        self.vehicle_predictions = {}  # id -> list of predictions
        self.history_length = int(config.PAST_HISTORY_SECONDS * config.DEFAULT_FPS)
        
        # Frame processing
        self.current_frame = 0
        self.prediction_interval = 1  # Predict EVERY frame for all vehicles
    
    def initialize(self):
        """Initialize all components of the system."""
        print("=" * 60)
        print("Behavior-Aware Trajectory Prediction System")
        print("=" * 60)
        
        # Load video info
        print("\n[1/7] Loading video information...")
        self._load_video_info()
        
        # Load tracking data
        print("\n[2/7] Loading pre-computed tracking data...")
        self._load_tracking_data()
        
        # Load road mask
        print("\n[3/7] Loading and rasterizing road mask...")
        self._load_road_mask()
        
        # Extract emerging lanes
        print("\n[4/7] Extracting emerging lanes from trajectories...")
        self._extract_lanes()
        
        # Initialize social potential field
        print("\n[5/7] Initializing social potential field system...")
        self._init_social_field()
        
        # Initialize game theory predictor
        print("\n[6/7] Initializing game theory cooperative predictor...")
        self._init_game_theory()
        
        # Initialize trajectory predictor
        print("\n[7/7] Initializing trajectory prediction engine...")
        self._init_trajectory_predictor()
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
    
    def _load_video_info(self):
        """Load video metadata."""
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.VIDEO_PATH}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()
        
        self.history_length = int(config.PAST_HISTORY_SECONDS * self.fps)
        
        print(f"  Video: {config.VIDEO_PATH}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.2f}")
        print(f"  Total frames: {self.total_frames}")
    
    def _load_tracking_data(self):
        """Load pre-computed tracking data."""
        self.track_loader = TrackDataLoader(config.TRACKING_DATA_PATH)
        
        print(f"  Loaded {len(self.track_loader.tracks)} tracks")
        print(f"  Frame range: {self.track_loader.metadata.get('min_frame', 0)} - "
              f"{self.track_loader.metadata.get('max_frame', 0)}")
        
        # Check for mismatch with video length
        max_track_frame = self.track_loader.metadata.get('max_frame', 0)
        if max_track_frame < self.total_frames * 0.9:
            print(f"\n  WARNING: Tracking data covers only {max_track_frame} frames")
            print(f"           Video has {self.total_frames} frames")
            print(f"           Processing will stop when tracking data ends.")
    
    def _load_road_mask(self):
        """Load and rasterize road mask."""
        self.road_mask_manager = RoadMaskManager(
            config.ROAD_MASK_PATH,
            self.frame_width,
            self.frame_height
        )
    
    def _extract_lanes(self):
        """Extract emerging lanes from tracking data."""
        print("  Extracting emerging lanes...")
        self.lane_extractor = EmergingLaneExtractor(self.track_loader, self.fps)
        lane_data = self.lane_extractor.extract_lanes()
        
        # Initialize lane manager with extracted lanes
        # Lane manager expects list of points, so we extract 'points' from the dicts
        raw_lanes = [l['points'] if isinstance(l, dict) else l for l in lane_data['lanes']]
        self.lane_manager = LaneManager()
        self.lane_manager.add_lanes_from_extractor(
            raw_lanes,
            lane_data['connections']
        )
        
        # Embed lanes into road mask potential surface
        print(f"  Embedding {len(lane_data['lanes'])} lanes into navigation potential surface...")
        for lane_obj in lane_data['lanes']:
            if isinstance(lane_obj, dict):
                points = lane_obj['points']
                score = lane_obj.get('score', 1.0)
            else:
                points = lane_obj
                score = 1.0
                
            self.road_mask_manager.embed_lane_spline(
                points, 
                sigma=20.0, 
                weight=0.6,
                confidence_score=score
            )
            
        print(f"  Extracted {len(lane_data['lanes'])} lanes")
        print(f"  Found {len(lane_data['connections'])} connections")
    
    def _init_social_field(self):
        """Initialize social potential field system."""
        self.social_field = SocialPotentialField(
            self.frame_width, self.frame_height
        )
        
        print(f"  Critical radius: {config.SOCIAL_POTENTIAL_RADIUS_CRITICAL}px")
        print(f"  Awareness radius: {config.SOCIAL_POTENTIAL_RADIUS_LOW}px")
    
    def _init_game_theory(self):
        """Initialize game theory predictor."""
        self.game_theory = GameTheoryPredictor()
        
        print(f"  Yield angle threshold: {config.YIELD_ANGLE_THRESHOLD}°")
        print(f"  Right-of-way distance: {config.RIGHT_OF_WAY_DISTANCE}px")
    
    def _init_trajectory_predictor(self):
        """Initialize trajectory prediction engine."""
        self.trajectory_predictor = TrajectoryPredictor(
            self.road_mask_manager,
            self.lane_manager,
            self.social_field,
            self.game_theory,
            self.fps
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(self.frame_width, self.frame_height)
        
        print(f"  History window: {config.PAST_HISTORY_SECONDS}s")
        print(f"  Prediction horizon: {config.FUTURE_PREDICTION_SECONDS}s")
        print(f"  Top paths to display: {config.NUM_PREDICTION_PATHS}")
    
    def process_frame(self, frame_num: int) -> Dict:
        """
        Process a single frame and generate predictions for ALL vehicles.
        
        Args:
            frame_num: Frame number to process
            
        Returns:
            Dictionary with vehicle states, predictions, and detections (with OBBs)
        """
        # Get detections for this frame
        detections = self.track_loader.get_frame_detections(frame_num)
        
        # Update vehicle histories
        current_vehicle_ids = set()
        
        for det in detections:
            track_id = det['track_id']
            position = (det['center_x'], det['center_y'])
            current_vehicle_ids.add(track_id)
            
            if track_id not in self.vehicle_histories:
                self.vehicle_histories[track_id] = deque(maxlen=self.history_length)
            
            self.vehicle_histories[track_id].append(position)
        
        # Remove vehicles no longer in frame
        old_ids = set(self.vehicle_histories.keys()) - current_vehicle_ids
        for old_id in old_ids:
            if old_id in self.vehicle_histories:
                del self.vehicle_histories[old_id]
            if old_id in self.vehicle_predictions:
                del self.vehicle_predictions[old_id]
        
        # Generate predictions for ALL vehicles EVERY frame (pass detections for OBB collision checking)
        self._generate_predictions(detections)
        
        # Build vehicle states for visualization (include ALL vehicles)
        vehicle_states = []
        for vid, history in self.vehicle_histories.items():
            if len(history) >= 2:  # Lower threshold to show more vehicles
                pos = history[-1]
                
                # Calculate velocity from history
                if len(history) >= 3:
                    kinematics = self.trajectory_predictor.compute_kinematics(list(history))
                    if kinematics:
                        vehicle_states.append({
                            'id': vid,
                            'position': kinematics['position'],
                            'velocity': kinematics['velocity'],
                            'direction': kinematics['direction'],
                            'speed': kinematics['speed'],
                            'history': list(history)
                        })
                else:
                    # For vehicles with short history, still show them
                    vehicle_states.append({
                        'id': vid,
                        'position': np.array(pos),
                        'velocity': np.array([0, 0]),
                        'direction': np.array([1, 0]),
                        'speed': 0,
                        'history': list(history)
                    })
        
        return {
            'vehicles': vehicle_states,
            'predictions': self.vehicle_predictions.copy(),
            'detections': detections  # Include raw detections with OBB data
        }
    
    def _generate_predictions(self, detections: List[Dict] = None):
        """Generate predictions for ALL tracked vehicles.
        
        Args:
            detections: Optional list of current frame detections with OBB data
        """
        # Prepare all vehicle histories (use lower threshold)
        all_histories = {
            vid: list(hist) for vid, hist in self.vehicle_histories.items()
            if len(hist) >= 5  # Lower threshold: ~0.08 seconds at 60fps
        }
        
        # Extract OBB data from detections for collision checking
        all_obbs = []
        if detections:
            for det in detections:
                if det.get('obb_corners') and len(det.get('obb_corners', [])) == 4:
                    all_obbs.append({
                        'track_id': det['track_id'],
                        'obb_corners': det['obb_corners']
                    })
        
        # Generate predictions for EACH vehicle
        for vid, history in all_histories.items():
            try:
                predictions = self.trajectory_predictor.predict_trajectories(
                    vid, history, all_histories,
                    config.FUTURE_PREDICTION_SECONDS,
                    all_obbs=all_obbs  # Pass OBBs for collision checking
                )
                
                if predictions:
                    self.vehicle_predictions[vid] = predictions
                else:
                    # Generate basic kinematic prediction if no advanced predictions
                    kinematics = self.trajectory_predictor.compute_kinematics(history)
                    if kinematics and kinematics['speed'] > 0.5:
                        basic_path = self.trajectory_predictor.predict_constant_velocity(
                            kinematics, config.FUTURE_PREDICTION_SECONDS
                        )
                        self.vehicle_predictions[vid] = [{
                            'path_points': basic_path,
                            'probability': 0.5,
                            'type': 'kinematic',
                            'details': {}
                        }]
            except Exception as e:
                # Don't fail on individual vehicle prediction
                pass
    
    def visualize_frame(self, frame: np.ndarray, 
                        frame_data: Dict,
                        frame_num: int) -> np.ndarray:
        """
        Create visualization for a frame.
        
        Args:
            frame: Input BGR frame
            frame_data: Output from process_frame
            frame_num: Current frame number
            
        Returns:
            Visualized frame
        """
        vehicles = frame_data['vehicles']
        predictions = frame_data['predictions']
        detections = frame_data.get('detections', [])
        
        # Create composite visualization with OBBs, tracking IDs, and gradient heatmap
        result = self.visualizer.create_composite_visualization(
            frame,
            self.road_mask_manager.road_mask,
            vehicles,
            predictions,
            self.lane_manager.lanes,
            frame_num,
            detections=detections  # Pass detections for OBB visualization
        )
        
        return result
    
    def run(self, frame_limit=None):
        """Run the full prediction pipeline on the video."""
        self.initialize()
        
        print("\n" + "=" * 60)
        print("Processing video...")
        print("=" * 60)
        
        # Setup video capture
        cap = cv2.VideoCapture(config.VIDEO_PATH)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.VIDEO_PATH}")
        
        # Setup output
        os.makedirs(config.PREDICTION_OUTPUT_DIR, exist_ok=True)
        
        # Add suffix if partial run
        filename = "trajectory_prediction_output.mp4"
        if frame_limit:
            filename = f"trajectory_prediction_output_{frame_limit}frames.mp4"
            
        output_path = os.path.join(
            config.PREDICTION_OUTPUT_DIR,
            filename
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                              (self.frame_width, self.frame_height))
        
        frame_num = 0
        max_track_frame = self.track_loader.metadata.get('max_frame', 0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Stop if frame limit reached
                if frame_limit and frame_num > frame_limit:
                    print(f"\nStopping: Reached frame limit ({frame_limit})")
                    break
                
                # Stop if we exceeded tracking data range significantly
                if frame_num > max_track_frame + 30:  # 30 frame buffer
                    print(f"\nStopping: Reached end of tracking data (Frame {max_track_frame})")
                    break
                
                self.current_frame = frame_num
                
                # Process frame
                frame_data = self.process_frame(frame_num)
                
                # Visualize
                result = self.visualize_frame(frame, frame_data, frame_num)
                
                # Write output
                out.write(result)
                
                # Progress update
                if frame_num % 10 == 0:  # More frequent updates for short runs
                    print(f"  Processed {frame_num}/{self.total_frames} frames "
                          f"({100*frame_num/self.total_frames:.1f}%)")
        
        finally:
            cap.release()
            out.release()
        
        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total frames processed: {frame_num}")
        
        return output_path


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Behavior-Aware Trajectory Prediction')
    parser.add_argument('--frames', type=int, help='Limit number of frames to process')
    args = parser.parse_args()

    try:
        # Verify input files exist
        print("Verifying input files...")
        
        if not os.path.exists(config.VIDEO_PATH):
            print(f"❌ Video not found: {config.VIDEO_PATH}")
            return 1
        print(f"✓ Video found: {config.VIDEO_PATH}")
        
        if not os.path.exists(config.TRACKING_DATA_PATH):
            print(f"❌ Tracking data not found: {config.TRACKING_DATA_PATH}")
            return 1
        print(f"✓ Tracking data found: {config.TRACKING_DATA_PATH}")
        
        if not os.path.exists(config.ROAD_MASK_PATH):
            print(f"❌ Road mask not found: {config.ROAD_MASK_PATH}")
            return 1
        print(f"✓ Road mask found: {config.ROAD_MASK_PATH}")
        
        # Run prediction system
        predictor = BehaviorAwarePredictor()
        output_path = predictor.run(frame_limit=args.frames)
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"\nOutput video saved to:")
        print(f"  {output_path}")
        print(f"\nEmerging lanes saved to:")
        print(f"  {config.LANES_OUTPUT_DIR}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
