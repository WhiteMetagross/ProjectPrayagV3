"""
System Evaluation Script
Runs the trajectory prediction system on a test set and computes quantitative metrics.
"""
import os
import numpy as np
from tqdm import tqdm
import config
from track_data_loader import TrackDataLoader
from road_mask_manager import RoadMaskManager
from lane_manager import LaneManager
from social_potential_field import SocialPotentialField
from game_theory_predictor import GameTheoryPredictor
from trajectory_predictor import TrajectoryPredictor
from evaluation_metrics import TrajectoryEvaluator
from emerging_lane_extractor import EmergingLaneExtractor

def run_evaluation():
    print("============================================================")
    print("Trajectory Prediction System Evaluation")
    print("============================================================")

    # 1. Load Data
    print("[1/5] Loading data...")
    track_loader = TrackDataLoader(config.TRACKING_DATA_PATH)
    
    # Hardcoded resolution for DJI dataset
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    road_mask = RoadMaskManager(config.ROAD_MASK_PATH, FRAME_WIDTH, FRAME_HEIGHT)
    
    # 2. Initialize Components
    print("[2/5] Initializing components...")
    # We need to extract lanes first (simplified for eval, or load from file if saved)
    # For accurate eval, we should ideally use the same lanes as main.py.
    # Here we'll re-run extraction briefly or assume it's fast enough.
    lane_manager = LaneManager()
    
    # Extract lanes (using same logic as main.py)
    print("  Extracting lanes for context...")
    lane_extractor = EmergingLaneExtractor(track_loader, config.DEFAULT_FPS)
    trajectories = track_loader.extract_lane_trajectories()
    lanes_data = lane_extractor.extract_lanes()
    lanes_dicts = lanes_data['lanes']
    connections = lanes_data['connections']
    
    # Extract just the points for lane manager
    lane_points_list = [l['points'] for l in lanes_dicts]
    lane_manager.add_lanes_from_extractor(lane_points_list, connections)
    
    social_field = SocialPotentialField(FRAME_WIDTH, FRAME_HEIGHT)
    game_theory = GameTheoryPredictor()
    
    # Initialize Predictor
    predictor = TrajectoryPredictor(
        road_mask, lane_manager, social_field, game_theory, config.DEFAULT_FPS
    )
    
    evaluator = TrajectoryEvaluator(road_mask)

    # 3. Define Test Set
    # We'll sample frames and vehicles.
    # Criteria: Vehicle must have history (4s) and future (2s).
    print("[3/5] Selecting test cases...")
    
    history_frames = int(config.PAST_HISTORY_SECONDS * config.DEFAULT_FPS)
    future_frames = int(config.FUTURE_PREDICTION_SECONDS * config.DEFAULT_FPS)
    min_track_len = history_frames + future_frames
    
    test_cases = []
    
    # Sample every 50th frame to avoid high correlation and reduce compute time
    start_frame = track_loader.metadata['min_frame'] + history_frames
    end_frame = track_loader.metadata['max_frame'] - future_frames
    frame_step = 50
    
    for frame_idx in range(start_frame, end_frame, frame_step):
        active_tracks = track_loader.get_active_tracks_at_frame(frame_idx)
        
        for track_id in active_tracks:
            track_data = track_loader.get_track(track_id)
            
            # Check if track covers the required window around this frame
            # We need data from [frame - history] to [frame + future]
            
            # Find index of current frame in track data
            curr_idx = -1
            for i, d in enumerate(track_data):
                if d['frame_id'] == frame_idx:
                    curr_idx = i
                    break
            
            if curr_idx == -1: continue
            
            # Check history availability
            # We need enough points before current frame
            if curr_idx < history_frames:
                continue
                
            # Check future availability
            if curr_idx + future_frames >= len(track_data):
                continue
                
            # Check continuity (optional, but good for quality)
            # Ensure no large gaps in frame_ids
            
            test_cases.append({
                'frame_id': frame_idx,
                'track_id': track_id,
                'curr_idx': curr_idx
            })
            
    print(f"  Found {len(test_cases)} valid test cases.")
    
    # 4. Run Evaluation Loop
    print("[4/5] Running evaluation...")
    
    metrics_agg = {
        'minADE@4': [],
        'minFDE@4': [],
        'collision': [],
        'off_road': []
    }
    
    # Limit test cases for quick testing if needed, or run full
    # test_cases = test_cases[:100] 
    
    for case in tqdm(test_cases):
        frame_id = case['frame_id']
        ego_id = case['track_id']
        curr_idx = case['curr_idx']
        
        # Get Ego History
        full_track = track_loader.get_track(ego_id)
        history_data = full_track[curr_idx - history_frames : curr_idx + 1]
        ego_history = [(d['center_x'], d['center_y']) for d in history_data]
        
        # Get Ego Ground Truth Future
        future_data = full_track[curr_idx + 1 : curr_idx + 1 + future_frames]
        gt_future = [(d['center_x'], d['center_y']) for d in future_data]
        
        # Get Context (Other Vehicles)
        # We need their history for prediction input
        # And their future for collision checking
        all_vehicles_history = {}
        other_agents_futures = []
        
        active_others = track_loader.get_active_tracks_at_frame(frame_id)
        for other_id in active_others:
            if other_id == ego_id: continue
            
            other_track = track_loader.get_track(other_id)
            
            # Find index for this frame
            o_curr_idx = -1
            for i, d in enumerate(other_track):
                if d['frame_id'] == frame_id:
                    o_curr_idx = i
                    break
            
            if o_curr_idx != -1:
                # Extract history
                # We take whatever is available up to history_frames
                start_idx = max(0, o_curr_idx - history_frames)
                o_hist = other_track[start_idx : o_curr_idx + 1]
                all_vehicles_history[other_id] = [(d['center_x'], d['center_y']) for d in o_hist]
                
                # Extract future (for collision check)
                o_fut = other_track[o_curr_idx + 1 : o_curr_idx + 1 + future_frames]
                if o_fut:
                    other_agents_futures.append([(d['center_x'], d['center_y']) for d in o_fut])

        # Add ego to all_vehicles_history (required by predictor signature)
        all_vehicles_history[ego_id] = ego_history
        
        # Predict
        predictions = predictor.predict_trajectories(
            ego_id=ego_id,
            ego_history=ego_history,
            all_vehicles=all_vehicles_history,
            time_horizon=config.FUTURE_PREDICTION_SECONDS
        )
        
        # Evaluate
        res = evaluator.evaluate_batch(predictions, gt_future, other_agents_futures, k=4)
        
        if res:
            metrics_agg['minADE@4'].append(res['minADE@4'])
            metrics_agg['minFDE@4'].append(res['minFDE@4'])
            metrics_agg['collision'].append(1 if res['collision'] else 0)
            metrics_agg['off_road'].append(1 if res['off_road'] else 0)

    # 5. Report Results
    print("\n[5/5] Results:")
    print("------------------------------------------------------------")
    if metrics_agg['minADE@4']:
        print(f"Samples: {len(metrics_agg['minADE@4'])}")
        print(f"minADE@4:       {np.mean(metrics_agg['minADE@4']):.2f} pixels")
        print(f"minFDE@4:       {np.mean(metrics_agg['minFDE@4']):.2f} pixels")
        print(f"Collision Rate: {np.mean(metrics_agg['collision']) * 100:.2f}%")
        print(f"Off-Road Rate:  {np.mean(metrics_agg['off_road']) * 100:.2f}%")
    else:
        print("No valid samples evaluated.")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    run_evaluation()
