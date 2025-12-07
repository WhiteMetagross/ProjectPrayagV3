"""
System Evaluation Script
Runs the trajectory prediction system on a test set and computes quantitative metrics.
"""
import os
import argparse
import glob
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

def evaluate_scenario(track_path, road_mask_path):
    print(f"Evaluating: {os.path.basename(track_path)}")
    
    # 1. Load Data
    track_loader = TrackDataLoader(track_path)
    
    # Hardcoded resolution for DJI dataset
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    road_mask = RoadMaskManager(road_mask_path, FRAME_WIDTH, FRAME_HEIGHT)
    
    # 2. Initialize Components
    lane_manager = LaneManager()
    
    # Extract lanes
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
    history_frames = int(config.PAST_HISTORY_SECONDS * config.DEFAULT_FPS)
    future_frames = int(config.FUTURE_PREDICTION_SECONDS * config.DEFAULT_FPS)
    
    test_cases = []
    
    # Sample every 50th frame
    start_frame = track_loader.metadata['min_frame'] + history_frames
    end_frame = track_loader.metadata['max_frame'] - future_frames
    frame_step = 50
    
    if start_frame >= end_frame:
        print("  Skipping: Not enough frames for history+future window.")
        return {}

    for frame_idx in range(start_frame, end_frame, frame_step):
        active_tracks = track_loader.get_active_tracks_at_frame(frame_idx)
        
        for track_id in active_tracks:
            track_data = track_loader.get_track(track_id)
            
            curr_idx = -1
            for i, d in enumerate(track_data):
                if d['frame_id'] == frame_idx:
                    curr_idx = i
                    break
            
            if curr_idx == -1: continue
            if curr_idx < history_frames: continue
            if curr_idx + future_frames >= len(track_data): continue
            
            test_cases.append({
                'frame_id': frame_idx,
                'track_id': track_id,
                'curr_idx': curr_idx
            })
            
    print(f"  Found {len(test_cases)} valid test cases.")
    
    # 4. Run Evaluation Loop
    metrics_agg = {
        'minADE@1': [],
        'minADE@4': [],
        'minFDE@1': [],
        'minFDE@4': [],
        'miss_rate_10': [],
        'miss_rate_20': [],
        'norm_fde': [],
        'apd': [],
        'nll': [],
        'collision': [],
        'off_road': []
    }
    
    for case in tqdm(test_cases, desc="Evaluating frames"):
        frame_id = case['frame_id']
        ego_id = case['track_id']
        curr_idx = case['curr_idx']
        
        full_track = track_loader.get_track(ego_id)
        history_data = full_track[curr_idx - history_frames : curr_idx + 1]
        ego_history = [(d['center_x'], d['center_y']) for d in history_data]
        
        future_data = full_track[curr_idx + 1 : curr_idx + 1 + future_frames]
        gt_future = [(d['center_x'], d['center_y']) for d in future_data]
        
        all_vehicles_history = {}
        other_agents_futures = []
        
        active_others = track_loader.get_active_tracks_at_frame(frame_id)
        for other_id in active_others:
            if other_id == ego_id: continue
            
            other_track = track_loader.get_track(other_id)
            
            o_curr_idx = -1
            for i, d in enumerate(other_track):
                if d['frame_id'] == frame_id:
                    o_curr_idx = i
                    break
            
            if o_curr_idx != -1:
                start_idx = max(0, o_curr_idx - history_frames)
                o_hist = other_track[start_idx : o_curr_idx + 1]
                all_vehicles_history[other_id] = [(d['center_x'], d['center_y']) for d in o_hist]
                
                o_fut = other_track[o_curr_idx + 1 : o_curr_idx + 1 + future_frames]
                if o_fut:
                    other_agents_futures.append([(d['center_x'], d['center_y']) for d in o_fut])

        all_vehicles_history[ego_id] = ego_history
        
        predictions = predictor.predict_trajectories(
            ego_id=ego_id,
            ego_history=ego_history,
            all_vehicles=all_vehicles_history,
            time_horizon=config.FUTURE_PREDICTION_SECONDS
        )
        
        res = evaluator.evaluate_batch(predictions, gt_future, other_agents_futures, k_list=[1, 4])
        
        if res:
            for k in metrics_agg.keys():
                if k in res:
                    metrics_agg[k].append(res[k])

    return metrics_agg

def main():
    parser = argparse.ArgumentParser(description='Evaluate Trajectory Prediction System')
    parser.add_argument('--split', type=str, choices=['val', 'test'], help='Dataset split to evaluate')
    args = parser.parse_args()

    if args.split:
        base_dir = os.path.join("ChunkedProjectPrayagBEVDataset", args.split, "annotations")
        track_files = glob.glob(os.path.join(base_dir, "*_tracks.csv"))
        
        print(f"Found {len(track_files)} track files in {base_dir}")
        
        all_metrics = {
            'minADE@1': [],
            'minADE@4': [],
            'minFDE@1': [],
            'minFDE@4': [],
            'miss_rate_10': [],
            'miss_rate_20': [],
            'norm_fde': [],
            'apd': [],
            'nll': [],
            'collision': [],
            'off_road': []
        }
        
        for track_path in track_files:
            # Find corresponding road mask
            # Assuming naming convention: DJI_0910_chunk_10_tracks.csv -> DJI_0910_chunk_10_road_annotation.json
            road_mask_path = track_path.replace("_tracks.csv", "_road_annotation.json")
            
            if not os.path.exists(road_mask_path):
                print(f"Warning: Road mask not found for {track_path}. Skipping.")
                continue
                
            metrics = evaluate_scenario(track_path, road_mask_path)
            
            for k, v in metrics.items():
                if k in all_metrics:
                    all_metrics[k].extend(v)
        
        print("\n" + "="*60)
        print(f"FINAL RESULTS FOR {args.split.upper()} SPLIT")
        print("="*60)
        if all_metrics['minADE@4']:
            print(f"Total Samples: {len(all_metrics['minADE@4'])}")
            print(f"minADE@1:       {np.mean(all_metrics['minADE@1']):.2f} pixels")
            print(f"minADE@4:       {np.mean(all_metrics['minADE@4']):.2f} pixels")
            print(f"minFDE@1:       {np.mean(all_metrics['minFDE@1']):.2f} pixels")
            print(f"minFDE@4:       {np.mean(all_metrics['minFDE@4']):.2f} pixels")
            print(f"Miss Rate @10px: {np.mean(all_metrics['miss_rate_10']) * 100:.2f}%")
            print(f"Miss Rate @20px: {np.mean(all_metrics['miss_rate_20']) * 100:.2f}%")
            print(f"Norm FDE:       {np.mean(all_metrics['norm_fde']):.4f}")
            print(f"APD (Diversity):{np.mean(all_metrics['apd']):.2f} pixels")
            print(f"NLL:            {np.mean(all_metrics['nll']):.4f}")
            print(f"Collision Rate: {np.mean(all_metrics['collision']) * 100:.2f}%")
            print(f"Off-Road Rate:  {np.mean(all_metrics['off_road']) * 100:.2f}%")
        else:
            print("No valid samples evaluated.")
        print("="*60)
        
    else:
        # Default behavior
        metrics = evaluate_scenario(config.TRACKING_DATA_PATH, config.ROAD_MASK_PATH)
        
        print("\n[5/5] Results:")
        print("------------------------------------------------------------")
        if metrics['minADE@4']:
            print(f"Samples: {len(metrics['minADE@4'])}")
            print(f"minADE@1:       {np.mean(metrics['minADE@1']):.2f} pixels")
            print(f"minADE@4:       {np.mean(metrics['minADE@4']):.2f} pixels")
            print(f"minFDE@1:       {np.mean(metrics['minFDE@1']):.2f} pixels")
            print(f"minFDE@4:       {np.mean(metrics['minFDE@4']):.2f} pixels")
            print(f"Miss Rate @10px: {np.mean(metrics['miss_rate_10']) * 100:.2f}%")
            print(f"Miss Rate @20px: {np.mean(metrics['miss_rate_20']) * 100:.2f}%")
            print(f"Norm FDE:       {np.mean(metrics['norm_fde']):.4f}")
            print(f"APD (Diversity):{np.mean(metrics['apd']):.2f} pixels")
            print(f"NLL:            {np.mean(metrics['nll']):.4f}")
            print(f"Collision Rate: {np.mean(metrics['collision']) * 100:.2f}%")
            print(f"Off-Road Rate:  {np.mean(metrics['off_road']) * 100:.2f}%")
        else:
            print("No valid samples evaluated.")
        print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
