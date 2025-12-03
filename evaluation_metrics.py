"""
Evaluation Metrics Module
Provides standard trajectory prediction metrics (ADE, FDE, Collision Rate, Off-Road Rate).
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from road_mask_manager import RoadMaskManager

class TrajectoryEvaluator:
    """
    Calculates evaluation metrics for trajectory prediction.
    """
    
    def __init__(self, road_mask: RoadMaskManager):
        """
        Initialize the evaluator.
        
        Args:
            road_mask: RoadMaskManager instance for off-road checks.
        """
        self.road_mask = road_mask

    def compute_ade(self, pred_path: List[Tuple[float, float]], gt_path: List[Tuple[float, float]]) -> float:
        """
        Compute Average Displacement Error (ADE).
        Mean Euclidean distance between predicted and ground truth points.
        """
        if not pred_path or not gt_path:
            return float('inf')
        
        # Truncate to shorter length to compare
        min_len = min(len(pred_path), len(gt_path))
        if min_len == 0:
            return float('inf')
            
        pred_arr = np.array(pred_path[:min_len])
        gt_arr = np.array(gt_path[:min_len])
        
        distances = np.linalg.norm(pred_arr - gt_arr, axis=1)
        return float(np.mean(distances))

    def compute_fde(self, pred_path: List[Tuple[float, float]], gt_path: List[Tuple[float, float]]) -> float:
        """
        Compute Final Displacement Error (FDE).
        Euclidean distance between the final predicted point and final ground truth point.
        """
        if not pred_path or not gt_path:
            return float('inf')
            
        # Compare the last point of the prediction horizon
        # Assuming paths are sampled at same rate and cover same horizon
        # If lengths differ, we compare the last available point of the shorter path
        # or strictly the point at the target horizon.
        # Here we compare the last point of the shorter sequence to be robust.
        min_len = min(len(pred_path), len(gt_path))
        if min_len == 0:
            return float('inf')
            
        pred_end = np.array(pred_path[min_len-1])
        gt_end = np.array(gt_path[min_len-1])
        
        return float(np.linalg.norm(pred_end - gt_end))

    def check_off_road(self, path: List[Tuple[float, float]]) -> bool:
        """
        Check if any point in the path is off-road.
        Returns True if off-road, False otherwise.
        """
        if not path:
            return False # Empty path is technically not off-road, or invalid.
            
        # We consider a path off-road if a significant portion or the end is off-road.
        # Strict metric: Any point off-road.
        for point in path:
            if not self.road_mask.is_point_on_road(point):
                return True
        return False

    def check_collision(self, pred_path: List[Tuple[float, float]], 
                        other_agents_futures: List[List[Tuple[float, float]]], 
                        collision_radius: float = 20.0) -> bool:
        """
        Check if the predicted path collides with any other agent's ground truth future.
        
        Args:
            pred_path: Predicted path (x, y) tuples.
            other_agents_futures: List of future paths for other agents.
            collision_radius: Distance threshold for collision.
            
        Returns:
            True if collision detected, False otherwise.
        """
        if not pred_path:
            return False
            
        # We assume time-synchronization. Point i in pred_path corresponds to Point i in other_agent_future
        for i, ego_pos in enumerate(pred_path):
            ego_pos_arr = np.array(ego_pos)
            
            for other_path in other_agents_futures:
                if i < len(other_path):
                    other_pos_arr = np.array(other_path[i])
                    dist = np.linalg.norm(ego_pos_arr - other_pos_arr)
                    if dist < collision_radius:
                        return True
        return False

    def evaluate_batch(self, predictions: List[Dict], gt_path: List[Tuple[float, float]], 
                       other_agents_futures: List[List[Tuple[float, float]]],
                       k: int = 4) -> Dict:
        """
        Compute metrics for a single test case (one ego vehicle).
        
        Args:
            predictions: List of prediction dicts (must have 'path_points').
            gt_path: Ground truth future path.
            other_agents_futures: Ground truth futures of neighbors.
            k: Number of top predictions to consider (minADE@K).
            
        Returns:
            Dictionary of metrics.
        """
        if not predictions or not gt_path:
            return {}
            
        # Sort predictions by probability/score if available, else take first K
        sorted_preds = sorted(predictions, key=lambda x: x.get('probability', 0), reverse=True)
        top_k_preds = sorted_preds[:k]
        
        ades = []
        fdes = []
        off_road_flags = []
        collision_flags = []
        
        for pred in top_k_preds:
            path = pred['path_points']
            ades.append(self.compute_ade(path, gt_path))
            fdes.append(self.compute_fde(path, gt_path))
            off_road_flags.append(self.check_off_road(path))
            collision_flags.append(self.check_collision(path, other_agents_futures))
            
        # minADE / minFDE
        min_ade = min(ades) if ades else float('inf')
        min_fde = min(fdes) if fdes else float('inf')
        
        # For collision and off-road rate, we usually evaluate the "best" trajectory 
        # (the one closest to GT) or the "most likely" one.
        # Standard benchmarks often report "Miss Rate" based on minFDE.
        # Here we will report if the *best matching* trajectory (minFDE one) had a collision/off-road.
        
        best_idx = np.argmin(fdes) if fdes else 0
        is_collision = collision_flags[best_idx] if collision_flags else False
        is_off_road = off_road_flags[best_idx] if off_road_flags else False
        
        return {
            f'minADE@{k}': min_ade,
            f'minFDE@{k}': min_fde,
            'collision': is_collision,
            'off_road': is_off_road
        }
