import json
import os
import numpy as np
from pose_reference import PoseReference
import streamlit as st

class FitnessDatasetProcessor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.pose_ref = PoseReference()
        
    def process_annotations(self, split: str = 'train'):
        """Process annotations file and extract pose information"""
        annotations_file = os.path.join(
            self.dataset_path,
            f"_annotations.{split}.jsonl"
        )
        
        if not os.path.exists(annotations_file):
            raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
            
        processed_poses = {
            "squat": {"correct": [], "incorrect": []},
            "plank": {"correct": [], "incorrect": []},
            "bicep_curl": {"correct": [], "incorrect": []},
            "lunge": {"correct": [], "incorrect": []}
        }
        
        with open(annotations_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Extract pose information and classify
                pose_data = self._extract_pose_data(data)
                if pose_data:
                    exercise_type = pose_data['exercise_type']
                    is_correct = pose_data['is_correct']
                    keypoints = pose_data['keypoints']
                    
                    category = "correct" if is_correct else "incorrect"
                    if exercise_type in processed_poses:
                        processed_poses[exercise_type][category].append(keypoints)
        
        return processed_poses
    
    def _extract_pose_data(self, annotation):
        """Extract pose keypoints and metadata from annotation"""
        try:
            # Extract exercise type from image path or metadata
            image_path = annotation.get('image_path', '')
            exercise_type = self._determine_exercise_type(image_path)
            
            # Extract keypoints from annotation
            keypoints = []
            if 'keypoints' in annotation:
                keypoints = self._convert_keypoints(annotation['keypoints'])
            
            # Determine if pose is correct
            is_correct = self._determine_correctness(annotation)
            
            return {
                'exercise_type': exercise_type,
                'is_correct': is_correct,
                'keypoints': keypoints
            }
        except Exception as e:
            print(f"Error processing annotation: {e}")
            return None
    
    def _determine_exercise_type(self, image_path: str) -> str:
        """Determine exercise type from image path or metadata"""
        # Implement logic to determine exercise type
        # This will need to be adjusted based on your dataset structure
        lower_path = image_path.lower()
        if 'squat' in lower_path:
            return 'squat'
        elif 'plank' in lower_path:
            return 'plank'
        elif 'curl' in lower_path:
            return 'bicep_curl'
        elif 'lunge' in lower_path:
            return 'lunge'
        return 'unknown'
    
    def _convert_keypoints(self, raw_keypoints):
        """Convert dataset keypoints to our format"""
        # Implement conversion logic based on your dataset format
        # This will need to be adjusted based on your keypoint format
        converted_keypoints = []
        for kp in raw_keypoints:
            if isinstance(kp, dict):
                x = kp.get('x', 0)
                y = kp.get('y', 0)
                converted_keypoints.append((x, y))
            elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                converted_keypoints.append((kp[0], kp[1]))
        return converted_keypoints
    
    def _determine_correctness(self, annotation):
        """Determine if the pose is correct based on annotation"""
        # Implement logic to determine if pose is correct
        # This will need to be adjusted based on your dataset labels
        return annotation.get('is_correct', False)
    
    def populate_reference_poses(self):
        """Process dataset and populate reference poses"""
        train_poses = self.process_annotations('train')
        valid_poses = self.process_annotations('valid')
        
        # Combine poses from both splits
        all_poses = train_poses.copy()
        for exercise in valid_poses:
            for category in valid_poses[exercise]:
                all_poses[exercise][category].extend(valid_poses[exercise][category])
        
        # Save reference poses
        for exercise in all_poses:
            for is_correct, poses in all_poses[exercise].items():
                for pose in poses:
                    self.pose_ref.save_pose(
                        exercise=exercise,
                        keypoints=pose,
                        is_correct=(is_correct == "correct")
                    )
        
        return len(all_poses) 

    def display_image(self, image_path):
        """Display an image from the dataset"""
        if not os.path.exists(image_path):
            st.error(f"Image not found: {image_path}")
            return
        
        st.image(image_path, use_container_width=True)