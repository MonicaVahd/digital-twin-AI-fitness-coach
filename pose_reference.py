import os
import numpy as np
import json
import cv2
from typing import Dict, List, Tuple, Optional

class PoseReference:
    def __init__(self, base_path: str = "reference_poses"):
        self.base_path = base_path
        self.exercises = ["downdog", "goddess", "plank", "tree", "warrior2"]
        self._create_directory_structure()
        
        # Exercise-specific thresholds
        self.thresholds = {
            "downdog": {
                "spine_angle": [30, 45],  # Degrees from horizontal
                "hip_angle": [90, 100],   # Degrees
                "shoulder_angle": [170, 180]  # Degrees
            },
            "goddess": {
                "knee_angle": [90, 110],  # Degrees
                "hip_width": [1.5, 2.0],  # Ratio to shoulder width
                "spine_angle": [0, 10]    # Vertical alignment
            },
            "plank": {
                "spine_angle": [0, 15],   # Degrees from horizontal
                "elbow_angle": [85, 95],  # Degrees
                "hip_angle": [170, 180]   # Degrees
            },
            "tree": {
                "standing_knee": [170, 180],  # Standing leg straightness
                "foot_height": [0.3, 0.6],    # Ratio of leg length
                "hip_alignment": [0, 10]      # Hip level deviation
            },
            "warrior2": {
                "front_knee": [85, 95],    # Front knee angle
                "back_leg": [150, 170],    # Back leg straightness
                "arm_alignment": [170, 180] # Arm straightness
            }
        }
        self.visibility_threshold = 0.6
        
    def _create_directory_structure(self):
        """Create the necessary directory structure for storing reference poses"""
        for exercise in self.exercises:
            for category in ["correct", "incorrect"]:
                path = os.path.join(self.base_path, exercise, category)
                os.makedirs(path, exist_ok=True)
    
    def save_pose(self, exercise: str, keypoints: List[Tuple[float, float]], 
                 is_correct: bool, metadata: Dict = None):
        """Save a pose to the reference dataset"""
        category = "correct" if is_correct else "incorrect"
        path = os.path.join(self.base_path, exercise, category)
        
        # Create a unique filename based on timestamp
        filename = f"{exercise}_{category}_{len(os.listdir(path))}.json"
        
        data = {
            "keypoints": keypoints,
            "metadata": metadata or {}
        }
        
        with open(os.path.join(path, filename), "w") as f:
            json.dump(data, f)
    
    def get_reference_pose(self, exercise: str) -> Optional[List[Tuple[float, float]]]:
        """Get a reference pose for comparison"""
        path = os.path.join(self.base_path, exercise, "correct")
        if not os.path.exists(path):
            return None
            
        files = os.listdir(path)
        if not files:
            return None
            
        # Get the first reference pose (can be enhanced to select best match)
        with open(os.path.join(path, files[0]), "r") as f:
            data = json.load(f)
            return data["keypoints"]
    
    def compare_poses(self, current_pose: List[Tuple[float, float]], 
                     reference_pose: List[Tuple[float, float]], 
                     exercise_type: str = "plank") -> Dict:
        """
        Compare current pose with reference pose and return analysis
        Returns dict with angle differences and alignment scores
        """
        if len(current_pose) != len(reference_pose):
            return {"error": "Keypoint count mismatch"}
            
        # Calculate angle differences for key joints
        angles_current = self._calculate_angles(current_pose, exercise_type)
        angles_reference = self._calculate_angles(reference_pose, exercise_type)
        
        differences = {
            joint: abs(angles_current[joint] - angles_reference[joint])
            for joint in angles_current.keys()
        }
        
        # Calculate overall alignment score with weighted importance
        weights = {
            'spine': 1.0,
            'elbow': 0.8,
            'shoulder': 0.8,
            'hip': 1.0,
            'knee': 0.6,
            'ankle': 0.6
        }
        
        weighted_diffs = [
            (1 - (diff / 180)) * weights.get(joint, 1.0)
            for joint, diff in differences.items()
        ]
        alignment_score = sum(weighted_diffs) / sum(weights.values())
        
        return {
            'angle_differences': differences,
            'alignment_score': alignment_score,
            'feedback': self._generate_feedback(differences, exercise_type)
        }
    
    def _calculate_angles(self, keypoints: List[Tuple[float, float]], 
                         exercise_type: str = "plank") -> Dict[str, float]:
        """Calculate key angles in the pose"""
        angles = {}
        
        if exercise_type == "downdog":
            # Convert keypoints to numpy arrays for easier calculation
            if len(keypoints) > 14:
                r_shoulder = np.array(keypoints[2])
                r_elbow = np.array(keypoints[3])
                r_wrist = np.array(keypoints[4])
                l_shoulder = np.array(keypoints[5])
                l_elbow = np.array(keypoints[6])
                l_wrist = np.array(keypoints[7])
                
                # Calculate elbow angles
                angles['r_elbow'] = self._angle_between_points(r_shoulder, r_elbow, r_wrist)
                angles['l_elbow'] = self._angle_between_points(l_shoulder, l_elbow, l_wrist)
                angles['elbow'] = (angles['r_elbow'] + angles['l_elbow']) / 2
                
                # Calculate back angle
                mid_shoulder = (r_shoulder + l_shoulder) / 2
                mid_hip = (np.array(keypoints[9]) + np.array(keypoints[12])) / 2
                angles['back_angle'] = self._angle_to_vertical(mid_shoulder - mid_hip)

        elif exercise_type == "goddess":
            if len(keypoints) > 14:
                # Calculate front and back leg angles
                hip = np.array(keypoints[9])  # Right hip
                front_knee = np.array(keypoints[10])
                front_ankle = np.array(keypoints[11])
                back_hip = np.array(keypoints[12])  # Left hip
                back_knee = np.array(keypoints[13])
                back_ankle = np.array(keypoints[14])
                
                angles['front_knee'] = self._angle_between_points(hip, front_knee, front_ankle)
                angles['back_knee'] = self._angle_between_points(back_hip, back_knee, back_ankle)
                
                # Calculate hip depth
                hip_height = (hip[1] + back_hip[1]) / 2
                ankle_height = (front_ankle[1] + back_ankle[1]) / 2
                leg_length = np.linalg.norm(hip - front_knee) + np.linalg.norm(front_knee - front_ankle)
                angles['hip_depth'] = (hip_height - ankle_height) / leg_length

        elif exercise_type == "tree":
            if len(keypoints) > 14:
                # Calculate front and back leg angles
                hip = np.array(keypoints[9])  # Right hip
                front_knee = np.array(keypoints[10])
                front_ankle = np.array(keypoints[11])
                back_hip = np.array(keypoints[12])  # Left hip
                back_knee = np.array(keypoints[13])
                back_ankle = np.array(keypoints[14])
                
                angles['front_knee'] = self._angle_between_points(hip, front_knee, front_ankle)
                angles['back_knee'] = self._angle_between_points(back_hip, back_knee, back_ankle)
                
                # Calculate hip depth
                hip_height = (hip[1] + back_hip[1]) / 2
                ankle_height = (front_ankle[1] + back_ankle[1]) / 2
                leg_length = np.linalg.norm(hip - front_knee) + np.linalg.norm(front_knee - front_ankle)
                angles['hip_depth'] = (hip_height - ankle_height) / leg_length

        elif exercise_type == "warrior2":
            if len(keypoints) > 14:
                # Calculate front and back leg angles
                hip = np.array(keypoints[9])  # Right hip
                front_knee = np.array(keypoints[10])
                front_ankle = np.array(keypoints[11])
                back_hip = np.array(keypoints[12])  # Left hip
                back_knee = np.array(keypoints[13])
                back_ankle = np.array(keypoints[14])
                
                angles['front_knee'] = self._angle_between_points(hip, front_knee, front_ankle)
                angles['back_knee'] = self._angle_between_points(back_hip, back_knee, back_ankle)
                
                # Calculate hip depth
                hip_height = (hip[1] + back_hip[1]) / 2
                ankle_height = (front_ankle[1] + back_ankle[1]) / 2
                leg_length = np.linalg.norm(hip - front_knee) + np.linalg.norm(front_knee - front_ankle)
                angles['hip_depth'] = (hip_height - ankle_height) / leg_length

        elif exercise_type == "plank":
            # Convert keypoints to numpy arrays for easier calculation
            if len(keypoints) > 14:  # Ensure we have all necessary keypoints
                # Key points for plank
                nose = np.array(keypoints[0])
                neck = np.array(keypoints[1])
                r_shoulder = np.array(keypoints[2])
                l_shoulder = np.array(keypoints[5])
                r_elbow = np.array(keypoints[3])
                l_elbow = np.array(keypoints[6])
                r_wrist = np.array(keypoints[4])
                l_wrist = np.array(keypoints[7])
                r_hip = np.array(keypoints[9])
                l_hip = np.array(keypoints[12])
                r_knee = np.array(keypoints[10])
                l_knee = np.array(keypoints[13])
                r_ankle = np.array(keypoints[11])
                l_ankle = np.array(keypoints[14])
                
                # Calculate midpoints
                mid_shoulder = (r_shoulder + l_shoulder) / 2
                mid_hip = (r_hip + l_hip) / 2
                mid_knee = (r_knee + l_knee) / 2
                mid_ankle = (r_ankle + l_ankle) / 2
                
                # Calculate key angles
                # Spine angle (should be close to 0° for perfect plank)
                angles['spine'] = self._angle_to_horizontal(mid_shoulder - mid_hip)
                
                # Elbow angle (should be close to 90° for proper plank)
                angles['r_elbow'] = self._angle_between_points(r_shoulder, r_elbow, r_wrist)
                angles['l_elbow'] = self._angle_between_points(l_shoulder, l_elbow, l_wrist)
                angles['elbow'] = (angles['r_elbow'] + angles['l_elbow']) / 2
                
                # Hip angle (should be 180° for straight body)
                angles['hip'] = self._angle_between_points(mid_shoulder, mid_hip, mid_knee)
                
                # Knee angle (should be 180° for straight legs)
                angles['knee'] = self._angle_between_points(mid_hip, mid_knee, mid_ankle)
                
                # Head position (should be neutral with spine)
                angles['head'] = self._angle_to_horizontal(neck - nose)
                
                # Shoulder alignment (should be parallel to ground)
                angles['shoulder'] = self._angle_to_horizontal(r_shoulder - l_shoulder)
                
                # Calculate body height variation (should be minimal)
                hip_height = (r_hip[1] + l_hip[1]) / 2
                shoulder_height = (r_shoulder[1] + l_shoulder[1]) / 2
                angles['height_var'] = abs(hip_height - shoulder_height)
        
        return angles
    
    def _angle_between_points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _angle_to_horizontal(self, vector: np.ndarray) -> float:
        """Calculate angle between a vector and the horizontal"""
        horizontal = np.array([1, 0])
        cos_angle = np.dot(vector, horizontal) / (np.linalg.norm(vector) * np.linalg.norm(horizontal))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _angle_to_vertical(self, vector: np.ndarray) -> float:
        """Calculate angle between a vector and the vertical"""
        vertical = np.array([0, 1])
        cos_angle = np.dot(vector, vertical) / (np.linalg.norm(vector) * np.linalg.norm(vertical))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def _generate_feedback(self, differences: Dict[str, float], exercise_type: str) -> List[str]:
        """Generate feedback based on angle differences"""
        feedback = []
        
        if exercise_type == "downdog":
            if differences.get('back_angle') is not None:
                if differences['back_angle'] > self.thresholds['downdog']['spine_angle'][1]:
                    feedback.append("Keep your back straight, avoid sagging hips")
            
            if differences.get('elbow') is not None:
                if differences['elbow'] < self.thresholds['downdog']['elbow_angle']['start'][0]:
                    feedback.append("Extend your arms fully at the bottom")
                elif differences['elbow'] > self.thresholds['downdog']['elbow_angle']['top'][1]:
                    feedback.append("Curl the weights higher, aim for 90° at the top")
            
            if differences.get('back_angle') is not None:
                if differences['back_angle'] > self.thresholds['downdog']['spine_angle'][1]:
                    feedback.append("Keep your back straight, avoid sagging hips")
        
        elif exercise_type == "goddess":
            if differences.get('front_knee') is not None:
                if differences['front_knee'] < self.thresholds['goddess']['knee_angle'][0]:
                    feedback.append("Front knee should be at 90°")
                elif differences['front_knee'] > self.thresholds['goddess']['knee_angle'][1]:
                    feedback.append("Lower your front knee more")
            
            if differences.get('hip_depth') is not None:
                if differences['hip_depth'] < self.thresholds['goddess']['hip_width'][0]:
                    feedback.append("Lower your hips more for proper depth")
                elif differences['hip_depth'] > self.thresholds['goddess']['hip_width'][1]:
                    feedback.append("Don't drop your hips too low")
        
        elif exercise_type == "tree":
            if differences.get('front_knee') is not None:
                if differences['front_knee'] < self.thresholds['tree']['standing_knee'][0]:
                    feedback.append("Front knee should be at 90°")
                elif differences['front_knee'] > self.thresholds['tree']['standing_knee'][1]:
                    feedback.append("Lower your front knee more")
            
            if differences.get('hip_depth') is not None:
                if differences['hip_depth'] < self.thresholds['tree']['foot_height'][0]:
                    feedback.append("Lower your hips more for proper depth")
                elif differences['hip_depth'] > self.thresholds['tree']['foot_height'][1]:
                    feedback.append("Don't drop your hips too low")
        
        elif exercise_type == "warrior2":
            if differences.get('front_knee') is not None:
                if differences['front_knee'] < self.thresholds['warrior2']['front_knee'][0]:
                    feedback.append("Front knee should be at 90°")
                elif differences['front_knee'] > self.thresholds['warrior2']['front_knee'][1]:
                    feedback.append("Lower your front knee more")
            
            if differences.get('back_knee') is not None:
                if differences['back_knee'] < self.thresholds['warrior2']['back_leg'][0]:
                    feedback.append("Back knee should be at 90°")
            
            if differences.get('hip_depth') is not None:
                if differences['hip_depth'] < self.thresholds['warrior2']['hip_width'][0]:
                    feedback.append("Lower your hips more for proper depth")
                elif differences['hip_depth'] > self.thresholds['warrior2']['hip_width'][1]:
                    feedback.append("Don't drop your hips too low")
        
        elif exercise_type == "plank":
            if differences.get('spine') is not None:
                if differences['spine'] > self.thresholds['plank']['spine_angle'][1]:
                    feedback.append("Keep your back straight, avoid sagging hips")
            
            if differences.get('elbow') is not None:
                if differences['elbow'] < self.thresholds['plank']['elbow_angle'][0]:
                    feedback.append("Position your elbows directly under shoulders")
                elif differences['elbow'] > self.thresholds['plank']['elbow_angle'][1]:
                    feedback.append("Bring your elbows closer to your body")
            
            if differences.get('hip') is not None:
                if differences['hip'] < self.thresholds['plank']['hip_angle'][0]:
                    feedback.append("Straighten your hips to maintain proper alignment")
        
        if not feedback:
            feedback.append("Good form! Keep it up!")
        
        return feedback 