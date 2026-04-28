import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import os
import streamlit as st
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionMethod(Enum):
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"

@dataclass
class DetectionResult:
    keypoints: np.ndarray
    confidence: float
    method: DetectionMethod
    landmarks: Optional[Dict] = None
    error: Optional[str] = None

class AdvancedPoseDetector:
    def __init__(self, config: Dict = None):
        """Initialize the advanced pose detector with multiple detection methods.
        
        Args:
            config: Configuration dictionary for detector settings
        """
        self.config = config or {
            'mediapipe': {
                'min_detection_confidence': 0.5,
                'min_tracking_confidence': 0.5,
                'model_complexity': 2
            },
            'openpose': {
                'model_folder': 'models',
                'number_people_max': 1
            }
        }
        
        # Initialize detectors
        self.mediapipe_detector = self._init_mediapipe()
        self.openpose_detector = self._init_openpose()
        
        # Initialize cache
        self.detection_cache = {}
        
    def _init_mediapipe(self):
        """Initialize MediaPipe Pose detector"""
        self.mediapipe_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _init_openpose(self):
        """OpenPose is invoked via subprocess (openpose.bin) in app.py — Python bindings are skipped to avoid protobuf version conflicts."""
        return None

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve detection."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            # Apply adaptive histogram equalization
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            
            return denoised
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
            
    def _detect_with_mediapipe(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Detect pose using MediaPipe."""
        try:
            if self.mediapipe_detector is None:
                return None
                
            # Process image
            results = self.mediapipe_detector.process(image)
            
            if not results.pose_landmarks:
                return None
                
            # Convert landmarks to keypoints
            keypoints = np.zeros((33, 3))
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y, landmark.z]
                
            # Calculate average confidence
            confidence = np.mean([landmark.visibility for landmark in results.pose_landmarks.landmark])
            
            return DetectionResult(
                keypoints=keypoints,
                confidence=confidence,
                method=DetectionMethod.MEDIAPIPE,
                landmarks=results.pose_landmarks
            )
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {e}")
            return None
            
    def _detect_with_openpose(self, image: np.ndarray) -> Optional[DetectionResult]:
        """Detect pose using OpenPose."""
        try:
            if self.openpose_detector is None:
                return None
            
            # Save image temporarily
            temp_path = "temp_pose.jpg"
            cv2.imwrite(temp_path, image)
            
            # Process image
            datum = op.Datum()
            datum.cvInputData = image
            self.openpose_detector.emplaceAndPop(op.VectorDatum([datum]))
            
            if datum.poseKeypoints is None:
                return None
            
            # Get keypoints for the first person
            keypoints = datum.poseKeypoints[0]
            
            # Calculate confidence
            confidence = np.mean(keypoints[:, 2])
            
            # Clean up
            os.remove(temp_path)
            
            return DetectionResult(
                keypoints=keypoints,
                confidence=confidence,
                method=DetectionMethod.OPENPOSE
            )
        except Exception as e:
            logger.error(f"OpenPose detection failed: {e}")
            return None
            
    def detect_pose(self, image: np.ndarray, use_cache: bool = True) -> DetectionResult:
        """Detect pose using multiple methods and return the best result.
        
        Args:
            image: Input image
            use_cache: Whether to use cached results
            
        Returns:
            DetectionResult containing the best detection
        """
        try:
            # Check cache
            if use_cache:
                image_hash = hash(image.tobytes())
                if image_hash in self.detection_cache:
                    return self.detection_cache[image_hash]
                    
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Try different detection methods
            results = []
            
            # Try MediaPipe first
            mediapipe_result = self._detect_with_mediapipe(processed_image)
            if mediapipe_result and mediapipe_result.confidence > 0.5:
                results.append(mediapipe_result)
            
            # Try OpenPose if MediaPipe failed or confidence is low
            if not results or results[0].confidence < 0.7:
                openpose_result = self._detect_with_openpose(processed_image)
                if openpose_result and openpose_result.confidence > 0.5:
                    results.append(openpose_result)
                
            # Select best result
            if results:
                best_result = max(results, key=lambda x: x.confidence)
                
                # Cache result
                if use_cache:
                    self.detection_cache[image_hash] = best_result
                    
                return best_result
            else:
                return DetectionResult(
                    keypoints=np.array([]),
                    confidence=0.0,
                    method=DetectionMethod.MEDIAPIPE,
                    error="No pose detected with any method"
                )
                
        except Exception as e:
            logger.error(f"Pose detection failed: {e}")
            return DetectionResult(
                keypoints=np.array([]),
                confidence=0.0,
                method=DetectionMethod.MEDIAPIPE,
                error=str(e)
            )
            
    def analyze_pose_quality(self, result: DetectionResult) -> Dict:
        """Analyze the quality of the detected pose.
        
        Args:
            result: DetectionResult from detect_pose
            
        Returns:
            Dictionary containing quality metrics
        """
        try:
            if result.error:
                return {"error": result.error}
                
            keypoints = result.keypoints
            
            # Calculate key measurements
            measurements = {
                'confidence': result.confidence,
                'keypoint_count': len(keypoints),
                'visible_keypoints': np.sum(keypoints[:, 2] > 0.5),
                'occlusion_ratio': 1 - (np.sum(keypoints[:, 2] > 0.5) / len(keypoints))
            }
            
            # Calculate pose stability
            if len(keypoints) >= 15:  # Minimum required keypoints
                # Calculate joint angles
                angles = self._calculate_joint_angles(keypoints)
                measurements['angles'] = angles
                
                # Calculate symmetry
                symmetry = self._calculate_pose_symmetry(keypoints)
                measurements['symmetry'] = symmetry
                
            return measurements
            
        except Exception as e:
            logger.error(f"Pose quality analysis failed: {e}")
            return {"error": str(e)}
            
    def _calculate_joint_angles(self, keypoints: np.ndarray) -> Dict:
        """Calculate angles between key joints."""
        try:
            angles = {}
            
            # Define joint connections
            connections = [
                ('shoulder', 'elbow', 'wrist'),  # Arm angles
                ('hip', 'knee', 'ankle'),        # Leg angles
                ('shoulder', 'hip', 'knee')      # Body angles
            ]
            
            # Calculate angles for each connection
            for start, mid, end in connections:
                if all(k in keypoints for k in [start, mid, end]):
                    v1 = keypoints[mid] - keypoints[start]
                    v2 = keypoints[end] - keypoints[mid]
                    angle = np.degrees(np.arccos(
                        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    ))
                    angles[f"{start}_{mid}_{end}"] = angle
                    
            return angles
            
        except Exception as e:
            logger.error(f"Joint angle calculation failed: {e}")
            return {}
            
    def _calculate_pose_symmetry(self, keypoints: np.ndarray) -> float:
        """Calculate pose symmetry score."""
        try:
            # Define symmetric pairs
            pairs = [
                ('left_shoulder', 'right_shoulder'),
                ('left_elbow', 'right_elbow'),
                ('left_hip', 'right_hip'),
                ('left_knee', 'right_knee')
            ]
            
            # Calculate symmetry for each pair
            symmetry_scores = []
            for left, right in pairs:
                if left in keypoints and right in keypoints:
                    # Calculate horizontal symmetry
                    left_x = keypoints[left][0]
                    right_x = keypoints[right][0]
                    center_x = (left_x + right_x) / 2
                    symmetry = 1 - abs((left_x - center_x) / center_x)
                    symmetry_scores.append(symmetry)
                    
            return np.mean(symmetry_scores) if symmetry_scores else 0.0
            
        except Exception as e:
            logger.error(f"Symmetry calculation failed: {e}")
            return 0.0

# --- Generate and play TTS for feedback in Text & Voice Feedback tab ---
def text_to_speech(text: str) -> str:
    # Implementation of text_to_speech function
    # This is a placeholder and should be replaced with the actual implementation
    return "path_to_generated_audio_file.mp3"

