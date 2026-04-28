import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    # On platforms where pysqlite3-binary is unavailable (e.g., Windows),
    # fall back to the standard sqlite3 module.
    pass

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

import streamlit as st
import openai
import re
import json
import glob
import shutil
import time
import numpy as np
import cv2
from dotenv import load_dotenv
from crew import fitness_crew
from storage import store_ltm, store_stm, get_ltm, get_stm
from agents import get_llm
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import textwrap
import mediapipe as mp
from PIL import Image
import io
from gtts import gTTS
import tempfile
from openai import OpenAI
import base64
import traceback
import matplotlib.gridspec as gridspec
from pose_detection import AdvancedPoseDetector, DetectionResult
import subprocess
from voice_processor import VoiceProcessor
# from st_audiorec import st_audiorec
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import streamlit.components.v1 as components

# Declare the local custom component with the correct path
_component_func = components.declare_component(
    "st_audiorec",
    path=os.path.join(
        os.path.dirname(__file__),
        "streamlit_audio_recorder",
        "st_audiorec",
        "frontend",
        "build"
    )
)

def st_audiorec(key=None):
    return _component_func(key=key)

# st.title("Test Audio Recorder")
# audio_bytes = st_audiorec()

# if audio_bytes:
#     st.write(f"Type of audio_bytes: {type(audio_bytes)}")
#     # If it's a string, try to decode from base64
#     if isinstance(audio_bytes, str):
#         try:
#             audio_bytes = base64.b64decode(audio_bytes)
#         except Exception as e:
#             st.error(f"Failed to decode audio: {e}")
#     st.audio(audio_bytes, format='audio/wav')



# Initialize required directories
WORKSPACE_ROOT = os.getcwd()
TEMP_IMAGES = os.path.join(WORKSPACE_ROOT, "temp_images")
OUTPUT_JSON = os.path.join(WORKSPACE_ROOT, "output_json")
OUTPUT_IMAGES = os.path.join(WORKSPACE_ROOT, "output_images")
TEMP_FRAMES = os.path.join(WORKSPACE_ROOT, "temp_frames")
CORRECT_FORM_IMAGES = os.path.join(WORKSPACE_ROOT, "correct_form_images")

# Create all necessary directories
required_dirs = [
    TEMP_IMAGES,
    OUTPUT_JSON,
    OUTPUT_IMAGES,
    TEMP_FRAMES,
    CORRECT_FORM_IMAGES
]

for dir_path in required_dirs:
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Directory ready: {dir_path}")
    except Exception as e:
        print(f"⚠️ Error creating directory {dir_path}: {e}")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the advanced pose detector
pose_detector = AdvancedPoseDetector()

# Function to analyze user input and extract keypoints
def analyze_user_input(user_input, exercise_type=None):
    """Analyze user input and generate visualization with reference pose"""
    try:
        # Create timestamp for unique filenames
        timestamp = int(time.time())
        temp_image_path = os.path.join(TEMP_IMAGES, f"pose_{timestamp}.png")
        
        # Save uploaded image
        with open(temp_image_path, "wb") as f:
            f.write(user_input.getvalue())
        
        # Analyze pose using MediaPipe
        image = cv2.imread(temp_image_path)
        results = analyze_pose_mediapipe(temp_image_path)
        
        if results is None or not results.pose_landmarks:
            return {
                "error": "No pose detected in the image",
                "similarity_score": 0
            }
        
        # Convert landmarks to our format
        landmarks_dict = {
            'pose_landmarks': [
                {
                    'x': landmark.x,
                    'y': landmark.y,
                    'visibility': landmark.visibility
                }
                for landmark in results.pose_landmarks.landmark
            ]
        }
        
        # Get reference pose and calculate similarity
        if exercise_type:
            reference_image = get_pose_from_database(exercise_type)
            if reference_image is not None:
                ref_results = analyze_pose_mediapipe(reference_image)
                if ref_results and ref_results.pose_landmarks:
                    # Calculate similarity score
                    similarity_score = calculate_pose_similarity(
                        landmarks_dict['pose_landmarks'],
                        [{'x': l.x, 'y': l.y, 'visibility': l.visibility} 
                         for l in ref_results.pose_landmarks.landmark]
                    )
                else:
                    similarity_score = 0
            else:
                similarity_score = 0
        else:
            similarity_score = 0
        
        # Generate analysis text
        analysis_text = f"""
Pose Comparison Results

Similarity Score: {similarity_score:.2f}%

1. The pose matches the reference fairly well, but there are some noticeable differences in alignment.

2. Key differences in alignment include:
   - The hips are slightly higher than the shoulders, creating a slight angle instead of a straight line from head to heels.
   - The elbows are not directly under the shoulders, which can put unnecessary strain on the shoulders and wrists.

3. Specific adjustments needed:
   - Focus on engaging the core muscles more to bring the hips in line with the shoulders and create a straight line from head to heels.
   - Ensure the elbows are directly under the shoulders to maintain proper alignment and prevent strain on the shoulders and wrists.

4. Safety considerations:
   - It's important to maintain proper alignment to avoid strain on the lower back, shoulders, and wrists.
   - Engaging the core muscles and adjusting the alignment as mentioned above can help prevent injury and make the pose more effective.
"""
        
        # Create visualization
        visualization_path = visualize_pose_analysis(
            image,
            landmarks_dict,
            analysis_text,
            exercise_type
        )
        
        if visualization_path is None:
            return {
                "error": "Failed to create visualization",
                "similarity_score": similarity_score
            }
        
        # Clean up temporary files
        os.remove(temp_image_path)
        
        return {
            "visualization_path": visualization_path,
            "similarity_score": similarity_score,
            "analysis_text": analysis_text
        }
        
    except Exception as e:
        print(f"Error in analyze_user_input: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# Function to create an image showing the correct posture based on keypoints
def create_posture_image(input_data, output_path):
    """Create a pose image from input data"""
    try:
        print("\n=== Creating pose image ===")
        print(f"Input type: {type(input_data)}")
        
        # If input is already an image (numpy array)
        if isinstance(input_data, np.ndarray):
            print("Input is already an image array")
            if input_data.size == 0:
                print("Empty image array")
                return None
            print(f"Input array shape: {input_data.shape}, dtype: {input_data.dtype}")
            img = input_data
        else:
            # Convert input data to image
            print("Converting input data to image")
            img = np.array(input_data)
            if img.size == 0:
                print("Empty image array after conversion")
                return None
            print(f"Converted array shape: {img.shape}, dtype: {img.dtype}")
        
        # Ensure minimum dimensions for MediaPipe
        min_height, min_width = 480, 640
        height, width = img.shape[:2]
        
        if height < min_height or width < min_width:
            print(f"Resizing image from {width}x{height} to meet minimum dimensions")
            scale = max(min_width/width, min_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
            print(f"New dimensions: {new_width}x{new_height}")
        
        # Save the image
        cv2.imwrite(output_path, img)
        print(f"Image saved to: {output_path}")
        return img
        
    except Exception as e:
        print(f"Error creating posture image: {e}")
        traceback.print_exc()
        return None

def get_pose_from_database(exercise_type, user_keypoints=None):
    """Get reference pose image from database with pose similarity comparison"""
    try:
        print(f"\n=== Loading reference image for {exercise_type} ===")
        
        # Check cached reference image first
        cache_dir = os.path.join('output', 'reference_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{exercise_type}_best_match.jpg')
        
        if os.path.exists(cache_path) and not user_keypoints:
            print("Loading cached reference image...")
            return cv2.imread(cache_path)
        
        # Define dataset paths for your yoga poses
        dataset_paths = [
            os.path.join("DATASET", "TRAIN", exercise_type),
            os.path.join("DATASET", "TEST", exercise_type)
        ]
        
        best_match_image = None
        best_similarity = -1
        best_image_path = None
        
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                print(f"Dataset path not found: {dataset_path}")
                continue
            
            print(f"Searching in: {dataset_path}")
            for img_file in os.listdir(dataset_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(dataset_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                    
                    if user_keypoints is not None:
                        # Get reference pose keypoints
                        results = analyze_pose_mediapipe(img_path)
                        if results and results.pose_landmarks:
                            # Convert landmarks to keypoints format
                            ref_keypoints = []
                            for landmark in results.pose_landmarks.landmark:
                                ref_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                            ref_keypoints = np.array(ref_keypoints).reshape(-1, 3)
                            
                            # Calculate similarity
                            similarity = calculate_pose_similarity(user_keypoints, ref_keypoints)
                            print(f"Pose similarity score: {similarity:.3f}")
                            
                            # Update best match if this is better
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_image = img.copy()
                                best_image_path = img_path
                                print(f"New best match found! Score: {similarity:.3f}")
                                
                                # Save to cache immediately
                                cv2.imwrite(cache_path, img)
                                print(f"Cached reference image saved to: {cache_path}")
                                
                                # If similarity is very high (>85%), stop searching
                                if similarity > 0.85:
                                    print("Found excellent match! Stopping search.")
                                    return best_match_image
                    else:
                        # If no user keypoints provided, use first valid image and cache it
                        print("No user keypoints provided, using first valid image")
                        cv2.imwrite(cache_path, img)
                        return img
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    traceback.print_exc()
                    continue
        
        if best_match_image is not None:
            print(f"\nBest matching pose found: {best_image_path}")
            print(f"Similarity score: {best_similarity:.3f}")
            return best_match_image
        
        print("\nNo suitable reference images found in directories:", dataset_paths)
        return None
        
    except Exception as e:
        print(f"Error in get_pose_from_database: {e}")
        traceback.print_exc()
        return None

def generate_exercise_image(exercise_type, analysis_text=None, keypoints=None):
    """Generate comprehensive exercise visualization with feedback"""
    try:
        print("\n=== Generating Exercise Visualization ===")
        
        # Create figure with three sections
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid for layout
        gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
        
        # Left subplot for current pose
        ax1 = fig.add_subplot(gs[0, 0])
        if keypoints is not None:
            visualize_keypoints(keypoints, ax1)
        ax1.set_title("Your Pose", fontsize=16, color='white', pad=20)
        ax1.axis('off')
        
        # Right subplot for reference pose
        ax2 = fig.add_subplot(gs[0, 1])
        reference_image = get_pose_from_database(exercise_type)
        if reference_image is not None:
            ax2.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
        ax2.set_title("Reference Pose", fontsize=16, color='white', pad=20)
        ax2.axis('off')
        
        # Bottom section for analysis text
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        
        if analysis_text:
            # Format analysis text
            wrapped_text = textwrap.fill(analysis_text, width=100)
            ax3.text(0.05, 0.95, "Pose Analysis:", fontsize=14, color='white', 
                    fontweight='bold', transform=ax3.transAxes)
            ax3.text(0.05, 0.85, wrapped_text, fontsize=12, color='white',
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7))
        
        # Add exercise type and level
        plt.figtext(0.02, 0.98, f"Exercise: {exercise_type.title()}", 
                   fontsize=14, color='white')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save visualization
        output_path = f"output/pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        plt.savefig(output_path, format='jpg', dpi=300, 
                   facecolor='black', edgecolor='none', bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error generating exercise image: {e}")
        traceback.print_exc()
        return None

def get_correct_form_image(exercise_type, user_id):
    """
    Get or generate the correct form image for a given exercise.
    """
    # Create directories if they don't exist
    correct_forms_dir = "correct_form_images"
    reference_dir = "reference_images"
    os.makedirs(correct_forms_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    
    # Define the output path
    output_path = os.path.join(correct_forms_dir, f"correct_posture_{user_id}.png")
    
    try:
        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(15, 6))
        
        # Left subplot for the reference image
        ax1 = fig.add_subplot(121)
        
        # Generate or load the reference image
        reference_image_path = os.path.join(reference_dir, f"reference_{exercise_type}.png")
        
        # If reference image doesn't exist, generate it
        if not os.path.exists(reference_image_path):
            generated_image = generate_exercise_image(exercise_type)
            if generated_image:
                with open(reference_image_path, 'wb') as f:
                    f.write(generated_image)
        
        if os.path.exists(reference_image_path):
            img = plt.imread(reference_image_path)
            ax1.imshow(img)
        else:
            ax1.text(0.5, 0.5, f"Could not generate\nreference image for\n{exercise_type}",
                    ha='center', va='center', fontsize=10, wrap=True)
        
        ax1.axis('off')
        ax1.set_title('AI-Generated Reference Form', fontsize=14, pad=20)
        
        # Right subplot for instructions
        ax2 = fig.add_subplot(122)
        
        # Exercise-specific instructions
        if exercise_type == "plank":
            instructions = {
                "title": "Key Points for Perfect Plank",
                "points": [
                    "1. Straight line from head to heels",
                    "2. Shoulders directly above elbows",
                    "3. Core and glutes engaged",
                    "4. Neutral neck position",
                    "5. Feet hip-width apart"
                ],
                "mistakes": [
                    "• Sagging hips",
                    "• Raised buttocks",
                    "• Looking up/down",
                    "• Shoulders forward of elbows",
                    "• Collapsed shoulder blades"
                ]
            }
        else:
            instructions = {
                "title": f"Key Points for {exercise_type.title()}",
                "points": ["Please add specific instructions for this exercise"],
                "mistakes": ["Add common mistakes to avoid"]
            }
        
        # Add instructions to the right subplot
        ax2.text(0.5, 0.9, instructions["title"], 
                ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Add key points
        y_pos = 0.7
        for point in instructions["points"]:
            ax2.text(0.5, y_pos, point, ha='center', va='center', fontsize=12)
            y_pos -= 0.1
        
        # Add common mistakes
        ax2.text(0.5, y_pos-0.1, "Common Mistakes to Avoid:", 
                ha='center', va='center', fontsize=12, fontweight='bold')
        y_pos -= 0.2
        mistake_text = "\n".join(instructions["mistakes"])
        ax2.text(0.5, y_pos, mistake_text, ha='center', va='center', fontsize=10)
        
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"Error creating reference image: {e}")
        return None

def convert_keypoints_to_structured_text(landmarks_dict):
    """
    Convert MediaPipe landmarks to structured text description for model input.
    """
    if not landmarks_dict:
        return "No landmarks detected"

    try:
        text_description = []
        mp_pose = mp.solutions.pose.PoseLandmark
        
        # Custom landmark indices
        CUSTOM_NECK = 100
        CUSTOM_MID_HIP = 101
        
        # Calculate spine angle
        if (CUSTOM_NECK in landmarks_dict and 
            CUSTOM_MID_HIP in landmarks_dict):
            
            # Get neck and mid hip points
            neck = np.array([
                landmarks_dict[CUSTOM_NECK]['x'],
                landmarks_dict[CUSTOM_NECK]['y']
            ])
            mid_hip = np.array([
                landmarks_dict[CUSTOM_MID_HIP]['x'],
                landmarks_dict[CUSTOM_MID_HIP]['y']
            ])
            
            # Calculate spine angle
            spine_vector = neck - mid_hip
            vertical = np.array([0, -1])
            spine_angle = calculate_angle_between_vectors(spine_vector, vertical)
            text_description.append(f"Spine Alignment: {spine_angle:.1f}° from vertical")
            
            # Calculate shoulder alignment
            if (mp_pose.RIGHT_SHOULDER in landmarks_dict and 
                mp_pose.LEFT_SHOULDER in landmarks_dict):
                right_shoulder = np.array([
                    landmarks_dict[mp_pose.RIGHT_SHOULDER]['x'],
                    landmarks_dict[mp_pose.RIGHT_SHOULDER]['y']
                ])
                left_shoulder = np.array([
                    landmarks_dict[mp_pose.LEFT_SHOULDER]['x'],
                    landmarks_dict[mp_pose.LEFT_SHOULDER]['y']
                ])
                shoulder_vector = right_shoulder - left_shoulder
                horizontal = np.array([1, 0])
                shoulder_angle = calculate_angle_between_vectors(shoulder_vector, horizontal)
                text_description.append(f"Shoulder Level: {shoulder_angle:.1f}° from horizontal")
            
            # Calculate hip alignment
            if (mp_pose.RIGHT_HIP in landmarks_dict and 
                mp_pose.LEFT_HIP in landmarks_dict):
                right_hip = np.array([
                    landmarks_dict[mp_pose.RIGHT_HIP]['x'],
                    landmarks_dict[mp_pose.RIGHT_HIP]['y']
                ])
                left_hip = np.array([
                    landmarks_dict[mp_pose.LEFT_HIP]['x'],
                    landmarks_dict[mp_pose.LEFT_HIP]['y']
                ])
                hip_vector = right_hip - left_hip
                hip_angle = calculate_angle_between_vectors(hip_vector, horizontal)
                text_description.append(f"Hip Level: {hip_angle:.1f}° from horizontal")
        
        # Add visibility information
        text_description.append("\nLandmark Visibility:")
        for landmark_id, landmark in landmarks_dict.items():
            # Convert custom landmarks to readable names
            landmark_name = str(landmark_id)
            if landmark_id == CUSTOM_NECK:
                landmark_name = "NECK"
            elif landmark_id == CUSTOM_MID_HIP:
                landmark_name = "MID_HIP"
            text_description.append(f"{landmark_name}: {landmark['visibility']:.2f}")
    
    except Exception as e:
        print(f"Error in keypoint conversion: {e}")
        text_description.append(f"Error calculating measurements: {str(e)}")
    
    return "\n".join(text_description)

def calculate_angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in degrees"""
    try:
        # Convert to unit vectors
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        # Calculate dot product and angle
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        
        return angle
        
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0.0

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points in degrees"""
    try:
        # Create vectors from points
        v1 = p1 - p2
        v2 = p3 - p2
        
        return calculate_angle_between_vectors(v1, v2)
        
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return 0.0

def calculate_horizontal_alignment(point1, point2):
    """Calculate alignment relative to horizontal"""
    import math
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))

def analyze_pose_mediapipe(image_path):
    """Analyze posture using MediaPipe."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Failed to read image from {image_path}")
            return None

        # Use advanced detector
        result = pose_detector.detect_pose(image)
        
        if result.error:
            print(f"⚠️ Pose detection failed: {result.error}")
            return None
            
        # Get quality analysis
        quality = pose_detector.analyze_pose_quality(result)
        
        # Create visualization
        annotated_image = image.copy()
        if result.landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                result.landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2
                )
            )
            
        return result.keypoints, annotated_image, quality

    except Exception as e:
        print(f"Error in pose analysis: {str(e)}")
        return None

def calculate_angle_to_horizontal(vector):
    """Calculate angle between vector and horizontal in degrees."""
    try:
        horizontal = np.array([1, 0])
        return calculate_angle_between_vectors(vector, horizontal)
    except Exception as e:
        print(f"Error calculating horizontal angle: {e}")
        return 0.0

def calculate_pose_angles(landmarks):
    """
    Calculate important angles from pose landmarks.
    """
    angles = {}
    
    try:
        # Convert landmarks to numpy array for easier calculation
        points = np.array(landmarks).reshape(-1, 3)
        
        # Calculate spine angle (neck to hip)
        neck = points[11][:2]  # Neck landmark
        hip = points[23][:2]   # Hip landmark
        vertical = np.array([neck[0], 0])  # Vertical reference
        spine_angle = calculate_angle_between_vectors(neck - hip, vertical - hip)
        angles["Spine Alignment"] = spine_angle
        
        # Calculate shoulder alignment
        left_shoulder = points[11][:2]
        right_shoulder = points[12][:2]
        shoulder_angle = calculate_angle_to_horizontal(left_shoulder - right_shoulder)
        angles["Shoulder Level"] = shoulder_angle
        
        # Calculate hip alignment
        left_hip = points[23][:2]
        right_hip = points[24][:2]
        hip_angle = calculate_angle_to_horizontal(left_hip - right_hip)
        angles["Hip Level"] = hip_angle
        
        return angles
    except Exception as e:
        print(f"Error calculating angles: {e}")
        return {"Error": 0.0}

def visualize_pose_analysis(image, landmarks_dict, analysis_text, exercise_type=None):
    """Create a comprehensive visualization with user's pose and reference pose"""
    try:
        print("\n=== Creating pose comparison visualization ===")
        
        # Create figure with 2 columns
        plt.style.use('dark_background')  # Set style before creating figure
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1])

        # Left column - User's pose
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if landmarks_dict and 'pose_landmarks' in landmarks_dict:
            # Draw landmarks on user's pose
            for landmark in landmarks_dict['pose_landmarks']:
                x = int(landmark['x'] * image.shape[1])
                y = int(landmark['y'] * image.shape[0])
                if landmark['visibility'] > 0.2:
                    ax1.plot(x, y, 'go', markersize=5)  # Green dots for landmarks
                    
            # Draw connections between landmarks
            POSE_CONNECTIONS = [
                (11, 12), # Shoulders
                (11, 13), (13, 15), # Left arm
                (12, 14), (14, 16), # Right arm
                (11, 23), (12, 24), # Torso
                (23, 24), # Hips
                (23, 25), (25, 27), # Left leg
                (24, 26), (26, 28)  # Right leg
            ]
            
            for connection in POSE_CONNECTIONS:
                if len(landmarks_dict['pose_landmarks']) > max(connection):
                    start = landmarks_dict['pose_landmarks'][connection[0]]
                    end = landmarks_dict['pose_landmarks'][connection[1]]
                    if start['visibility'] > 0.2 and end['visibility'] > 0.2:
                        ax1.plot([start['x'] * image.shape[1], end['x'] * image.shape[1]],
                               [start['y'] * image.shape[0], end['y'] * image.shape[0]],
                               'g-', linewidth=2, alpha=0.7)
        
        ax1.axis('off')
        ax1.set_title("Your Pose", fontsize=14, color='white')

        # Right column - Reference pose from dataset
        ax2 = plt.subplot(gs[0, 1])
        if exercise_type:
            reference_image = get_pose_from_database(exercise_type)
            if reference_image is not None:
                ax2.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
                # Get landmarks for reference image
                ref_results = analyze_pose_mediapipe(reference_image)
                if ref_results and ref_results[0] is not None:  # Check first return value (keypoints)
                    ref_keypoints = ref_results[0]
                    # Draw reference pose landmarks
                    for i in range(0, len(ref_keypoints), 3):
                        x, y, conf = ref_keypoints[i:i+3]
                        if conf > 0.2:
                            ax2.plot(x * reference_image.shape[1], 
                                   y * reference_image.shape[0], 
                                   'bo', markersize=5)  # Blue dots for reference
        ax2.axis('off')
        ax2.set_title("Reference Pose", fontsize=14, color='white')

        # Bottom row - Analysis text
        ax3 = plt.subplot(gs[1, :])
        ax3.axis('off')
        
        # Format analysis text
        wrapped_text = textwrap.fill(analysis_text, width=100)
        ax3.text(0.5, 0.5, wrapped_text,
                ha='center', va='center',
                fontsize=10, color='white',
                wrap=True)

        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_IMAGES, exist_ok=True)
        
        # Save visualization with timestamp
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_IMAGES, f"analysis_{timestamp}.png")
        
        # Save with white text visible
        plt.savefig(output_path,
                   format='png',
                   bbox_inches='tight',
                   facecolor='black',
                   edgecolor='none',
                   dpi=300)
        plt.close()

        print(f"Visualization saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()
        return None

def visualize_keypoints(keypoints, ax):
    """
    Visualize keypoints with improved error handling and visualization.
    """
    try:
        # Validate input
        if not isinstance(keypoints, (list, np.ndarray)):
            raise ValueError("Invalid keypoint format")
            
        # Convert to numpy array if needed
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints)
            
        # Ensure we have 3D data (x, y, confidence)
        if len(keypoints.shape) != 2 or keypoints.shape[1] != 3:
            raise ValueError("Keypoints must be in (N, 3) format")
            
        # Extract coordinates and confidences
        x_coords = keypoints[:, 0]
        y_coords = keypoints[:, 1]
        confidences = keypoints[:, 2]
        
        # Clear previous plot
        ax.clear()
        
        # Define connections for visualization
        connections = [
            (0, 1),   # Nose to Neck
            (1, 2),   # Neck to RShoulder
            (2, 3),   # RShoulder to RElbow
            (3, 4),   # RElbow to RWrist
            (1, 5),   # Neck to LShoulder
            (5, 6),   # LShoulder to LElbow
            (6, 7),   # LElbow to LWrist
            (1, 8),   # Neck to MidHip
            (8, 9),   # MidHip to RHip
            (9, 10),  # RHip to RKnee
            (10, 11), # RKnee to RAnkle
            (8, 12),  # MidHip to LHip
            (12, 13), # LHip to LKnee
            (13, 14)  # LKnee to LAnkle
        ]
        
        # Plot connections with confidence-based coloring
        for start_idx, end_idx in connections:
            if (start_idx < len(keypoints) and 
                end_idx < len(keypoints) and 
                confidences[start_idx] > 0.2 and 
                confidences[end_idx] > 0.2):
                
                confidence = min(confidences[start_idx], confidences[end_idx])
                color = plt.cm.YlOrRd(confidence)
                
                ax.plot([x_coords[start_idx], x_coords[end_idx]],
                       [y_coords[start_idx], y_coords[end_idx]],
                       color=color, linewidth=2, alpha=0.7)
        
        # Plot keypoints with confidence-based coloring
        scatter = ax.scatter(x_coords, y_coords,
                           c=confidences,
                           cmap='YlOrRd',
                           s=100,
                           alpha=0.8)
        
        # Add keypoint labels with improved visibility
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.2:
                ax.annotate(f'{i}',
                          (x, y),
                          xytext=(5, 5),
                          textcoords='offset points',
                          fontsize=8,
                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Customize plot
        ax.set_title('Pose Analysis', pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add confidence colorbar
        plt.colorbar(scatter, ax=ax, label='Confidence')
        
        # Set aspect ratio to equal for better visualization
        ax.set_aspect('equal')
        
        # Add legend for keypoint types
        legend_elements = [
            plt.Line2D([0], [0], color='r', marker='o', label='High Confidence',
                      markerfacecolor='r', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], color='yellow', marker='o', label='Medium Confidence',
                      markerfacecolor='yellow', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], color='b', marker='o', label='Low Confidence',
                      markerfacecolor='b', markersize=8, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
    except Exception as e:
        print(f"Error in visualize_keypoints: {e}")
        ax.text(0.5, 0.5,
                f"Error visualizing keypoints:\n{str(e)}",
                ha='center', va='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.8))
        ax.axis('off')

def log_system_status(user_id, memory_type, action, data):
    """Log system status for monitoring"""
    print(f"\n{'='*20} System Status {'='*20}")
    print(f"🔍 Action: {action}")
    print(f"👤 User ID: {user_id}")
    print(f"💾 Memory Type: {memory_type}")
    print(f"📊 Data: {data[:200]}..." if isinstance(data, str) else f"📊 Data: {data}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55 + "\n")

# Initialize OpenPose Paths (configure OPENPOSE_DIR in .env)
OPENPOSE_DIR = os.getenv("OPENPOSE_DIR", "/mnt/d/projects/openpose")
OPENPOSE_MODEL_DIR = os.path.join(OPENPOSE_DIR, "models")
OPENPOSE_BIN = os.path.join(OPENPOSE_DIR, "build", "examples", "openpose", "openpose.bin")
OUTPUT_JSON = os.path.join(OPENPOSE_DIR, "output_json")
OUTPUT_IMAGES = os.path.join(OPENPOSE_DIR, "output_images")
TEMP_IMAGES = os.path.join(OPENPOSE_DIR, "temp_images")
TEMP_FRAMES = os.path.join(OPENPOSE_DIR, "temp_frames")

# Create necessary directories if they don't exist
for directory in [OUTPUT_JSON, OUTPUT_IMAGES, TEMP_IMAGES, TEMP_FRAMES]:
    os.makedirs(directory, exist_ok=True)

def convert_to_wsl_path(windows_path):
    """Convert Windows path to WSL path"""
    return windows_path.replace('\\', '/').replace('D:', '/mnt/d')

def format_workout_history(history):
    """Format workout history for display"""
    if not history or isinstance(history, str):
        return "No previous history"
    
    formatted = "**Previous Sessions:**\n\n"
    for entry in history:
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        formatted += f"- {timestamp}: {entry['data']}\n"
    return formatted

def validate_user_id(user_id):
    """
    Validate user ID and check if user exists in storage
    Returns: (is_valid, status_message, is_new_user)
    """
    if not user_id or len(user_id.strip()) < 4:
        log_system_status(user_id, "both", "Validation", "Invalid user ID (too short)")
        return False, "User ID must be at least 4 characters long", False
    
    # Check if user exists in storage
    existing_data = get_ltm(user_id)
    is_new_user = (
        not existing_data or 
        existing_data == "No previous history" or
        (isinstance(existing_data, list) and len(existing_data) == 0)
    )
    
    log_system_status(
        user_id, 
        "LTM", 
        "Validation", 
        f"{'New user (no history found)' if is_new_user else 'Existing user with history'}"
    )
    print(f"\n=== User Validation Debug ===")
    print(f"User ID: {user_id}")
    print(f"Existing Data Type: {type(existing_data)}")
    print(f"Existing Data: {existing_data}")
    print(f"Is New User: {is_new_user}")
    print("===========================\n")

    return True, "New user" if is_new_user else "Existing user", is_new_user

def store_user_data(user_id, data, memory_type="both", level=None):
    """Store user data with level information"""
    timestamp = datetime.now().isoformat()
    entry = {
        'timestamp': timestamp,
        'data': data,
        'level': level
    }
    
    log_system_status(user_id, memory_type, "Storage", f"Storing data for {level} level user")
    
    if memory_type in ["both", "ltm"]:
        store_ltm(user_id, entry)
    if memory_type in ["both", "stm"]:
        store_stm(user_id, f"Latest activity ({level} level): {data[:100]}...")    

def update_user_level(user_id, new_level):
    """Update and persist user level"""
    st.session_state.user_level = new_level
    st.session_state.last_user_id = user_id
    store_user_data(user_id, f"User level updated to {new_level}", "ltm", new_level)

def get_user_level(user_id):
    """Get user level from storage or return default"""
    if not user_id:  # Add safety check for empty user_id
        print("⚠️ WARNING: Empty user_id passed to get_user_level")
        return "beginner"

    history = get_ltm(user_id)
    if isinstance(history, list):
        for entry in reversed(history):
            if isinstance(entry, dict) and 'level' in entry:
                level = entry['level']
                if level:  # Make sure level is not None or empty
                    return level
    print(f"⚠️ WARNING: No level found in history for user {user_id}, returning default")
    return "beginner"

def speech_to_text(audio):
    if audio is None:
        return ""
    with open(audio, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text

def get_latest_json():
    json_files = sorted(glob.glob(os.path.join(OUTPUT_JSON, "*.json")), key=os.path.getmtime, reverse=True)
    return json_files[0] if json_files else None

def extract_feedback_markdown(ai_response):
    try:
        if hasattr(ai_response, 'content'):
            feedback = ai_response.content.strip()
        elif isinstance(ai_response, dict) and "content" in ai_response:
            feedback = ai_response["content"].strip()
        elif isinstance(ai_response, str):
            feedback = ai_response.strip()
        else:
            feedback = "**⚠️ No valid feedback received from AI.**"
        
        return feedback if feedback else "**⚠️ No structured feedback received from AI.**"
    except Exception as e:
        print(f"Error in extract_feedback_markdown: {e}")
        return "**⚠️ Error processing AI feedback.**"

def extract_frames(video_path, max_frames=10):
    frames = []
    frame_paths = []
    timestamp = int(time.time())
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                frame_path = os.path.join(TEMP_FRAMES, f"frame_{timestamp}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                frames.append(frame)
                frame_paths.append(frame_path)
                
                if len(frames) >= max_frames:
                    break
                    
            frame_count += 1
            
        cap.release()
        return frame_paths
    except Exception as e:
        st.error(f"Error extracting video frames: {str(e)}")
        return []

def analyze_pose_openpose(image_path):
    """Analyze posture image using OpenPose"""
    # Convert paths to WSL format
    wsl_openpose_bin = convert_to_wsl_path(OPENPOSE_BIN)
    wsl_model_dir = convert_to_wsl_path(OPENPOSE_MODEL_DIR)
    wsl_output_json = convert_to_wsl_path(OUTPUT_JSON)
    wsl_output_images = convert_to_wsl_path(OUTPUT_IMAGES)
    wsl_image_path = convert_to_wsl_path(image_path)

    # Check if OpenPose is properly installed and configured
    if not os.path.exists(wsl_openpose_bin):
        st.error("⚠️ OpenPose binary not found. Please ensure OpenPose is properly installed.")
        return "OpenPose binary not found."

    if not os.path.exists(wsl_model_dir):
        st.error("⚠️ OpenPose models not found. Please check the models directory.")
        return "OpenPose models not found."

    try:
        # Prepare command with WSL paths
        command = f'{wsl_openpose_bin} --image_dir "{os.path.dirname(wsl_image_path)}" --write_json "{wsl_output_json}" --write_images "{wsl_output_images}" --display 0 --render_pose 1 --model_folder "{wsl_model_dir}"'
        
        print(f"📋 Executing OpenPose command:\n{command}")

        # Set LD_LIBRARY_PATH so the binary finds its shared libraries at runtime
        openpose_env = os.environ.copy()
        openpose_env["LD_LIBRARY_PATH"] = (
            f"{OPENPOSE_DIR}/build/src/openpose:"
            f"{OPENPOSE_DIR}/build/caffe/lib:"
            "/usr/local/lib:"
            + openpose_env.get("LD_LIBRARY_PATH", "")
        )
        
        # Execute OpenPose command
        result = subprocess.run(command, shell=True, check=True, env=openpose_env,
                                capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"OpenPose stderr: {e.stderr}")
        print(f"OpenPose stdout: {e.stdout}")
        st.error(f"⚠️ OpenPose command failed: {e.stderr[:300] if e.stderr else 'no output'}")
        return "OpenPose execution failed."

def determine_user_level(text, user_id):
    """Analyze user input to determine their experience level"""
    if not text or not isinstance(text, str):
        log_system_status(user_id, "both", "Level Determination", "No valid text provided")
        return "beginner"  # Default if no valid text
        
    text = text.lower()
    
    beginner_keywords = ['beginner', 'new', 'start', 'novice', 'basic', 'never', 'first time']
    intermediate_keywords = ['intermediate', 'some experience', 'familiar', 'moderate']
    advanced_keywords = ['advanced', 'expert', 'experienced', 'professional', 'athlete']
    
    log_system_status(user_id, "STM", "Level Detection", f"Analyzing text: {text[:100]}...")
    
    for keyword in advanced_keywords:
        if keyword in text:
            print(f"✓ Found advanced keyword: '{keyword}'")
            return "advanced"
            
    for keyword in intermediate_keywords:
        if keyword in text:
            print(f"✓ Found intermediate keyword: '{keyword}' in text: '...{text[max(0, text.find(keyword)-20):text.find(keyword)+20]}...'")
            return "intermediate"
            
    for keyword in beginner_keywords:
        if keyword in text:
            print(f"✓ Found beginner keyword: '{keyword}'")
            return "beginner"
    
    print("No specific level keywords found, defaulting to beginner")
    return "beginner"

def process_video(video_file, user_id, llm_provider, user_level="beginner"):
    """Process video file and generate feedback"""
    try:
        log_system_status(user_id, "both", "Video Processing", f"Starting video processing for {user_level} level user")
        timestamp = int(time.time())
        video_path = os.path.join(TEMP_IMAGES, f"video_{timestamp}.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getvalue())
            
        frame_paths = extract_frames(video_path)
        if not frame_paths:
            return "**⚠️ No frames could be extracted from the video.**"
            
        analyses = []
        # Add progress bar here
        progress_bar = st.progress(0)
        st.info("🎥 Analyzing video frames...")

        for i, frame_path in enumerate(frame_paths):
            # Update progress bar
            progress_bar.progress((i + 1) / len(frame_paths))       
            st.caption(f"Processing frame {i+1} of {len(frame_paths)}")    
            
            result = generate_workout_with_level(
                user_id,
                "", None, "", None,
                frame_path,
                llm_provider,
                user_level  
            )
            analyses.append(f"### Frame {i+1}\n{result}")
            
        # Clear progress bar
        progress_bar.empty()
        st.success("✅ Video analysis complete!")

        final_output = "\n\n".join(analyses)
        
        # Use store_user_data instead of direct storage calls
        store_user_data(
            user_id,
            f"Video Analysis: {len(frame_paths)} frames analyzed",
            "both",
            user_level
        )
        
        return final_output
        
    except Exception as e:
        return f"**⚠️ Error processing video: {str(e)}**"
        
    finally:
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        for frame_path in frame_paths if 'frame_paths' in locals() else []:
            if os.path.exists(frame_path):
                os.remove(frame_path)

def generate_workout_with_level(user_id, workout_prefs_text=None, workout_prefs_audio=None, feedback_text=None, feedback_audio=None, posture_image=None, llm_provider=None, user_level=None):
    """Modified version of generate_workout that accepts an explicit user level parameter"""
    log_system_status(user_id, "both", "Workout Generation", "Starting workout generation")
    print("\n" + "="*50)
    print("🚀 Starting Workout Generation Process")
    print("="*50)
    
    start_time = time.time()
    print(f"👤 User ID: {user_id}")
    print(f"🤖 Selected LLM Provider: {llm_provider}")
    print(f"👥 Explicit User Level: {user_level}")

    db_level = get_user_level(user_id)
    if db_level and db_level != user_level:
        print(f"⚠️ WARNING: Level mismatch! Provided: {user_level}, Database: {db_level}")
        print(f"⚠️ Overriding with database level: {db_level}")
        user_level = db_level

    # Log input processing
    print("\n📝 Processing Inputs:")
    workout_prefs_from_audio = ""
    feedback_from_audio = ""
    
    if workout_prefs_audio:
        print("- Converting workout preferences audio to text...")
        workout_prefs_from_audio = speech_to_text(workout_prefs_audio)
        print(f"  Audio transcription: {workout_prefs_from_audio}")
    
    if feedback_audio:
        print("- Converting feedback audio to text...")
        feedback_from_audio = speech_to_text(feedback_audio)
        print(f"  Audio transcription: {feedback_from_audio}")

    final_workout_prefs = f"{workout_prefs_text.strip()} {workout_prefs_from_audio.strip()}".strip()
    final_feedback = f"{feedback_text.strip()} {feedback_from_audio.strip()}".strip()

    print("\n💾 Storing User Information:")
    print(f"- Workout Preferences: {final_workout_prefs}")
    print(f"- Real-time Feedback: {final_feedback}")
    print(f"- Using User Level: {user_level}")

    store_ltm(user_id, final_workout_prefs)
    store_stm(user_id, final_feedback)

    print("\n📚 Retrieving User History:")
    user_history = get_ltm(user_id)
    live_feedback = get_stm(user_id)
    print(f"- Long-term Memory: {user_history}")
    print(f"- Short-term Memory: {live_feedback}")

    print("\n🤖 Initializing AI Model:")
    selected_llm = get_llm(llm_provider)
    print(f"- Using {llm_provider} model")

    st.info("🟢 Running Multi-Agent System...")
    print("\n🔄 Starting Multi-Agent Analysis")

    pose_data = None
    if posture_image:
        print("- Analyzing posture image with OpenPose...")
        pose_data = analyze_pose_openpose(posture_image)
        print(f"- Pose analysis complete: {'Keypoints detected' if isinstance(pose_data, list) else 'No valid keypoints'}")
        print(f"Debug - pose_data: {pose_data}") 

        # Check if pose_data is valid before calling feedback_agent
        if isinstance(pose_data, list) and len(pose_data) > 0:
            try:
                # Call feedback_agent with the keypoints
                feedback_result = feedback_agent(pose_data, user_id)
                if 'image_path' in feedback_result:
                    print(f"✓ Posture image generated at: {feedback_result['image_path']}")
                    
                    # Generate analysis using the LLM
                    print("\n📊 Preparing Pose Analysis Input")
                    try:
                        llm_input = (
                            f"Analyze the following pose keypoints for a {user_level} level user:\n"
                            f"{json.dumps(pose_data)[:1000]}...\n"
                            f"Provide structured feedback under these sections:\n"
                            f"- **Posture Overview:** Summarize general alignment.\n"
                            f"- **Key Misalignments:** List detected issues.\n"
                            f"- **Correction Tips:** Provide {'detailed technical' if user_level == 'advanced' else 'basic'} fixes.\n"
                        )
                    
                        print("Waiting for AI response...")
                        raw_ai_response = selected_llm.invoke(llm_input)
                        ai_feedback = extract_feedback_markdown(raw_ai_response)
                        
                        return {
                            "image_path": feedback_result['image_path'],
                            "message": "Posture image generated successfully.",
                            "analysis": ai_feedback
                        }
                    except Exception as e:
                        print(f"Error generating analysis: {e}")
                        return {
                            "image_path": feedback_result['image_path'],
                            "message": "Posture image generated, but analysis failed.",
                            "analysis": "**⚠️ Error generating analysis.**"
                        }
                else:
                    print("⚠️ No image path in feedback result")
                    return {"message": feedback_result.get('message', 'Error generating posture image')}
            except Exception as e:
                print(f"Error in feedback generation: {e}")
                return {"message": f"Error in feedback generation: {str(e)}"}
        else:
            print("⚠️ No valid pose data to analyze.")
            return {"message": "No keypoints detected."}

    if isinstance(pose_data, list) and len(pose_data) > 0:
        print("\n📊 Preparing Pose Analysis Input")
        llm_input = (
            f"Analyze the following pose keypoints for a {user_level} level user:\n"
            f"{json.dumps(pose_data)[:1000]}...\n"
            f"Provide structured feedback under these sections:\n"
            f"- **Posture Overview:** Summarize general alignment.\n"
            f"- **Key Misalignments:** List detected issues.\n"
            f"- **Correction Tips:** Provide {'detailed technical' if user_level == 'advanced' else 'basic'} fixes.\n"
        )
    else:
        print("\n📝 Preparing Text Analysis Input")
        
        # Customize prompts based on user level
        if user_level == "beginner":
            prompt_sections = (
                "- **Overview:** Analyze the user's goals and provide beginner-friendly guidance\n"
                "- **Starting Plan:** Suggest basic exercises with proper form instructions\n"
                "- **Safety Tips:** Provide essential safety guidelines for beginners\n"
                "- **Progress Timeline:** Outline realistic milestones for the first few months\n"
            )
        elif user_level == "intermediate":
            prompt_sections = (
                "- **Overview:** Analyze the user's current routine and goals\n"
                "- **Optimization:** Suggest ways to improve existing workout routine\n"
                "- **Advanced Techniques:** Introduce more challenging variations\n"
                "- **Progress Metrics:** Define measurable performance indicators\n"
            )
        else:  # advanced
            prompt_sections = (
                "- **Advanced Analysis:** Deep dive into the user's specialized goals\n"
                "- **Performance Optimization:** Suggest advanced training techniques\n"
                "- **Periodization:** Provide detailed training cycle recommendations\n"
                "- **Recovery Strategies:** Advanced recovery and maintenance protocols\n"
            )

        llm_input = (
            f"Based on the following information for a {user_level} level user:\n"
            f"Workout Preferences: {final_workout_prefs}\n"
            f"User Feedback: {final_feedback}\n\n"
            f"Provide structured feedback under these sections:\n{prompt_sections}"
        )

    print("\n🤖 Sending Request to AI Model")
    print(f"Input prompt:\n{llm_input}\n")

    try:
        print("Waiting for AI response...")
        raw_ai_response = selected_llm.invoke(llm_input)
        print("\n✅ AI Response Received")
        print(f"Raw response type: {type(raw_ai_response)}")
        
        with st.expander("Debug Information", expanded=False):
            st.write("🔍 DEBUG: Raw AI Response:", raw_ai_response)
            st.write("Response type:", type(raw_ai_response))
            st.write("Response structure:", vars(raw_ai_response) if hasattr(raw_ai_response, '__dict__') else raw_ai_response)
        
        ai_feedback = extract_feedback_markdown(raw_ai_response)
        print("\n📋 Processed Feedback:")
        print(ai_feedback)
        
    except Exception as e:
        print(f"\n❌ Error during AI processing: {e}")
        st.error(f"⚠️ LLM Error: {e}")
        ai_feedback = "**⚠️ AI was unable to process the request.**"

    processing_time = round(time.time() - start_time, 2)
    print(f"\n⏱️ Total Processing Time: {processing_time} seconds")

    # Safety check for None values, but don't hardcode any specific user ID
    if user_level is None:
        # Instead of a hardcoded override, get a fresh copy from the database
        db_level = get_user_level(user_id)
        if db_level:
            user_level = db_level
            print(f"⚠️ SAFETY CHECK: Recovered level '{user_level}' from database for user {user_id}")
        else:
            user_level = "beginner"  # Default to beginner if level is None
            print(f"⚠️ WARNING: Level was None for user {user_id}, defaulting to beginner")

    final_output = f"""
# {user_level.title()} Level Workout Analysis

{ai_feedback}

---
⏳ Processing Time: {processing_time} seconds
""".strip()

    print("\n" + "="*50)
    print("✅ Workout Generation Complete")
    print("="*50 + "\n")

    return final_output

def process_keypoints(keypoints):
    """Process and validate keypoints"""
    try:
        print("\n=== Processing keypoints ===")
        print(f"Input type: {type(keypoints)}")
        
        # Convert input to numpy array if it's not already
        if isinstance(keypoints, (list, tuple)):
            print("Converting input to numpy array")
            try:
                # Explicitly convert to float32
                keypoints = np.array(keypoints, dtype=np.float32)
                print(f"Converted to array with dtype: {keypoints.dtype}")
            except Exception as e:
                print(f"Error converting to numpy array: {e}")
                return None
        elif not isinstance(keypoints, np.ndarray):
            print(f"Error: Input keypoints must be a list, tuple, or numpy array, got {type(keypoints)}")
            return None
            
        # Ensure numeric type
        if not np.issubdtype(keypoints.dtype, np.number):
            print(f"Converting dtype from {keypoints.dtype} to float32")
            try:
                keypoints = keypoints.astype(np.float32)
            except Exception as e:
                print(f"Error converting dtype: {e}")
            return None
            
        print(f"Initial shape: {keypoints.shape}, dtype: {keypoints.dtype}")
        
        # Handle different input shapes
        if len(keypoints.shape) == 1:
            print("Reshaping 1D array")
            # If we have a flat array of coordinates
            if keypoints.size % 2 == 0:  # Even number of elements
                keypoints = keypoints.reshape(-1, 2)
            elif keypoints.size % 3 == 0:  # Triplets (x,y,confidence)
                keypoints = keypoints.reshape(-1, 3)
            else:
                print(f"Error: Invalid number of elements: {keypoints.size}")
                return None
        elif len(keypoints.shape) == 2:
            if keypoints.shape[1] == 2:  # (N,2) format
                print("Adding confidence values")
                # Add confidence values of 1.0
                conf = np.ones((keypoints.shape[0], 1), dtype=np.float32)
                keypoints = np.hstack((keypoints, conf))
            elif keypoints.shape[1] != 3:  # Not (N,3) format
                print(f"Error: Expected 2 or 3 values per point, got {keypoints.shape[1]}")
                return None
        else:
            print(f"Error: Invalid dimensions: {len(keypoints.shape)}")
            return None
            
        print(f"Processed shape: {keypoints.shape}, dtype: {keypoints.dtype}")
        
        # Validate the processed keypoints
        if keypoints.shape[0] < 1:
            print("Error: No keypoints found")
            return None
            
        # Check for invalid values
        if not np.all(np.isfinite(keypoints)):
            print("Error: Keypoints contain invalid values (inf/nan)")
            # Try to clean up invalid values
            keypoints = np.nan_to_num(keypoints, nan=0.0, posinf=1.0, neginf=0.0)
            print("Cleaned up invalid values")
            
        # Ensure all values are within reasonable bounds
        if np.any(np.abs(keypoints[:, :2]) > 1e6):
            print("Warning: Very large coordinate values detected, normalizing")
            # Normalize to 0-1 range
            for i in range(2):
                min_val = np.min(keypoints[:, i])
                max_val = np.max(keypoints[:, i])
                if max_val > min_val:
                    keypoints[:, i] = (keypoints[:, i] - min_val) / (max_val - min_val)
            print("Normalized coordinates to 0-1 range")
            
        print(f"Final keypoints shape: {keypoints.shape}, dtype: {keypoints.dtype}")
        print(f"Number of keypoints: {len(keypoints)}")
        return keypoints
        
    except Exception as e:
        print(f"Error processing keypoints: {e}")
        traceback.print_exc()
        return None

def convert_openpose_to_mediapipe(keypoints):
    """Convert OpenPose keypoints to MediaPipe format"""
    try:
        print("\n=== Converting keypoints to MediaPipe format ===")
        print(f"Debug - keypoints type: {type(keypoints)}")
        print(f"Debug - keypoints shape: {keypoints.shape if isinstance(keypoints, np.ndarray) else len(keypoints)}")
        
        # Ensure keypoints is a numpy array
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        
        # If we have a single point or flat array, reshape it
        if len(keypoints.shape) == 1:
            if len(keypoints) % 3 == 0:
                keypoints = keypoints.reshape(-1, 3)
            elif len(keypoints) % 2 == 0:
                # Add confidence values of 1.0
                points_2d = keypoints.reshape(-1, 2)
                confidence = np.ones((points_2d.shape[0], 1))
                keypoints = np.hstack((points_2d, confidence))
            else:
                print(f"Error: Invalid number of values in keypoints array: {len(keypoints)}")
                return None
        
        # If we only have x,y coordinates, add confidence values
        if keypoints.shape[1] == 2:
            confidence = np.ones((keypoints.shape[0], 1))
            keypoints = np.hstack((keypoints, confidence))
        
        print(f"Processed keypoints shape: {keypoints.shape}")
        
        # Validate we have enough keypoints
        if keypoints.shape[0] < 15:
            print(f"Error: Insufficient number of keypoints: {keypoints.shape[0]}, need at least 15")
            # Create dummy keypoints for visualization
            dummy_keypoints = np.zeros((33, 3))  # MediaPipe uses 33 keypoints
            # Copy the keypoints we have
            dummy_keypoints[:keypoints.shape[0]] = keypoints
            keypoints = dummy_keypoints
            print("Created dummy keypoints for visualization")
        
        # Convert to MediaPipe format
        landmarks_dict = {
            'pose_landmarks': []
        }
        
        # Map keypoints to MediaPipe format
        for i in range(keypoints.shape[0]):
            landmark = {
                'x': float(keypoints[i, 0]),
                'y': float(keypoints[i, 1]),
                'z': 0.0,  # MediaPipe uses 3D coordinates
                'visibility': float(keypoints[i, 2]) if keypoints.shape[1] > 2 else 1.0
            }
            landmarks_dict['pose_landmarks'].append(landmark)
        
        print(f"Successfully converted {len(landmarks_dict['pose_landmarks'])} landmarks")
        return landmarks_dict
        
    except Exception as e:
        print(f"Error converting keypoints: {e}")
        traceback.print_exc()
        print(f"Debug - keypoints type: {type(keypoints)}")
        if isinstance(keypoints, (list, np.ndarray)):
            print(f"Debug - keypoints shape/length: {np.shape(keypoints)}")
        return None

def text_to_speech(text, output_path=None):
    """Convert text to speech and save as MP3"""
    try:
        # Clean up the text - remove markdown and make it more speech-friendly
        clean_text = text.replace('*', '').replace('#', '').replace('\n\n', '. ').replace('\n', '. ')
        
        # Create speech
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # If no output path specified, create a temporary file
        if output_path is None:
            temp_dir = "temp_audio"
            os.makedirs(temp_dir, exist_ok=True)
            output_path = os.path.join(temp_dir, f"analysis_{int(time.time())}.mp3")
            
        # Save the audio file
        tts.save(output_path)
        return output_path
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def generate_concise_feedback(analysis_text):
    """Generate a concise, user-friendly version of the analysis"""
    try:
        # Use OpenAI to create a concise version
        prompt = f"""
        Convert this technical exercise analysis into a brief, clear, motivational feedback:
        {analysis_text}
        
        Make it:
        1. Short and clear (max 3-4 sentences)
        2. Focus on key corrections needed
        3. Include positive reinforcement
        4. Use natural, conversational language
        """
        
        selected_llm = get_llm("openai")
        concise_feedback = extract_feedback_markdown(selected_llm.invoke(prompt))
        return concise_feedback
    except Exception as e:
        print(f"Error generating concise feedback: {e}")
        return analysis_text

def detect_exercise_type(keypoints, landmarks_dict):
    """Detect exercise type based on pose keypoints and landmarks"""
    try:
        print("\nStarting exercise type detection...")
        
        # Handle case where keypoints is None
        if keypoints is None and landmarks_dict is None:
            print("No keypoints or landmarks provided")
            return "general"
            
        # Convert keypoints to numpy array if needed and present
        if isinstance(keypoints, list):
            keypoints = np.array(keypoints).reshape(-1, 3)
            
        # Check if we have enough valid points in keypoints
        if keypoints is not None and len(keypoints) < 15:  # Minimum points needed
            print("Not enough keypoints for detection")
            return "general"
            
        # First, check available exercise types in dataset
        dataset_path = os.path.join("DATASET", "TRAIN")
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            return "general"
        
        available_poses = [d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Available poses in dataset: {available_poses}")
        
        # For tree pose detection - use landmarks if available
        is_tree = False
        if landmarks_dict and all(key in landmarks_dict for key in [
            mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE,
            mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE
        ]):
            # Calculate leg positions and angles
            right_hip = np.array([landmarks_dict[mp_pose.RIGHT_HIP]['x'],
                          landmarks_dict[mp_pose.RIGHT_HIP]['y']])
            right_knee = np.array([landmarks_dict[mp_pose.RIGHT_KNEE]['x'],
                           landmarks_dict[mp_pose.RIGHT_KNEE]['y']])
            right_ankle = np.array([landmarks_dict[mp_pose.RIGHT_ANKLE]['x'],
                            landmarks_dict[mp_pose.RIGHT_ANKLE]['y']])
            
            left_hip = np.array([landmarks_dict[mp_pose.LEFT_HIP]['x'],
                              landmarks_dict[mp_pose.LEFT_HIP]['y']])
            left_knee = np.array([landmarks_dict[mp_pose.LEFT_KNEE]['x'],
                               landmarks_dict[mp_pose.LEFT_KNEE]['y']])
            left_ankle = np.array([landmarks_dict[mp_pose.LEFT_ANKLE]['x'],
                                landmarks_dict[mp_pose.LEFT_ANKLE]['y']])
            
            # Calculate angles
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            
            print(f"Tree pose angles - Right leg: {right_leg_angle:.2f}, Left leg: {left_leg_angle:.2f}")
            
            # Tree pose typically has one straight leg and one bent leg
            if (abs(right_leg_angle - 180) < 20 and abs(left_leg_angle - 90) < 30) or \
               (abs(left_leg_angle - 180) < 20 and abs(right_leg_angle - 90) < 30):
                is_tree = True
                print("Detected tree pose")
        
        # Return detected exercise type
        if is_tree and "tree" in available_poses:
            return "tree"
            
        # If we have keypoints, try similarity matching
        if keypoints is not None and len(keypoints) >= 15:
            max_similarity = 0
            best_match = "general"
            
            for pose_type in available_poses:
                reference_path = os.path.join(dataset_path, pose_type)
                if os.path.exists(reference_path):
                    for img_file in os.listdir(reference_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            reference_keypoints = analyze_pose_openpose(os.path.join(reference_path, img_file))
                            if isinstance(reference_keypoints, list):
                                similarity = calculate_pose_similarity(keypoints, reference_keypoints)
                                print(f"Similarity with {pose_type}: {similarity:.2f}")
                                if similarity > max_similarity:
                                    max_similarity = similarity
                                    if similarity > 0.7:  # High similarity threshold
                                        best_match = pose_type
                                        print(f"Detected {pose_type} pose with similarity {similarity:.2f}")
            
            if max_similarity > 0.7:
                return best_match
    
    except Exception as e:
        print(f"Error in detect_exercise_type: {e}")
        traceback.print_exc()
        return "general"

def feedback_agent(user_input, user_id):
    """Generate comprehensive feedback including pose comparison and audio"""
    try:
        print("\n=== Generating Comprehensive Feedback ===")
        
        # Extract keypoints and landmarks from user_input
        keypoints = user_input.get('keypoints')
        landmarks_dict = user_input.get('landmarks_dict')
        
        # Detect exercise type using both keypoints and landmarks
        exercise_type = detect_exercise_type(keypoints, landmarks_dict)
        user_level = get_user_level(user_id) or "beginner"
        print(f"Exercise: {exercise_type}, Level: {user_level}")
        
        # Generate pose comparison analysis
        comparison_results = []
        
        # Analyze key aspects based on exercise type
        if exercise_type == "tree":
            # Analyze balance and alignment
            if landmarks_dict and 'pose_landmarks' in landmarks_dict:
                landmarks = landmarks_dict['pose_landmarks']
                
                # Check vertical alignment
                hip_alignment = calculate_horizontal_alignment(
                    landmarks[23],  # Left hip
                    landmarks[24]   # Right hip
                )
                shoulder_alignment = calculate_horizontal_alignment(
                    landmarks[11],  # Left shoulder
                    landmarks[12]   # Right shoulder
                )
                
                comparison_results.extend([
                    "Pose Comparison Analysis:",
                    "1. Balance and Stability:",
                    f"   - Hip alignment: {'Good' if hip_alignment < 10 else 'Needs adjustment'}",
                    f"   - Shoulder alignment: {'Good' if shoulder_alignment < 10 else 'Needs adjustment'}",
                    "",
                    "2. Form Corrections:",
                    "   - Keep your spine straight and aligned",
                    "   - Engage your core muscles",
                    "   - Press your foot firmly against your inner thigh",
                    "",
                    "3. Breathing:",
                    "   - Maintain steady, deep breaths",
                    "   - Breathe in through your nose and out through your mouth"
                ])
            else:
                print("No landmarks available for detailed analysis")
                comparison_results.extend([
                    "Basic Form Analysis:",
                    "1. Maintain proper alignment",
                    "2. Focus on balance and stability",
                    "3. Keep breathing steady and controlled"
                ])
        
        # Generate text feedback
        feedback_text = "\n".join(comparison_results)
        
        # Generate concise audio feedback
        audio_feedback = generate_concise_feedback(feedback_text)
        audio_path = os.path.join('output', f'feedback_{time.strftime("%Y%m%d_%H%M%S")}.mp3')
        text_to_speech(audio_feedback, audio_path)
        
        print("\n=== Feedback Generation Complete ===")
        return {
            'exercise_type': exercise_type,
            'user_level': user_level,
            'feedback_text': feedback_text,
            'audio_path': audio_path
        }
                                
    except Exception as e:
        print(f"Error generating feedback: {e}")
        traceback.print_exc()
        return {
            'exercise_type': 'general',
            'user_level': 'beginner',
            'feedback_text': '',
            'audio_path': None
        }

def visualize_pose_comparison(current_image, landmarks_dict, analysis_text, exercise_type=None, keypoints=None):
    """Visualize pose comparison between current pose and reference pose"""
    try:
        print("\n=== Creating pose comparison visualization ===")
        
        # Convert landmarks to keypoints array for similarity comparison
        if landmarks_dict and 'pose_landmarks' in landmarks_dict:
            user_keypoints = []
            for landmark in landmarks_dict['pose_landmarks']:
                user_keypoints.extend([landmark['x'], landmark['y'], landmark['visibility']])
            user_keypoints = np.array(user_keypoints).reshape(-1, 3)
            print(f"Extracted user keypoints shape: {user_keypoints.shape}")
        else:
            print("No landmarks found in current pose")
            user_keypoints = None
        
        # Get reference image with pose similarity comparison
        reference_image = get_pose_from_database(exercise_type, user_keypoints)
        
        if reference_image is None:
            print("Failed to get reference image")
            return current_image  # Return only user's image if reference not found
        
        # Create figure with two subplots side by side
        plt.style.use('dark_background')  # Use dark theme for better visibility
        fig = plt.figure(figsize=(20, 10))
        
        # Left subplot - User's pose
        ax1 = fig.add_subplot(121)
        ax1.imshow(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
        if landmarks_dict and 'pose_landmarks' in landmarks_dict:
            plot_landmarks(ax1, landmarks_dict['pose_landmarks'], color='g', alpha=0.8)
        ax1.axis('off')
        ax1.set_title("Your Pose", fontsize=14, color='white')
        
        # Right subplot - Reference pose
        ax2 = fig.add_subplot(122)
        ax2.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
        # Get landmarks for reference image
        ref_results = analyze_pose_mediapipe(reference_image)
        if ref_results and ref_results.pose_landmarks:
            ref_landmarks = []
            for landmark in ref_results.pose_landmarks.landmark:
                ref_landmarks.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'visibility': landmark.visibility
                })
            plot_landmarks(ax2, ref_landmarks, color='b', alpha=0.8)
        ax2.axis('off')
        ax2.set_title("Reference Pose", fontsize=14, color='white')
        
        # Add similarity score as text
        if user_keypoints is not None:
            similarity_score = calculate_pose_similarity(user_keypoints, ref_keypoints)
            fig.suptitle(f'Pose Similarity Score: {similarity_score:.1f}%', 
                        fontsize=16, color='white', y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        comparison_image = Image.open(buf)
        comparison_image = np.array(comparison_image)
        
        plt.close()
        
        return comparison_image
        
    except Exception as e:
        print(f"Error in visualize_pose_comparison: {e}")
        traceback.print_exc()
        return current_image  # Return original image if visualization fails

def draw_pose_landmarks(ax, landmarks_dict, height, width):
    """Helper function to draw pose landmarks and connections"""
    try:
        # Define custom landmark indices
        CUSTOM_NECK = 100
        CUSTOM_MID_HIP = 101
        mp_pose = mp.solutions.pose.PoseLandmark
        
        # Define connections for visualization
        connections = [
            (mp_pose.NOSE, CUSTOM_NECK),
            (CUSTOM_NECK, mp_pose.RIGHT_SHOULDER),
            (mp_pose.RIGHT_SHOULDER, mp_pose.RIGHT_ELBOW),
            (mp_pose.RIGHT_ELBOW, mp_pose.RIGHT_WRIST),
            (CUSTOM_NECK, mp_pose.LEFT_SHOULDER),
            (mp_pose.LEFT_SHOULDER, mp_pose.LEFT_ELBOW),
            (mp_pose.LEFT_ELBOW, mp_pose.LEFT_WRIST),
            (CUSTOM_NECK, CUSTOM_MID_HIP),
            (CUSTOM_MID_HIP, mp_pose.RIGHT_HIP),
            (mp_pose.RIGHT_HIP, mp_pose.RIGHT_KNEE),
            (mp_pose.RIGHT_KNEE, mp_pose.RIGHT_ANKLE),
            (CUSTOM_MID_HIP, mp_pose.LEFT_HIP),
            (mp_pose.LEFT_HIP, mp_pose.LEFT_KNEE),
            (mp_pose.LEFT_KNEE, mp_pose.LEFT_ANKLE)
        ]
        
        # Draw connections
        for start, end in connections:
            if start in landmarks_dict and end in landmarks_dict:
                start_point = landmarks_dict[start]
                end_point = landmarks_dict[end]
                
                if start_point['visibility'] > 0.2 and end_point['visibility'] > 0.2:
                    start_xy = (int(start_point['x'] * width), int(start_point['y'] * height))
                    end_xy = (int(end_point['x'] * width), int(end_point['y'] * height))
                    ax.plot([start_xy[0], end_xy[0]], [start_xy[1], end_xy[1]], 
                           color='g', linewidth=2, alpha=0.7)
        
        # Draw landmarks
        for landmark_id, point in landmarks_dict.items():
            if point['visibility'] > 0.2:
                x = int(point['x'] * width)
                y = int(point['y'] * height)
                ax.scatter(x, y, c='r', s=50)
                
                # Add labels with improved visibility
                label = str(landmark_id)
                if landmark_id == CUSTOM_NECK:
                    label = "NECK"
                elif landmark_id == CUSTOM_MID_HIP:
                    label = "MID_HIP"
                elif isinstance(landmark_id, mp_pose):
                    label = landmark_id.name
                    
                ax.annotate(label, (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                           fontsize=8)
                           
    except Exception as e:
        print(f"Error drawing landmarks: {e}")


def main():
    voice_processor = VoiceProcessor()  # <--- This should be the first line in main()
    st.set_page_config(page_title="🏋️ AI-Powered Fitness Coach", layout="wide")
    
    # Initialize session state for user level and persistence
    if "user_level" not in st.session_state:
        st.session_state.user_level = "beginner"
    if "last_user_id" not in st.session_state:
        st.session_state.last_user_id = None
    if "shared_user_id" not in st.session_state:
        st.session_state.shared_user_id = ""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    if "browser_recorded_audio_path" not in st.session_state:
        st.session_state.browser_recorded_audio_path = None
    if "browser_recorded_transcription" not in st.session_state:
        st.session_state.browser_recorded_transcription = None
   

    # Reset button
    if st.sidebar.button("🔄 Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.success("✨ Session reset successfully!")
        st.session_state.user_level = "beginner"  # Reinitialize default values
        return    

    st.title("🏋️ AI-Powered Fitness Coach")
    st.markdown("Get personalized feedback on your workout form and posture")

    tabs = st.tabs(["📸 Image Analysis", "🎥 Video Analysis", "💬 Text & Voice Feedback"])

    with tabs[0]:  # Image Analysis Tab
        st.header("Posture Image Analysis")
        
        # User ID input section
        if st.session_state.shared_user_id:
            user_id = st.text_input("User ID (e.g., user1)", 
                                value=st.session_state.shared_user_id,
                                key="img_user_id")
        else:
            user_id = st.text_input("User ID (e.g., user1)", key="img_user_id")
        
        # Update the shared user ID when this one changes
        if user_id:
            st.session_state.shared_user_id = user_id
            
            # FORCE check the most current level from database every time
            current_level = get_user_level(user_id)
            if current_level:
                st.session_state.user_level = current_level
                with st.expander("Current Level", expanded=True):
                    st.info(f"Current level from database: {current_level.title()}")
            else:
                st.session_state.user_level = "beginner"

        posture_image = st.file_uploader(
            "Upload Workout Posture Image", 
            type=['png', 'jpg', 'jpeg'],
            key="image_upload"
        )

        # Add webcam capture option
        st.write("---")
        st.write("📸 Or Use Webcam")
        exercise_type = st.selectbox(
            "Select Exercise Type",
            options=["plank", "downdog", "goddess", "tree", "warrior2"],
            key="exercise_type"
        )
        
        camera_image = st.camera_input("Take a picture")
        
        # Use either uploaded image or camera image
        if camera_image is not None:
            posture_image = camera_image
            st.info("Using webcam image for analysis")
            
        # Show preview of selected image
        if posture_image:
            st.image(posture_image, caption="Your pose", use_container_width=True)
            
        # AI Model Selection
        llm_provider = st.selectbox(
            "Select AI Model",
            options=["openai", "groq"],
            key="img_llm_select",
            help="OpenAI uses GPT-4, Groq uses Mixtral-8x7b"
        )
        
        if st.button("Analyze Image", key="analyze_image"):
            if not posture_image:
                log_system_status(user_id, "none", "Error", "No image uploaded")
                st.error("Please upload an image first")
                return
            
            # Validate LLM provider
            if llm_provider == "groq":
                try:
                    # Test Groq connection first
                    selected_llm = get_llm("groq")
                    test_response = selected_llm.invoke("Test connection")
                    if not test_response:
                        st.error("Failed to connect to Groq. Falling back to OpenAI.")
                        llm_provider = "openai"
                except Exception as e:
                    st.error(f"Groq connection error: {str(e)}. Falling back to OpenAI.")
                    llm_provider = "openai"

            # new validation check
            is_valid, user_status, is_new_user = validate_user_id(user_id)
            if not is_valid:
                st.error(user_status)
                return

            log_system_status(user_id, "both", "Validation", "User ID validated")
            
            # Get the most current user level
            fresh_level = get_user_level(user_id)
            if fresh_level is None:
                fresh_level = "beginner"
                print(f"⚠️ WARNING: No level found in database for user {user_id}, defaulting to beginner")

            # Show user status with appropriate icon and set level
            if is_new_user:
                st.info("🆕 New user", icon="👤")
                current_level = "beginner"
            else:
                st.info("👤 Existing user", icon="✅")
                current_level = fresh_level if fresh_level else "beginner"
                st.warning(f"⚠️ CRITICAL LEVEL CHECK: Using {current_level} from database")
                
            # Update session state
            st.session_state.user_level = current_level
            st.session_state.last_user_id = user_id

            st.write("---")
            st.write(f"**LEVEL TRACKING** - User ID: {user_id}")
            st.write(f"Level from Database: **{fresh_level}**")
            st.write(f"Level used for Analysis: **{current_level}**")
            st.write("---")

            # Get user history outside the try block
            user_history = get_ltm(user_id)

            try:
                timestamp = int(time.time())
                temp_path = os.path.join(TEMP_IMAGES, f"pose_image_{timestamp}.jpg")
                
                with open(temp_path, "wb") as f:
                    f.write(posture_image.getvalue())

                with st.spinner('Analyzing posture...'):
                    # Generate analysis using the confirmed level
                    result = generate_workout_with_level(
                        user_id,
                        "", None, "", None,
                        temp_path,
                        llm_provider,
                        current_level
                    )

                    # Compare with reference pose if exercise type is selected
                    if exercise_type:
                        # Load reference pose from dataset
                        reference_path = f"DATASET/TRAIN/{exercise_type}"
                        if os.path.exists(reference_path):
                            # Get reference keypoints
                            reference_image = os.path.join(reference_path, os.listdir(reference_path)[0])
                            reference_keypoints = analyze_pose_openpose(reference_image)
                            
                            # Get current pose keypoints
                            current_keypoints = analyze_pose_openpose(temp_path)
                            
                            if isinstance(current_keypoints, list) and isinstance(reference_keypoints, list):
                                # Calculate pose similarity
                                similarity_score = calculate_pose_similarity(current_keypoints, reference_keypoints)
                                
                                # Generate comparison feedback
                                comparison_prompt = f"""
                                Analyze the pose comparison for {exercise_type} exercise:
                                - Similarity score: {similarity_score}
                                - User level: {current_level}
                                
                                Provide specific feedback on:
                                1. How well the pose matches the reference
                                2. Key differences in alignment
                                3. Specific adjustments needed
                                4. Safety considerations
                                """
                                
                                selected_llm = get_llm(llm_provider)
                                comparison_feedback = extract_feedback_markdown(selected_llm.invoke(comparison_prompt))
                                
                                # Add comparison results to the output
                                final_analysis = f"""
                                ### Pose Comparison Results
                                
                                Similarity Score: {similarity_score:.2%}
                                
                                {comparison_feedback}
                                
                                ---
                                
                                {result.get('analysis', '')}
                                """
                                
                                # Update the result with combined analysis
                                result["analysis"] = final_analysis
                                
                                # Generate audio from the complete analysis
                                audio_path = text_to_speech(final_analysis)
                                result["audio_path"] = audio_path

                    # Display results
                    st.markdown("### Analysis Result")
                    
                    # Display the visualization with both poses
                    if "image_path" in result and os.path.exists(result["image_path"]):
                        # Read and display the comparison visualization
                        with open(result["image_path"], "rb") as f:
                            visualization_bytes = f.read()
                        st.image(visualization_bytes, caption="Pose Comparison Analysis", use_container_width=True)
                    
                    if isinstance(result, dict) and "analysis" in result:
                        st.markdown(result["analysis"])
                        
                        # --- Generate and play TTS for feedback ---
                        import re
                        feedback_text = re.sub(r'<.*?>', '', result["analysis"])  # Remove HTML tags if any
                        feedback_text = re.sub(r'\\*', '', feedback_text)  # Remove markdown asterisks
                        feedback_text = feedback_text.strip()

                        # Generate TTS audio if not already done for this feedback
                        if 'last_feedback_text' not in st.session_state or st.session_state['last_feedback_text'] != feedback_text:
                            feedback_audio_path = text_to_speech(feedback_text)
                            st.session_state['feedback_audio_path'] = feedback_audio_path
                            st.session_state['last_feedback_text'] = feedback_text

                        # Show audio player if audio was generated
                        if st.session_state.get('feedback_audio_path') and os.path.exists(st.session_state['feedback_audio_path']):
                            st.audio(st.session_state['feedback_audio_path'], format="audio/mp3")
                        
                            # Add download button for audio
                            with open(st.session_state['feedback_audio_path'], "rb") as f:
                                audio_bytes = f.read()
                            st.download_button(
                                "⬇️ Download Voice Feedback",
                                audio_bytes,
                                file_name="voice_feedback.mp3",
                                mime="audio/mp3"
                            )
                    
                    # Add debug information in an expander
                    with st.expander("Debug Information"):
                        st.write("Processing Details:")
                        st.write(f"- Exercise Type: {exercise_type}")
                        st.write(f"- User Level: {current_level}")
                        if "image_path" in result:
                            st.write(f"- Visualization Path: {result['image_path']}")
                        if "message" in result:
                            st.write(f"- Status: {result['message']}")
                            
                    # Show previous sessions button
                    if st.button("Previous Sessions"):
                        st.write("Previous workout sessions:")
                        if user_history:
                            st.markdown(format_workout_history(user_history))
                        else:
                            st.info("No previous sessions found")

                    # User-friendly history display
                    with st.expander("Previous Sessions", expanded=False):
                        st.markdown(format_workout_history(user_history))
                    
                    # Store the analysis result with timestamp and level
                    store_user_data(
                        user_id,
                        f"Image Analysis: {result['analysis'] if isinstance(result, dict) and 'analysis' in result else str(result)[:200]}...",
                        "both",
                        current_level
                    )
                    
                    store_stm(user_id, f"Latest image analysis completed at {datetime.now().strftime('%H:%M:%S')}")
                    
                    # Show debug info
                    with st.expander("Debug Information", expanded=False):
                        st.write("User ID:", user_id)
                        st.write("Shared User ID:", st.session_state.shared_user_id)
                        st.write("Fresh Database Level:", fresh_level if 'fresh_level' in locals() else "Not fetched yet") 
                        st.write("Level Used For Analysis:", current_level if 'current_level' in locals() else "Not set yet")
                        st.write("Session State Level:", st.session_state.get('user_level'))
                        st.write("STM Data:", get_stm(user_id))
                        st.write("User Status:", user_status if 'user_status' in locals() else "Not checked yet")
                        st.write("Is New User:", is_new_user if 'is_new_user' in locals() else "Not checked yet")
                        st.write("Analysis Timestamp:", datetime.now().isoformat())
                                        
                    # Add detailed logging for terminal
                    print("\n=== Memory Status ===")
                    print(f"STM Data: {get_stm(user_id)}")
                    print(f"LTM History: {user_history}")
                    print(f"Current Level: {current_level}")
                    print("==================\n")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                print(f"Detailed error information: {e}")
                print("Current session state:", st.session_state)
                print("User level from storage:", get_user_level(user_id))
            
            finally:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)

    with tabs[1]:  # Video Analysis Tab

        if st.session_state.shared_user_id:
            # Always check the current level from the database
            db_level = get_user_level(st.session_state.shared_user_id)
            if db_level:
                st.session_state.user_level = db_level
                # Only display this for debugging
                with st.expander("Debug"):
                    st.write(f"User level from DB: {db_level}")

        st.header("Video Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Use the shared user ID if it exists
            if st.session_state.shared_user_id:
                user_id_video = st.text_input("User ID (e.g., user1)", 
                                            value=st.session_state.shared_user_id,
                                            key="video_user_id")
            else:
                user_id_video = st.text_input("User ID (e.g., user1)", key="video_user_id")
                
            # Update the shared ID
            if user_id_video:
                st.session_state.shared_user_id = user_id_video
                
                # FORCE check current level from database
                current_level = get_user_level(user_id_video)
                if current_level:
                    st.session_state.user_level = current_level
                    st.info(f"🎯 Current user level: {current_level.title()}")
                else:
                    st.session_state.user_level = "beginner"
                    st.info("🎯 Current user level: Beginner")

            video_file = st.file_uploader(
                "Upload Workout Video", 
                type=['mp4', 'mov', 'avi'],
                key="video_upload"
            )

            # file size check 
            if video_file:
                file_size = video_file.getvalue().size / (1024 * 1024)  # Convert to MB
                if file_size > 50:  # 50MB limit
                    st.error("⚠️ Video file is too large. Please upload a file smaller than 50MB.")
                    video_file = None
                else:
                    st.success(f"✅ Video size: {file_size:.1f}MB")

            llm_provider_video = st.selectbox(
                "Select AI Model",
                options=["openai", "groq"],
                key="video_llm_select"
            )
            
        with col2:
            st.write("Preview:")
            if video_file:
                st.video(video_file)
        
        if st.button("Analyze Video", key="analyze_video"):
            if not video_file:
                st.error("Please upload a video first")
                return
                
            is_valid, user_status, is_new_user = validate_user_id(user_id_video)
            if not is_valid:
                st.error(user_status)
                return
            # Show user status with appropriate icon
            if is_new_user:
                st.info("🆕 New user", icon="👤")
            else:
                st.info("👤 Existing user", icon="✅")
                
            with st.spinner('Processing video...'):
                # Get user history before processing
                user_history = get_ltm(user_id_video)

                        # IMPORTANT: Get fresh level directly from the database
                fresh_level = get_user_level(user_id_video)
                if fresh_level and not is_new_user:
                    current_level = fresh_level
                    st.success(f"Using {current_level.title()} level for analysis")
                else:
                    current_level = "beginner"
                
                # Update the session state
                st.session_state.user_level = current_level
                
                log_system_status(user_id_video, "both", "Video Analysis", f"Processing video for {current_level} level user")
        
                result = process_video(video_file, user_id_video, llm_provider_video, current_level)
                
                # Display results
                st.markdown("### Analysis Result")
                st.markdown(result)
                
                # User-friendly history display
                with st.expander("Previous Sessions", expanded=False):
                    st.markdown(format_workout_history(user_history))
                
                # Keep debug info hidden (for development only)
                with st.expander("Debug Information", expanded=False):
                    st.write("User ID:", user_id_video)
                    st.write("Shared User ID:", st.session_state.shared_user_id)
                    st.write("Database User Level:", get_user_level(user_id_video))
                    st.write("Session State Level:", st.session_state.get('user_level'))
                    st.write("User Status:", user_status if 'user_status' in locals() else "Not checked yet")
                    st.write("Is New User:", is_new_user if 'is_new_user' in locals() else "Not checked yet")
                    st.write("Analysis Timestamp:", datetime.now().isoformat())
                
                # Add detailed logging for terminal
                print("\n=== Video Analysis Memory Status ===")
                print(f"STM Data: {get_stm(user_id_video)}")
                print(f"LTM History: {user_history}")
                print("==================\n")

    with tabs[2]:  # Text & Voice Feedback Tab

        if st.session_state.shared_user_id:
            # Always check the current level from the database
            db_level = get_user_level(st.session_state.shared_user_id)
            if db_level:
                st.session_state.user_level = db_level
                # Only display this for debugging
                with st.expander("Debug"):
                    st.write(f"User level from DB: {db_level}")

        st.header("Additional Feedback")
    
        # User ID
        if st.session_state.shared_user_id:
            user_id_feedback = st.text_input("User ID (e.g., user1)", 
                                            value=st.session_state.shared_user_id,
                                            key="feedback_user_id")
        else:
            user_id_feedback = st.text_input("User ID (e.g., user1)", key="feedback_user_id")

        # Update the shared ID
        if user_id_feedback:
            st.session_state.shared_user_id = user_id_feedback
            
            # FORCE check current level from database
            current_level = get_user_level(user_id_feedback)
            if current_level:
                st.session_state.user_level = current_level
            else:
                st.session_state.user_level = "beginner"
               
        user_level_choice = st.radio(
            "Select your experience level:", 
            ["beginner", "intermediate", "advanced"],
            index={"beginner": 0, "intermediate": 1, "advanced": 2}.get(
                get_user_level(user_id_feedback) if user_id_feedback else "beginner", 0
            )
        )

        # Workout Preferences Section
        st.subheader("💪 Workout Preferences")
        st.markdown("""
        Share your fitness goals, preferred exercises, limitations, or experience level.
        Choose either text or voice input (or both).
        """)
        
        pref_col1, pref_col2 = st.columns(2)
        with pref_col1:
            workout_prefs_text = st.text_area(
                "Text Input",
                placeholder="Example: I'm intermediate level, interested in strength training and want to improve my lower back strength...",
                help="Type your workout preferences and feedback here"
            )

    
        with pref_col2:
            workout_prefs_audio = st.file_uploader(
                "Voice Input (Optional)", 
                type=['wav', 'mp3'],
                key="workout_audio",
                help="Record and upload your preferences as voice"
            )
            if workout_prefs_audio:
                st.audio(workout_prefs_audio)
        
        # AI Model Selection
        llm_provider_feedback = st.selectbox(
            "Select AI Model",
            options=["openai", "groq"],
            key="feedback_llm_select"
        )

        # Process Button
        if st.button("Get AI Feedback", key="process_feedback"):
            # Validate user ID first
            is_valid, user_status, is_new_user = validate_user_id(user_id_feedback)
            if not is_valid:
                st.error(user_status)
                return

            # Show user status with appropriate icon
            if is_new_user:
                st.info("🆕 New user", icon="👤")
            else:
                st.info("👤 Existing user", icon="✅")
                existing_level = get_user_level(user_id_feedback)
                st.info(f"Previous level: {existing_level}")
                
                # Show user history
                with st.expander("Previous Sessions"):
                    user_history = get_ltm(user_id_feedback)
                    st.markdown(format_workout_history(user_history))

            try:
                # Initialize audio paths
                workout_audio_path = None
                timestamp = int(time.time())
                
                # Process workout preferences audio
                if workout_prefs_audio:
                    workout_audio_path = os.path.join(TEMP_IMAGES, f"workout_audio_{timestamp}.wav")
                    with open(workout_audio_path, "wb") as f:
                        f.write(workout_prefs_audio.getvalue())
                    workout_audio_text = speech_to_text(workout_audio_path)
                    st.info("✓ Audio processed")
                else:
                    workout_audio_text = ""

                # Get transcription from browser recording if available
                browser_recording_text = st.session_state.get("browser_recorded_transcription", "")

                # Combine all possible inputs
                combined_text = f"{workout_prefs_text} {workout_audio_text} {browser_recording_text}".strip()
                if not combined_text:
                    st.error("Please provide at least one input (text or voice).")
                    return

                # --- EMOTION DETECTION INTEGRATION ---
                predicted_emotion, emotion_feedback, scores = None, None, None
                if st.session_state.get('browser_recorded_audio_path'):
                    emotion_result = voice_processor.process_and_save_feedback(
                        user_id_feedback, 
                        audio_path=st.session_state['browser_recorded_audio_path']
                    )
                    if emotion_result:
                        predicted_emotion, emotion_feedback, scores = emotion_result
                        st.success(f"Detected Emotion: {predicted_emotion}")
                        st.info(f"Emotion Feedback: {emotion_feedback}")

                # Add emotion info to the prompt if available
                emotion_info = f"\nDetected Emotion: {predicted_emotion}\nEmotion Feedback: {emotion_feedback}\n" if predicted_emotion else ""
                combined_text_with_emotion = f"{combined_text}\n{emotion_info}"    

                # Use the directly selected level instead of determining from text
                new_level = user_level_choice
                
                # Show level change if user exists
                if not is_new_user and new_level != existing_level:
                    st.warning(f"Level change detected: {existing_level} → {new_level}")
                
                log_system_status(user_id_feedback, "both", "Level Assignment", 
                                f"User level updated from {existing_level if not is_new_user else 'none'} to {new_level}")

                # Store data in both LTM and STM
                store_user_data(
                    user_id_feedback,
                    f"Level: {new_level}\nPreferences: {workout_prefs_text} {workout_audio_text}",
                    "both",
                    new_level
                )

                # Update session state
                st.session_state.user_level = new_level
                st.session_state.last_user_id = user_id_feedback
                
                # Generate and display workout analysis
                with st.spinner('Processing your feedback...'):
                    result = generate_workout_with_level(
                        user_id_feedback,
                        combined_text_with_emotion,
                        workout_audio_path,
                        "", # Empty feedback_text
                        None, # No feedback_audio
                        None,
                        llm_provider_feedback,
                        new_level
                    )
                    st.markdown(result)
                # Only run this if 'result' exists and is not empty
                if result:
                    import re 
                    feedback_text = re.sub(r'<.*?>', '', result)  # Remove HTML tags if any
                    feedback_text = re.sub(r'\\*', '', feedback_text)  # Remove markdown asterisks
                    feedback_text = feedback_text.strip()

                    # Generate TTS audio if not already done for this feedback
                    if 'last_feedback_text' not in st.session_state or st.session_state['last_feedback_text'] != feedback_text:
                        feedback_audio_path = text_to_speech(feedback_text)
                        st.session_state['feedback_audio_path'] = feedback_audio_path
                        st.session_state['last_feedback_text'] = feedback_text

                    # Show audio player if audio was generated
                    if st.session_state.get('feedback_audio_path') and os.path.exists(st.session_state['feedback_audio_path']):
                        st.audio(st.session_state['feedback_audio_path'], format="audio/mp3")    
                    
                # Show debug information
                with st.expander("Debug Information", expanded=False):
                    st.write("User ID:", user_id_feedback)
                    st.write("Shared User ID:", st.session_state.shared_user_id)
                    st.write("Database User Level:", get_user_level(user_id_feedback))
                    st.write("Session State Level:", st.session_state.get('user_level'))
                    st.write("Previous Level:", existing_level if 'existing_level' in locals() else "N/A")
                    st.write("New Level:", new_level if 'new_level' in locals() else "N/A")
                    st.write("STM Data:", get_stm(user_id_feedback))
                    st.write("Combined Input:", combined_text if 'combined_text' in locals() else "N/A")
                    st.write("Level Determination Test:", determine_user_level(combined_text, user_id_feedback) if 'combined_text' in locals() else "N/A")

            except Exception as e:
                st.error(f"Error processing feedback: {str(e)}")
                print(f"Error details: {e}")
            
            finally:
                # Cleanup temporary files
                if workout_audio_path and os.path.exists(workout_audio_path):
                    os.remove(workout_audio_path)

        # Add help section at the bottom
        with st.expander("ℹ️ How to Use"):
            st.markdown("""
            1. **Image Analysis**: 
            - Upload a photo of your workout pose
            - Get immediate feedback on your form
            
            2. **Video Analysis**: 
            - Upload a video clip (up to 30 seconds recommended)
            - Get frame-by-frame analysis of your movement
            - Review feedback for each key moment
            
            3. **Text & Voice Feedback**: 
            - Type or record your workout preferences
            - Provide additional context or questions
            - Choose your preferred AI model for analysis
            
            **Note**: For best results, ensure good lighting and clear visibility in your images and videos.
            """)

        st.markdown("**🎤 Record your preferences by voice:**")

        # Use the local st_audiorec component
        audio_bytes = st_audiorec(key="audiorec_preferences")

        if audio_bytes and isinstance(audio_bytes, bytes):
            st.audio(audio_bytes, format='audio/wav')

            # Save to temp .wav file (optional, for transcription or emotion analysis)
            temp_audio_path = f"temp_voice_input_{int(time.time())}.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_bytes)

            # Transcribe the audio (you can replace this with your own function)
            try:
                transcription = speech_to_text(temp_audio_path)
                st.success("📝 Transcription:")
                st.write(transcription)
                # Optional: store in session state
                st.session_state['approved_voice_transcription'] = transcription
                st.session_state['approved_voice_file'] = temp_audio_path
            except Exception as e:
                st.error(f"Transcription failed: {str(e)}")

        else:
            st.info("Click the red mic button and speak to record your preferences.")

        # Add this where you want the audio recorder to appear
        st.subheader("Record Audio")
        wav_audio_data = st_audiorec(key="audiorec_main")
        
        if wav_audio_data is not None:
            st.write("Audio recorded! Playing back...")
            st.audio(wav_audio_data, format='audio/wav')
            
            # If you want to process the audio (e.g., convert to text)
            if st.button("Process Audio"):
                try:
                    text = speech_to_text(wav_audio_data)
                    st.write("Transcribed text:", text)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

def test_database_connection():
    """
    Test the connection to the API Ninjas exercise database
    """
    try:
        # Load API key
        API_NINJAS_KEY = os.getenv('EXRX_API_KEY')
        if not API_NINJAS_KEY:
            return "API Ninjas key not found in .env file"
            
        # Test API endpoint
        API_BASE_URL = "https://api.api-ninjas.com/v1/exercises"
        
        # Try to get a simple exercise
        response = requests.get(
            API_BASE_URL,
            params={'name': 'plank'},
            headers={'X-Api-Key': API_NINJAS_KEY}
        )
        
        if response.status_code == 200:
            exercises = response.json()
            if exercises:
                return f"Successfully connected to API Ninjas database. Found {len(exercises)} exercises."
            return "Connected to database but no exercises found"
        elif response.status_code == 401:
            return "Invalid API key"
        elif response.status_code == 404:
            return "API endpoint not found"
        else:
            return f"Error connecting to database: {response.status_code}"
            
    except Exception as e:
        return f"Connection error: {str(e)}"

def generate_reference_pose(exercise_type, description):
    """Generate a reference pose image using DALL-E"""
    try:
        client = OpenAI()
        
        # Create a detailed prompt for DALL-E
        prompt = f"""
        A clear, professional fitness instruction photo showing perfect form for {exercise_type} exercise.
        The image should:
        - Show a person in athletic wear against a plain background
        - Demonstrate proper {exercise_type} form
        - Highlight key alignment points
        - Be well-lit and clear
        Additional details: {description}
        """
        
        # Generate image with DALL-E
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        # Download the image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        # Save the image
        output_dir = "reference_poses"
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, f"reference_{exercise_type}_{int(time.time())}.png")
        
        with open(image_path, 'wb') as f:
            f.write(image_response.content)
            
        return image_path
    except Exception as e:
        print(f"Error generating reference pose: {e}")
        return None

def calculate_pose_similarity(current_keypoints, reference_keypoints):
    """Calculate similarity between two poses using keypoint distances"""
    try:
        # Convert to numpy arrays
        current = np.array(current_keypoints).reshape(-1, 3)
        reference = np.array(reference_keypoints).reshape(-1, 3)
        
        # Filter out low confidence points
        confidence_threshold = 0.1  # Lowered threshold
        current_valid = current[current[:, 2] > confidence_threshold]
        reference_valid = reference[reference[:, 2] > confidence_threshold]
        
        # If we don't have enough valid points, return low similarity
        if len(current_valid) < 5 or len(reference_valid) < 5:
            print("Not enough valid points for comparison")
            return 0.0
            
        # Extract coordinates and normalize each pose independently
        current_coords = current_valid[:, :2]
        reference_coords = reference_valid[:, :2]
        
        # Normalize each pose to its own coordinate space
        def normalize_pose(coords):
            # Center the pose
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            
            # Scale to unit size
            scale = np.max(np.abs(centered))
            if scale > 0:
                normalized = centered / scale
            else:
                normalized = centered
            return normalized
        
        current_normalized = normalize_pose(current_coords)
        reference_normalized = normalize_pose(reference_coords)
        
        # Calculate similarity using Procrustes analysis
        min_points = min(len(current_normalized), len(reference_normalized))
        current_points = current_normalized[:min_points]
        reference_points = reference_normalized[:min_points]
        
        # Calculate the distance between corresponding points
        distances = np.linalg.norm(current_points - reference_points, axis=1)
        mean_distance = np.mean(distances)
        
        # Convert distance to similarity score (0 to 1)
        # Using exponential decay for more intuitive scaling
        similarity = np.exp(-mean_distance)
        
        # Add debug output
        print(f"Debug - Number of points compared: {min_points}")
        print(f"Debug - Mean distance: {mean_distance}")
        print(f"Debug - Calculated similarity: {similarity}")
        
        return similarity
        
    except Exception as e:
        print(f"Error calculating pose similarity: {e}")
        return 0.0

def get_pose_from_database(exercise_type, user_keypoints=None):
    """Get reference pose image from database with pose similarity comparison"""
    try:
        print(f"\n=== Loading reference image for {exercise_type} ===")
        
        # Check cached reference image first
        cache_dir = os.path.join('output', 'reference_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'{exercise_type}_best_match.jpg')
        
        if os.path.exists(cache_path) and not user_keypoints:
            print("Loading cached reference image...")
            return cv2.imread(cache_path)
        
        # Define dataset paths for your yoga poses
        dataset_paths = [
            os.path.join("DATASET", "TRAIN", exercise_type),
            os.path.join("DATASET", "TEST", exercise_type)
        ]
        
        best_match_image = None
        best_similarity = -1
        best_image_path = None
        
        for dataset_path in dataset_paths:
            if not os.path.exists(dataset_path):
                print(f"Dataset path not found: {dataset_path}")
                continue
            
            print(f"Searching in: {dataset_path}")
            for img_file in os.listdir(dataset_path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = os.path.join(dataset_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue
                    
                    if user_keypoints is not None:
                        # Get reference pose keypoints
                        results = analyze_pose_mediapipe(img_path)
                        if results and results.pose_landmarks:
                            # Convert landmarks to keypoints format
                            ref_keypoints = []
                            for landmark in results.pose_landmarks.landmark:
                                ref_keypoints.extend([landmark.x, landmark.y, landmark.visibility])
                            ref_keypoints = np.array(ref_keypoints).reshape(-1, 3)
                            
                            # Calculate similarity
                            similarity = calculate_pose_similarity(user_keypoints, ref_keypoints)
                            print(f"Pose similarity score: {similarity:.3f}")
                            
                            # Update best match if this is better
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match_image = img.copy()
                                best_image_path = img_path
                                print(f"New best match found! Score: {similarity:.3f}")
                                
                                # Save to cache immediately
                                cv2.imwrite(cache_path, img)
                                print(f"Cached reference image saved to: {cache_path}")
                                
                                # If similarity is very high (>85%), stop searching
                                if similarity > 0.85:
                                    print("Found excellent match! Stopping search.")
                                    return best_match_image
                    else:
                        # If no user keypoints provided, use first valid image and cache it
                        print("No user keypoints provided, using first valid image")
                        cv2.imwrite(cache_path, img)
                        return img
                
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                    traceback.print_exc()
                    continue
        
        if best_match_image is not None:
            print(f"\nBest matching pose found: {best_image_path}")
            print(f"Similarity score: {best_similarity:.3f}")
            return best_match_image
        
        print("\nNo suitable reference images found in directories:", dataset_paths)
        return None
        
    except Exception as e:
        print(f"Error in get_pose_from_database: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()