import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import textwrap
import os
from PIL import Image
from openai import OpenAI
import traceback
import tempfile
import streamlit as st
from typing import List, Tuple, Dict, Optional, Union
import requests
import mediapipe as mp
from enum import Enum
from dataclasses import dataclass
from pose_detection import AdvancedPoseDetector, DetectionMethod, DetectionResult
import time
from datetime import datetime
from pathlib import Path

class DetectionMethod(Enum):
    MEDIAPIPE = "mediapipe"
    OPENPOSE = "openpose"
    BLAZEPOSE = "blazepose"

@dataclass
class DetectionResult:
    keypoints: np.ndarray
    confidence: float
    method: DetectionMethod
    landmarks: Optional[Dict] = None
    error: Optional[str] = None

def visualize_pose(
    image_path: str,
    keypoints: List[Tuple[float, float]],
    output_path: str
) -> None:
    """Create a visualization of pose keypoints.
    
    Args:
        image_path: Path to image
        keypoints: List of (x, y) keypoint coordinates
        output_path: Path to save visualization
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Display image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot keypoints and connections
        plot_pose_keypoints(keypoints)
        
        plt.title("Pose Analysis")
        plt.axis('off')
        
        # Save visualization
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        raise

def download_image(url: str) -> np.ndarray:
    """Download an image from URL and convert to OpenCV format.
    
    Args:
        url: Image URL
        
    Returns:
        Image as numpy array in BGR format
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Convert to numpy array
        arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
            
        return img
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

def resize_image(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to target height while maintaining aspect ratio.
    
    Args:
        img: Input image
        target_height: Desired height in pixels
        
    Returns:
        Resized image
    """
    h, w = img.shape[:2]
    ratio = target_height / float(h)
    target_width = int(w * ratio)
    return cv2.resize(img, (target_width, target_height))

def plot_pose_keypoints(keypoints: List[Tuple[float, float]]):
    """Plot pose keypoints and connections.
    
    Args:
        keypoints: List of (x, y) keypoint coordinates
    """
    # Define BODY_25 keypoint pairs for connections
    POSE_PAIRS = [
        # Torso
        (1, 8), (1, 2), (1, 5),
        # Right arm
        (2, 3), (3, 4),
        # Left arm
        (5, 6), (6, 7),
        # Right leg
        (8, 9), (9, 10), (10, 11),
        # Left leg
        (8, 12), (12, 13), (13, 14),
        # Face
        (0, 1),
        # Feet
        (11, 22), (11, 24), (14, 21), (14, 19)
    ]
    
    # Plot keypoints
    x = [p[0] for p in keypoints]
    y = [p[1] for p in keypoints]
    plt.scatter(x, y, c='r', s=50)
    
    # Plot connections
    for pair in POSE_PAIRS:
        if pair[0] < len(keypoints) and pair[1] < len(keypoints):
            p1 = keypoints[pair[0]]
            p2 = keypoints[pair[1]]
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:  # Only draw if points are valid
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)

def visualize_keypoints_confidence(keypoints):
    """Create a confidence visualization for keypoints."""
    plt.figure(figsize=(8, 6))
    confidence = keypoints[:, 2]
    plt.bar(range(len(confidence)), confidence)
    plt.title("Keypoint Confidence Scores")
    plt.xlabel("Keypoint Index")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    return plt

def add_squat_guidelines(ax, reference_pose):
    """Add visual guidelines for proper squat form"""
    if len(reference_pose) > 14:
        # Get key points
        r_shoulder = np.array(reference_pose[2])
        l_shoulder = np.array(reference_pose[5])
        r_hip = np.array(reference_pose[9])
        l_hip = np.array(reference_pose[12])
        r_knee = np.array(reference_pose[10])
        l_knee = np.array(reference_pose[13])
        r_ankle = np.array(reference_pose[11])
        l_ankle = np.array(reference_pose[14])
        
        # Calculate midpoints
        mid_shoulder = (r_shoulder + l_shoulder) / 2
        mid_hip = (r_hip + l_hip) / 2
        mid_knee = (r_knee + l_knee) / 2
        mid_ankle = (r_ankle + l_ankle) / 2
        
        # Draw vertical alignment line
        ax.plot([mid_ankle[0], mid_ankle[0]], [0, ax.get_ylim()[0]], 
                'b--', alpha=0.5, label='Vertical Alignment')
        
        # Draw shoulder width reference
        shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
        ideal_stance = shoulder_width * 1.3  # Slightly wider than shoulders
        
        # Draw ideal foot position
        ax.plot([mid_ankle[0] - ideal_stance/2, mid_ankle[0] + ideal_stance/2],
                [mid_ankle[1], mid_ankle[1]], 'g--', alpha=0.5, 
                label='Ideal Stance Width')
        
        # Add angle annotations
        ax.text(mid_hip[0] + 20, mid_hip[1], 
                'Hip angle: 170-180°', fontsize=8)
        ax.text(mid_knee[0] + 20, mid_knee[1], 
                'Knee angle: 170-180°', fontsize=8)
        
        ax.legend(loc='upper right', fontsize=8)

def plot_skeleton(ax, keypoints: List[Tuple[float, float]], color: str = 'blue', alpha: float = 1.0):
    """Plot skeleton connecting keypoints"""
    # Define connections for COCO format
    connections = [
        (1, 2), (1, 5),  # Shoulders
        (2, 3), (3, 4),  # Right arm
        (5, 6), (6, 7),  # Left arm
        (1, 8),  # Spine
        (8, 9), (9, 10), (10, 11),  # Right leg
        (8, 12), (12, 13), (13, 14)  # Left leg
    ]
    
    # Plot points
    xs = [p[0] for p in keypoints]
    ys = [p[1] for p in keypoints]
    ax.scatter(xs, ys, c=color, alpha=alpha)
    
    # Plot connections
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            ax.plot([start[0], end[0]], [start[1], end[1]], c=color, alpha=alpha)

def plot_angle_comparison(ax, angle_differences: Dict[str, float]):
    """Plot bar chart comparing angles"""
    joints = list(angle_differences.keys())
    differences = list(angle_differences.values())
    
    if not joints:
        ax.text(0.5, 0.5, "No angle data available", ha='center', va='center')
        return
        
    y_pos = np.arange(len(joints))
    ax.barh(y_pos, differences)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(joints)
    ax.set_xlabel("Angle Difference (degrees)")
    ax.set_title("Joint Angle Differences")
    
    # Add reference lines
    ax.axvline(x=15, color='yellow', linestyle='--', alpha=0.5)  # Warning threshold
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.5)  # Critical threshold

def detect_exercise_type(keypoints, landmarks_dict):
    """Detect exercise type based on body position and key angles"""
    try:
        # Get key body points
        shoulders = np.array([
            (landmarks_dict['LEFT_SHOULDER']['x'] + landmarks_dict['RIGHT_SHOULDER']['x']) / 2,
            (landmarks_dict['LEFT_SHOULDER']['y'] + landmarks_dict['RIGHT_SHOULDER']['y']) / 2
        ])
        
        elbows = np.array([
            (landmarks_dict['LEFT_ELBOW']['x'] + landmarks_dict['RIGHT_ELBOW']['x']) / 2,
            (landmarks_dict['LEFT_ELBOW']['y'] + landmarks_dict['RIGHT_ELBOW']['y']) / 2
        ])
        
        wrists = np.array([
            (landmarks_dict['LEFT_WRIST']['x'] + landmarks_dict['RIGHT_WRIST']['x']) / 2,
            (landmarks_dict['LEFT_WRIST']['y'] + landmarks_dict['RIGHT_WRIST']['y']) / 2
        ])
        
        hips = np.array([
            (landmarks_dict['LEFT_HIP']['x'] + landmarks_dict['RIGHT_HIP']['x']) / 2,
            (landmarks_dict['LEFT_HIP']['y'] + landmarks_dict['RIGHT_HIP']['y']) / 2
        ])

        # Calculate key measurements for pull-up detection
        arms_above_shoulders = wrists[1] < shoulders[1]  # Wrists higher than shoulders
        elbows_bent = calculate_angle_between_vectors(shoulders - elbows, elbows - wrists) < 150
        body_vertical = abs(calculate_angle_to_horizontal(shoulders - hips)) > 60
        
        # Check for pull-up position
        is_pullup = (
            arms_above_shoulders and
            elbows_bent and
            body_vertical and
            all(landmarks_dict[key]['visibility'] > 0.5 for key in 
                ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST'])
        )
        
        if is_pullup:
            return "pullup"
            
        return "general"
        
    except Exception as e:
        print(f"Error detecting exercise type: {e}")
        return "general"

def analyze_plank_form(landmarks_dict):
    """Analyze specific aspects of plank form"""
    try:
        # Calculate key positions
        shoulders = np.array([
            (landmarks_dict['LEFT_SHOULDER']['x'] + landmarks_dict['RIGHT_SHOULDER']['x']) / 2,
            (landmarks_dict['LEFT_SHOULDER']['y'] + landmarks_dict['RIGHT_SHOULDER']['y']) / 2
        ])
        
        elbows = np.array([
            (landmarks_dict['LEFT_ELBOW']['x'] + landmarks_dict['RIGHT_ELBOW']['x']) / 2,
            (landmarks_dict['LEFT_ELBOW']['y'] + landmarks_dict['RIGHT_ELBOW']['y']) / 2
        ])
        
        hips = np.array([
            (landmarks_dict['LEFT_HIP']['x'] + landmarks_dict['RIGHT_HIP']['x']) / 2,
            (landmarks_dict['LEFT_HIP']['y'] + landmarks_dict['RIGHT_HIP']['y']) / 2
        ])
        
        ankles = np.array([
            (landmarks_dict['LEFT_ANKLE']['x'] + landmarks_dict['RIGHT_ANKLE']['x']) / 2,
            (landmarks_dict['LEFT_ANKLE']['y'] + landmarks_dict['RIGHT_ANKLE']['y']) / 2
        ])
        
        # Calculate key measurements
        body_line = shoulders - ankles
        elbow_line = shoulders - elbows
        hip_line = hips - shoulders
        
        # Calculate angles
        body_angle = np.degrees(np.arctan2(abs(body_line[1]), abs(body_line[0])))
        elbow_angle = np.degrees(np.arctan2(abs(elbow_line[1]), abs(elbow_line[0])))
        hip_angle = np.degrees(np.arctan2(abs(hip_line[1]), abs(hip_line[0])))
        
        # Calculate deviations
        hip_deviation = np.abs(hips[1] - shoulders[1]) / np.linalg.norm(shoulders - ankles)
        
        return {
            'body_angle': body_angle,
            'elbow_angle': elbow_angle,
            'hip_angle': hip_angle,
            'hip_deviation': hip_deviation,
            'shoulder_height': shoulders[1],
            'hip_height': hips[1],
            'ankle_height': ankles[1]
        }
    except Exception as e:
        print(f"Error analyzing plank form: {e}")
        return None

def analyze_pose_geometry(keypoints, landmarks_dict):
    """Enhanced pose geometry analysis"""
    if detect_exercise_type(keypoints, landmarks_dict) == "plank":
        plank_analysis = analyze_plank_form(landmarks_dict)
        if plank_analysis:
            return {
                'body_angle': plank_analysis['body_angle'],
                'elbow_angle': plank_analysis['elbow_angle'],
                'hip_alignment': plank_analysis['hip_deviation'],
                'form_details': {
                    'hip_angle': plank_analysis['hip_angle'],
                    'shoulder_height': plank_analysis['shoulder_height'],
                    'hip_height': plank_analysis['hip_height'],
                    'ankle_height': plank_analysis['ankle_height']
                }
            }
    return None

def generate_precise_dalle_prompt(pose_measurements):
    prompt = f"""Create a minimalist anatomical illustration of a person performing a perfect plank exercise:

    Exercise form specifications:
    - Person viewed from side angle
    - Body forming {pose_measurements['body_angle']:.1f} degree angle with ground
    - Forearms flat on ground at {pose_measurements['elbow_angle']:.1f} degree angle
    - Straight line from head through spine to heels
    - Hip position perfectly aligned (current deviation: {pose_measurements['hip_alignment']:.2f})

    Essential details:
    - Show ONLY the exercise form
    - NO gym equipment
    - NO background elements
    - Simple stick figure or basic anatomical outline
    - Include dotted lines showing proper alignment
    - Add angle measurements for body position
    
    Style requirements:
    - Clean, technical illustration
    - White or light background
    - Black or dark blue lines
    - Use thin lines for body outline
    - Use dotted lines for alignment guides
    - Include simple angle markers"""
    return prompt

def generate_exercise_image(exercise_type, keypoints, landmarks_dict):
    if exercise_type == "plank":
        # Get exact pose measurements
        pose_geometry = analyze_pose_geometry(keypoints, landmarks_dict)
        
        # Generate precise prompt
        prompt = generate_precise_dalle_prompt(pose_geometry)
        
        # Generate image with DALL-E using precise prompt
        response = generate_dalle_image(prompt)
        
        return response

def generate_dalle_image(prompt):
    """Generate an image using DALL-E with specific exercise form parameters"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Enhance the prompt with specific style requirements
        enhanced_prompt = f"""Technical diagram of exercise form. {prompt}
        
        CRITICAL REQUIREMENTS:
        1. Style: Simple 2D technical diagram
        2. View: Perfect side view only
        3. Figure: Basic stick figure or simple anatomical outline
        4. Colors: Black lines on white background only
        5. Must include:
           - Dotted reference lines showing proper alignment
           - Angle measurements in degrees
           - Labels for key body positions
        6. Must NOT include:
           - Any gym equipment
           - Realistic human features
           - Backgrounds or decorative elements
           - 3D perspective or shading
        
        The image should look like a technical exercise form diagram from a fitness textbook."""
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            style="natural"  # Using natural style for cleaner, more technical look
        )
        
        return response.data[0].url
        
    except Exception as e:
        print(f"Error generating DALL-E image: {e}")
        return None

def create_comparison_diagram(current_landmarks, ideal_angles):
    """Create a technical diagram comparing current pose with ideal form"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Pose Comparison Analysis', pad=20)
        
        # Draw current pose
        draw_pose(ax, current_landmarks, color='red', alpha=0.7, label='Current Pose')
        
        # Calculate and draw ideal pose based on correct angles
        ideal_landmarks = generate_ideal_pose_landmarks(current_landmarks, ideal_angles)
        draw_pose(ax, ideal_landmarks, color='green', alpha=0.7, label='Ideal Form')
        
        # Add alignment guides
        draw_alignment_guides(ax, current_landmarks, ideal_landmarks)
        
        # Add angle measurements
        add_angle_measurements(ax, current_landmarks, ideal_landmarks)
        
        # Add legend
        ax.legend()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf
        
    except Exception as e:
        print(f"Error creating comparison diagram: {e}")
        return None

def draw_pose(ax, landmarks, color='blue', alpha=1.0, label=None):
    """Draw a pose skeleton with the given landmarks"""
    connections = [
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'LEFT_HIP'),
        ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'),
        ('LEFT_HIP', 'LEFT_KNEE'),
        ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'),
        ('RIGHT_KNEE', 'RIGHT_ANKLE')
    ]
    
    # Draw connections
    for start, end in connections:
        if start in landmarks and end in landmarks:
            start_point = landmarks[start]
            end_point = landmarks[end]
            if start_point['visibility'] > 0.5 and end_point['visibility'] > 0.5:
                ax.plot([start_point['x'], end_point['x']], 
                       [start_point['y'], end_point['y']], 
                       f'{color}-', linewidth=2, alpha=alpha, label=label if label else None)
                label = None  # Only show label once in legend

def generate_ideal_pose_landmarks(current_landmarks, ideal_angles):
    """Generate ideal pose landmarks based on current pose and correct angles"""
    ideal_landmarks = current_landmarks.copy()
    
    # Calculate reference points (shoulders as base)
    shoulder_center = np.array([
        (current_landmarks['LEFT_SHOULDER']['x'] + current_landmarks['RIGHT_SHOULDER']['x']) / 2,
        (current_landmarks['LEFT_SHOULDER']['y'] + current_landmarks['RIGHT_SHOULDER']['y']) / 2
    ])
    
    # Adjust positions based on ideal angles
    body_length = np.linalg.norm(
        np.array([current_landmarks['LEFT_SHOULDER']['x'], current_landmarks['LEFT_SHOULDER']['y']]) -
        np.array([current_landmarks['LEFT_ANKLE']['x'], current_landmarks['LEFT_ANKLE']['y']])
    )
    
    # Calculate ideal positions
    ideal_landmarks['LEFT_HIP']['y'] = shoulder_center[1]  # Level hips
    ideal_landmarks['RIGHT_HIP']['y'] = shoulder_center[1]
    
    # Adjust elbow position for 90-degree angle
    elbow_offset = body_length * 0.2
    ideal_landmarks['LEFT_ELBOW']['y'] = shoulder_center[1] + elbow_offset
    ideal_landmarks['RIGHT_ELBOW']['y'] = shoulder_center[1] + elbow_offset
    
    return ideal_landmarks

def draw_alignment_guides(ax, current_landmarks, ideal_landmarks):
    """Draw alignment guides and highlight deviations"""
    # Draw horizontal alignment line for perfect plank
    ax.axhline(y=ideal_landmarks['LEFT_SHOULDER']['y'], color='gray', linestyle='--', alpha=0.5)
    
    # Draw vertical alignment guides
    for landmark in ['LEFT_SHOULDER', 'LEFT_HIP', 'LEFT_ANKLE']:
        current_point = current_landmarks[landmark]
        ideal_point = ideal_landmarks[landmark]
        
        # Draw deviation arrows
        if abs(current_point['y'] - ideal_point['y']) > 0.05:
            ax.annotate('', 
                xy=(current_point['x'], current_point['y']),
                xytext=(current_point['x'], ideal_point['y']),
                arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5))

def add_angle_measurements(ax, current_landmarks, ideal_landmarks):
    """Add angle measurements to the diagram"""
    # Add body angle measurement
    body_angle = calculate_body_angle(current_landmarks)
    ideal_body_angle = calculate_body_angle(ideal_landmarks)
    
    ax.text(0.05, 0.95, f'Current body angle: {body_angle:.1f}°\nIdeal body angle: {ideal_body_angle:.1f}°',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

def calculate_body_angle(landmarks):
    """Calculate the angle between shoulders and ankles relative to horizontal"""
    shoulders = np.array([
        (landmarks['LEFT_SHOULDER']['x'] + landmarks['RIGHT_SHOULDER']['x']) / 2,
        (landmarks['LEFT_SHOULDER']['y'] + landmarks['RIGHT_SHOULDER']['y']) / 2
    ])
    ankles = np.array([
        (landmarks['LEFT_ANKLE']['x'] + landmarks['RIGHT_ANKLE']['x']) / 2,
        (landmarks['LEFT_ANKLE']['y'] + landmarks['RIGHT_ANKLE']['y']) / 2
    ])
    
    return np.degrees(np.arctan2(abs(shoulders[1] - ankles[1]), abs(shoulders[0] - ankles[0])))

base_prompts = {
    "plank": """Create a minimalist anatomical illustration of a person in perfect plank position:
    
    Must include:
    - Side view of person in plank
    - Straight line from head to heels
    - Forearms flat on ground
    - Elbows under shoulders
    - Level hips
    - NO gym equipment
    - NO background elements
    
    Technical elements:
    - Simple stick figure or basic anatomical outline
    - Dotted alignment lines
    - Angle measurements
    - White background
    - Black/dark blue lines
    - Clean, technical style""",
    # ... rest of your prompts ...
}

def create_simple_visualization(image_bytes, landmarks_dict, analysis_text):
    """Create a simple visualization with the original image and pose analysis"""
    try:
        # Create figure with two columns
        fig = plt.figure(figsize=(15, 6))
        
        # Left subplot for original image
        ax1 = fig.add_subplot(121)
        ax1.set_title('Your Form', fontsize=14, pad=20)
        
        # Display the original image
        if image_bytes is not None:
            img = Image.open(io.BytesIO(image_bytes))
            ax1.imshow(img)
            
            # Overlay pose landmarks if available
            if landmarks_dict:
                for name, point in landmarks_dict.items():
                    if point['visibility'] > 0.5:
                        ax1.plot(point['x'] * img.width, point['y'] * img.height, 'ro', markersize=5)
        
        ax1.axis('off')
        
        # Right subplot for analysis
        ax2 = fig.add_subplot(122)
        ax2.set_title('Analysis', fontsize=14, pad=20)
        ax2.axis('off')
        
        if analysis_text:
            wrapped_text = textwrap.fill(analysis_text, width=50)
            ax2.text(0.1, 0.9, wrapped_text, fontsize=12, va='top', linespacing=1.5)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        
        return buf
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()
        return None

def analyze_pullup_form(landmarks_dict):
    """Analyze pull-up form and calculate key measurements"""
    try:
        # Calculate key positions
        shoulders = np.array([
            (landmarks_dict['LEFT_SHOULDER']['x'] + landmarks_dict['RIGHT_SHOULDER']['x']) / 2,
            (landmarks_dict['LEFT_SHOULDER']['y'] + landmarks_dict['RIGHT_SHOULDER']['y']) / 2
        ])
        
        elbows = np.array([
            (landmarks_dict['LEFT_ELBOW']['x'] + landmarks_dict['RIGHT_ELBOW']['x']) / 2,
            (landmarks_dict['LEFT_ELBOW']['y'] + landmarks_dict['RIGHT_ELBOW']['y']) / 2
        ])
        
        wrists = np.array([
            (landmarks_dict['LEFT_WRIST']['x'] + landmarks_dict['RIGHT_WRIST']['x']) / 2,
            (landmarks_dict['LEFT_WRIST']['y'] + landmarks_dict['RIGHT_WRIST']['y']) / 2
        ])
        
        hips = np.array([
            (landmarks_dict['LEFT_HIP']['x'] + landmarks_dict['RIGHT_HIP']['x']) / 2,
            (landmarks_dict['LEFT_HIP']['y'] + landmarks_dict['RIGHT_HIP']['y']) / 2
        ])
        
        # Calculate angles and measurements
        elbow_angle = calculate_angle_between_vectors(shoulders - elbows, elbows - wrists)
        body_angle = calculate_angle_to_horizontal(shoulders - hips)
        grip_width = np.abs(landmarks_dict['LEFT_WRIST']['x'] - landmarks_dict['RIGHT_WRIST']['x'])
        
        return {
            'elbow_angle': elbow_angle,
            'body_angle': body_angle,
            'grip_width': grip_width,
            'shoulder_height': shoulders[1],
            'elbow_height': elbows[1],
            'wrist_height': wrists[1]
        }
    except Exception as e:
        print(f"Error analyzing pull-up form: {e}")
        return None

def create_pullup_visualization(image_bytes, landmarks_dict, analysis_text):
    """Create a technical visualization for pull-up form analysis"""
    try:
        # Create figure with three columns
        fig = plt.figure(figsize=(20, 8))
        
        # Left subplot for original image with pose overlay
        ax1 = fig.add_subplot(131)
        ax1.set_title('Current Form', fontsize=14, pad=20)
        
        # Display the original image
        if image_bytes is not None:
            img = Image.open(io.BytesIO(image_bytes))
            ax1.imshow(img)
            
            # Overlay pose landmarks and connections
            if landmarks_dict:
                draw_pose_connections(ax1, landmarks_dict, img.height, img.width)
        
        ax1.axis('off')
        
        # Middle subplot for technical analysis
        ax2 = fig.add_subplot(132)
        ax2.set_title('Form Analysis', fontsize=14, pad=20)
        
        if landmarks_dict:
            # Get form measurements
            measurements = analyze_pullup_form(landmarks_dict)
            
            # Draw technical diagram
            draw_technical_analysis(ax2, landmarks_dict, measurements)
            
            # Add measurement annotations
            add_measurement_annotations(ax2, measurements)
        
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # Right subplot for feedback text
        ax3 = fig.add_subplot(133)
        ax3.set_title('Feedback', fontsize=14, pad=20)
        ax3.axis('off')
        
        if analysis_text:
            wrapped_text = textwrap.fill(analysis_text, width=50)
            ax3.text(0.1, 0.9, wrapped_text, fontsize=12, va='top', linespacing=1.5)
        
        plt.tight_layout()
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        
        return buf
        
    except Exception as e:
        print(f"Error creating pull-up visualization: {e}")
        traceback.print_exc()
        return None

def draw_pose_connections(ax, landmarks_dict, height, width):
    """Draw pose connections on the image"""
    connections = [
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_ELBOW'),
        ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
        ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'LEFT_HIP'),
        ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP')
    ]
    
    # Draw connections
    for start, end in connections:
        if start in landmarks_dict and end in landmarks_dict:
            start_point = landmarks_dict[start]
            end_point = landmarks_dict[end]
            if start_point['visibility'] > 0.5 and end_point['visibility'] > 0.5:
                ax.plot([start_point['x'] * width, end_point['x'] * width],
                       [start_point['y'] * height, end_point['y'] * height],
                       'g-', linewidth=2, alpha=0.7)
    
    # Draw landmarks
    for name, point in landmarks_dict.items():
        if point['visibility'] > 0.5:
            ax.plot(point['x'] * width, point['y'] * height, 'ro', markersize=5)

def draw_technical_analysis(ax, landmarks_dict, measurements):
    """Draw technical analysis diagram"""
    # Draw ideal form reference (simplified stick figure)
    draw_ideal_pullup_form(ax)
    
    # Draw current form overlay
    draw_pose(ax, landmarks_dict, color='red', alpha=0.5, label='Current Form')
    
    # Add measurement lines and angles
    add_form_measurements(ax, measurements)

def draw_ideal_pullup_form(ax):
    """Draw ideal pull-up form reference"""
    # Draw bar
    ax.plot([0.3, 0.7], [0.9, 0.9], 'k-', linewidth=2)
    
    # Draw ideal form (simplified)
    ideal_points = {
        'bar_left': (0.4, 0.9),
        'bar_right': (0.6, 0.9),
        'wrist_left': (0.4, 0.9),
        'wrist_right': (0.6, 0.9),
        'elbow_left': (0.35, 0.8),
        'elbow_right': (0.65, 0.8),
        'shoulder_left': (0.4, 0.7),
        'shoulder_right': (0.6, 0.7),
        'hip': (0.5, 0.5),
        'knee': (0.5, 0.3),
        'ankle': (0.5, 0.1)
    }
    
    # Draw connections
    ax.plot([ideal_points['wrist_left'][0], ideal_points['elbow_left'][0]],
            [ideal_points['wrist_left'][1], ideal_points['elbow_left'][1]],
            'g--', alpha=0.5, label='Ideal Form')
    # ... (draw other connections)
    
    ax.legend()

def add_form_measurements(ax, measurements):
    """Add measurement annotations to the diagram"""
    # Add angle measurements
    ax.text(0.05, 0.95, 
            f"Elbow angle: {measurements['elbow_angle']:.1f}°\n" +
            f"Body angle: {measurements['body_angle']:.1f}°\n" +
            f"Grip width: {measurements['grip_width']:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

def add_measurement_annotations(ax, measurements):
    """Add detailed measurement annotations"""
    # Add form analysis text
    analysis = []
    
    # Analyze elbow angle
    if measurements['elbow_angle'] < 90:
        analysis.append("Elbows too bent")
    elif measurements['elbow_angle'] > 160:
        analysis.append("Arms not engaged")
    
    # Analyze body angle
    if abs(measurements['body_angle']) > 15:
        analysis.append("Body not vertical")
    
    # Analyze grip width
    if measurements['grip_width'] < 0.3:
        analysis.append("Grip too narrow")
    elif measurements['grip_width'] > 0.7:
        analysis.append("Grip too wide")
    
    # Add analysis text
    ax.text(0.05, 0.05, '\n'.join(analysis),
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom')

def visualize_pose_analysis(
    image: np.ndarray,
    detection_result: DetectionResult,
    quality_metrics: Dict[str, float],
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a simple visualization of pose detection results.
    
    Args:
        image: Input image as numpy array
        detection_result: DetectionResult containing landmarks and detection info
        quality_metrics: Dictionary of quality metrics
        output_path: Optional path to save visualization
        
    Returns:
        Annotated image as numpy array
    """
    try:
        # Create a copy of the image for drawing
        output_img = image.copy()
        
        # Convert BGR to RGB
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        
        # Draw landmarks if available
        if detection_result and detection_result.landmarks is not None:
            # Draw connections first
            connections = mp.solutions.pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start = detection_result.landmarks[start_idx]
                end = detection_result.landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                    end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                    cv2.line(output_img, start_point, end_point, (0, 255, 255), 2)
            
            # Draw landmarks
            for landmark in detection_result.landmarks:
                if landmark.visibility > 0.5:
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    cv2.circle(output_img, (x, y), 5, (0, 255, 0), -1)
        
        # Add text for quality metrics
        y_offset = 30
        cv2.putText(output_img, f"Detection Method: {detection_result.method.name}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(output_img, f"Confidence: {detection_result.confidence:.2f}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add quality metrics
        for metric, value in quality_metrics.items():
            y_offset += 30
            metric_name = " ".join(word.capitalize() for word in metric.split("_"))
            if isinstance(value, float):
                text = f"{metric_name}: {value:.2f}"
            else:
                text = f"{metric_name}: {value}"
            cv2.putText(output_img, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
            
        return output_img
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()
        return image  # Return original image if visualization fails

def simple_visualize_pose(image: np.ndarray, landmarks: List[dict]) -> np.ndarray:
    """
    Create a simple visualization of pose landmarks on an image.
    
    Args:
        image: Input image as numpy array
        landmarks: List of landmark dictionaries with x, y, visibility
        
    Returns:
        Image with pose landmarks drawn
    """
    # Create a copy of the image
    output_img = image.copy()
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Draw landmarks
    for landmark in landmarks:
        if landmark['visibility'] > 0.5:
            x = int(landmark['x'] * width)
            y = int(landmark['y'] * height)
            # Draw a circle at each landmark
            cv2.circle(output_img, (x, y), 5, (0, 255, 0), -1)
    
    # Draw connections between landmarks
    connections = [
        (0, 1),  # nose to neck
        (1, 2), (1, 5),  # neck to shoulders
        (2, 3), (3, 4),  # right arm
        (5, 6), (6, 7),  # left arm
        (1, 8),  # neck to hip
        (8, 9), (9, 10),  # right leg
        (8, 11), (11, 12)  # left leg
    ]
    
    for start_idx, end_idx in connections:
        if (start_idx < len(landmarks) and end_idx < len(landmarks) and
            landmarks[start_idx]['visibility'] > 0.5 and
            landmarks[end_idx]['visibility'] > 0.5):
            
            start_point = (
                int(landmarks[start_idx]['x'] * width),
                int(landmarks[start_idx]['y'] * height)
            )
            end_point = (
                int(landmarks[end_idx]['x'] * width),
                int(landmarks[end_idx]['y'] * height)
            )
            cv2.line(output_img, start_point, end_point, (0, 255, 255), 2)
    
    return output_img

def test_simple_visualization():
    """Test the simple visualization function"""
    # Create a test image (black background)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create sample landmarks
    landmarks = [
        {'x': 0.5, 'y': 0.1, 'visibility': 0.9},  # head
        {'x': 0.5, 'y': 0.2, 'visibility': 0.9},  # neck
        {'x': 0.4, 'y': 0.2, 'visibility': 0.9},  # right shoulder
        {'x': 0.3, 'y': 0.3, 'visibility': 0.9},  # right elbow
        {'x': 0.2, 'y': 0.4, 'visibility': 0.9},  # right wrist
        {'x': 0.6, 'y': 0.2, 'visibility': 0.9},  # left shoulder
        {'x': 0.7, 'y': 0.3, 'visibility': 0.9},  # left elbow
        {'x': 0.8, 'y': 0.4, 'visibility': 0.9},  # left wrist
        {'x': 0.5, 'y': 0.5, 'visibility': 0.9},  # hip
        {'x': 0.4, 'y': 0.7, 'visibility': 0.9},  # right knee
        {'x': 0.3, 'y': 0.9, 'visibility': 0.9},  # right ankle
        {'x': 0.6, 'y': 0.7, 'visibility': 0.9},  # left knee
        {'x': 0.7, 'y': 0.9, 'visibility': 0.9},  # left ankle
    ]
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    cv2.imwrite(os.path.join(output_dir, "simple_test_original.png"), test_image)
    
    # Create and save visualization
    viz_image = simple_visualize_pose(test_image, landmarks)
    cv2.imwrite(os.path.join(output_dir, "simple_test_processed.png"), viz_image)
    
    print("Test images saved to test_output directory")
    print("Original image: simple_test_original.png")
    print("Processed image: simple_test_processed.png")

def minimal_visualize_pose(image_path: str) -> None:
    """
    Create a minimal visualization using only OpenCV.
    
    Args:
        image_path: Path to the image file
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return
            
        # Create a simple stick figure for testing
        height, width = image.shape[:2]
        center_x = width // 2
        
        # Draw a basic stick figure
        # Head
        cv2.circle(image, (center_x, 100), 30, (0, 255, 0), 2)
        # Body
        cv2.line(image, (center_x, 130), (center_x, 300), (0, 255, 0), 2)
        # Arms
        cv2.line(image, (center_x, 200), (center_x - 80, 200), (0, 255, 0), 2)
        cv2.line(image, (center_x, 200), (center_x + 80, 200), (0, 255, 0), 2)
        # Legs
        cv2.line(image, (center_x, 300), (center_x - 50, 400), (0, 255, 0), 2)
        cv2.line(image, (center_x, 300), (center_x + 50, 400), (0, 255, 0), 2)
        
        # Add some text
        cv2.putText(image, "Test Visualization", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save the result
        output_path = "test_visualization.png"
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
        
        # Try to display the image
        cv2.imshow("Pose Visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Create a black image for testing
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_path = "test_image.png"
    cv2.imwrite(test_path, test_image)
    
    # Run visualization
    minimal_visualize_pose(test_path)
    
    # Clean up test image
    try:
        os.remove(test_path)
    except:
        pass

    test_simple_visualization() 