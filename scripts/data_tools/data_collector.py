import streamlit as st
import cv2
import numpy as np
import os
import json
import tempfile
from datetime import datetime
from pose_reference import PoseReference
from visualization import visualize_pose_comparison, plot_skeleton
import matplotlib.pyplot as plt
from PIL import Image
import io

IMPORTANT_LMS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE"
]

def main():
    st.title("Exercise Form Reference Data Collector")
    
    # Initialize PoseReference
    pose_ref = PoseReference()
    
    # Sidebar for data collection controls
    st.sidebar.header("Data Collection Controls")
    
    # Exercise selection
    exercise_type = st.sidebar.selectbox(
        "Select Exercise Type",
        ["plank", "pushup", "squat"]
    )
    
    # Form quality selection
    is_correct_form = st.sidebar.radio(
        "Form Quality",
        ["correct", "incorrect"]
    )
    
    # Metadata input
    st.sidebar.subheader("Metadata")
    difficulty = st.sidebar.select_slider(
        "Difficulty Level",
        options=["beginner", "intermediate", "advanced"]
    )
    
    view_angle = st.sidebar.select_slider(
        "View Angle",
        options=["front", "side", "45-degree"]
    )
    
    notes = st.sidebar.text_area(
        "Additional Notes",
        "Enter any additional observations about the form..."
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of the exercise form",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Read image
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Convert image for OpenPose
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Save temporary file for OpenPose
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, opencv_image)
            
            try:
                # Get pose keypoints using your existing OpenPose function
                from app import analyze_pose_openpose, convert_openpose_to_mediapipe
                keypoints = analyze_pose_openpose(tmp_file.name)
                landmarks_dict = convert_openpose_to_mediapipe(keypoints) if keypoints else None
                
                if landmarks_dict and keypoints:
                    with col2:
                        st.subheader("Detected Pose")
                        # Create matplotlib figure for pose visualization
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(image)
                        plot_skeleton(ax, keypoints, color='green', alpha=0.7)
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()
                    
                    # Save button
                    if st.button("Save Reference Pose"):
                        # Prepare metadata
                        metadata = {
                            'difficulty': difficulty,
                            'view_angle': view_angle,
                            'notes': notes,
                            'timestamp': datetime.now().isoformat(),
                            'image_path': uploaded_file.name
                        }
                        
                        # Save pose
                        pose_ref.save_pose(
                            exercise=exercise_type,
                            keypoints=keypoints,
                            is_correct=is_correct_form == "correct",
                            metadata=metadata
                        )
                        
                        st.success(f"Successfully saved {exercise_type} reference pose!")
                        
                        # Display statistics
                        st.subheader("Dataset Statistics")
                        stats = get_dataset_statistics(pose_ref)
                        st.json(stats)
                else:
                    st.error("Could not detect pose landmarks in the image.")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)

def get_dataset_statistics(pose_ref):
    """Get statistics about the collected dataset"""
    stats = {}
    
    for exercise in pose_ref.exercises:
        stats[exercise] = {
            'correct': len(os.listdir(os.path.join(pose_ref.base_path, exercise, 'correct'))),
            'incorrect': len(os.listdir(os.path.join(pose_ref.base_path, exercise, 'incorrect')))
        }
        stats[exercise]['total'] = stats[exercise]['correct'] + stats[exercise]['incorrect']
    
    return stats

def check_squat_form(keypoints):
    # Calculate key angles
    back_angle = calculate_angle_to_vertical(mid_shoulder - mid_hip)
    hip_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)
    knee_angle = calculate_angle(mid_hip, mid_knee, mid_ankle)
    
    # Calculate stance width
    stance_width = np.linalg.norm(r_ankle - l_ankle)
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
    stance_ratio = stance_width / shoulder_width
    
    feedback = []
    
    # Check starting position
    if back_angle > 10:
        feedback.append(f"Stand more upright. Your back is {back_angle:.1f}° from vertical")
        
    if hip_angle < 170:
        feedback.append("Fully extend your hips in the starting position")
        
    if knee_angle < 170:
        feedback.append("Start with legs straight but not locked")
        
    if stance_ratio < 1.1:
        feedback.append("Widen your stance slightly")
    elif stance_ratio > 1.5:
        feedback.append("Narrow your stance slightly")
        
    return feedback

def check_squat_starting_position(keypoints):
    return {
        'back_vertical': is_back_vertical(keypoints),
        'hips_extended': are_hips_extended(keypoints),
        'knees_straight': are_knees_straight(keypoints),
        'stance_width': check_stance_width(keypoints),
        'weight_distribution': check_weight_distribution(keypoints),
        'symmetry': check_symmetry(keypoints)
    }

if __name__ == "__main__":
    main() 