# Import required libraries
import streamlit as st
import openai
import os
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

def analyze_user_input(user_input):
    # Initialize temp_image_path
    timestamp = int(time.time())
    temp_image_path = os.path.join(TEMP_IMAGES, f"pose_{timestamp}.png")
    
    try:
        # Handle different input types
        if isinstance(user_input, list):
            print("Processing OpenPose keypoints...")
            
            # Process and validate keypoints
            processed_keypoints = process_keypoints(user_input)
            if processed_keypoints is None:
                return {"message": "Invalid or insufficient keypoints"}
                
            # Convert to MediaPipe format
            mediapipe_keypoints = convert_openpose_to_mediapipe(processed_keypoints)
            if mediapipe_keypoints is None:
                return {"message": "Failed to convert pose data"}
                
            # Process the keypoints
            landmarks_dict = process_keypoints(mediapipe_keypoints)
            if landmarks_dict is None:
                return {"message": "Failed to process keypoints"}
                
            # Analyze the pose
            analysis = analyze_pose_mediapipe(landmarks_dict)
            if analysis is None:
                return {"message": "No pose detected in image"}
                
            # Generate feedback
            try:
                # Generate analysis text
                analysis_text = generate_concise_feedback(analysis)
                
                # Generate audio feedback
                audio_path = text_to_speech(analysis_text)
                
                # Create visualization
                output_buffer = io.BytesIO()
                visualize_pose_comparison(None, landmarks_dict, analysis_text)
                plt.savefig(output_buffer, format='png', bbox_inches='tight')
                plt.close()
                
                # Save result
                output_dir = "correct_form_images"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"analysis_{user_id}.png")
                
                with open(output_path, 'wb') as f:
                    f.write(output_buffer.getvalue())
                    
                print(f"Analysis saved to: {output_path}")
                
                return {
                    "image_path": output_path,
                    "audio_path": audio_path,
                    "message": "Analysis complete",
                    "analysis": analysis_text,
                    "full_analysis": analysis
                }
                
            except Exception as e:
                print(f"Error generating analysis: {str(e)}")
                return {
                    "image_path": image_path,
                    "message": "Image processed but analysis failed",
                    "analysis": "Error generating detailed analysis"
                }
        else:
            return {"message": "Invalid input type"}
            
    except Exception as e:
        return {"message": f"Error processing input: {str(e)}"} 