import streamlit as st
import cv2
import numpy as np
import os
from visualization import visualize_pose
import tempfile
from PIL import Image
from openpose import pyopenpose as op
from typing import Dict, List, Tuple

class ExerciseAnalyzer:
    def __init__(self, model_folder: str = "models", max_people: int = 1):
        """Initialize the exercise analyzer.
        
        Args:
            model_folder: Path to OpenPose model folder
            max_people: Maximum number of people to detect
        """
        # Initialize OpenPose
        try:
            params = {
                "model_folder": model_folder,
                "number_people_max": max_people,
                "disable_blending": False
            }
            self.opWrapper = op.WrapperPython()
            self.opWrapper.configure(params)
            self.opWrapper.start()
            self.datum = op.Datum()
            print("OpenPose initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenPose: {e}")
            raise
            
    def analyze_exercise(self, image_path: str) -> Dict:
        """Analyze an exercise image and extract pose keypoints.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing keypoints
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
                
            # Process with OpenPose
            self.datum.cvInputData = image
            self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
            
            if self.datum.poseKeypoints is None:
                raise ValueError("No pose detected in image")
                
            # Get keypoints for the first person
            keypoints = self.datum.poseKeypoints[0]
            keypoints_2d = [(x, y) for x, y, _ in keypoints]
            
            return {
                "keypoints": keypoints_2d
            }
            
        except Exception as e:
            print(f"Error analyzing exercise: {e}")
            return {
                "error": str(e)
            }

def main():
    st.title("Exercise Form Analysis")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of your exercise form",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        
        # Save image temporarily for OpenPose
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            
            try:
                # Get pose keypoints using OpenPose
                analyzer = ExerciseAnalyzer()
                result = analyzer.analyze_exercise(tmp_file.name)
                
                if result.get("error"):
                    st.error(result["error"])
                else:
                    # Visualize pose
                    output_path = "temp_visualization.png"
                    visualize_pose(
                        image_path=tmp_file.name,
                        keypoints=result["keypoints"],
                        output_path=output_path
                    )
                    st.subheader("Analysis Result")
                        
            except Exception as e:
                st.error(f"Error analyzing pose: {str(e)}")
            finally:
                # Clean up temporary files
                os.unlink(tmp_file.name)
                if os.path.exists("temp_visualization.png"):
                    os.remove("temp_visualization.png")

if __name__ == "__main__":
    main() 