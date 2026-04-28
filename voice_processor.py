import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import openai
from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
import json
from gtts import gTTS
import pygame
import io
import time
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import psycopg2
import redis
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@dataclass
class VoiceResponse:
    text: str
    audio: Optional[np.ndarray] = None
    emotion: Optional[str] = None
    confidence: float = 0.0

class VoiceProcessor:
    def __init__(self):
        try:
            # Initialize pygame mixer with error handling
            try:
                pygame.mixer.init()
                logger.info("Successfully initialized pygame mixer")
            except Exception as e:
                logger.warning(f"Failed to initialize pygame mixer: {str(e)}")
            
            # Set default sample rate and channels
            self.sample_rate = 44100
            self.channels = 1
            
            # Try to find a working input device
            self.input_device = self.find_working_input_device()
            if self.input_device is None:
                logger.warning("No working input device found, will use default device")
                self.input_device = None  # Let sounddevice use the default device
            
        except Exception as e:
            logger.error(f"Error initializing VoiceProcessor: {str(e)}")
            raise

    def find_working_input_device(self):
        """Find a working input device"""
        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                logger.info(f"Device {i}: {device['name']}")
                logger.info(f"  Input channels: {device['max_input_channels']}")
                logger.info(f"  Output channels: {device['max_output_channels']}")
                
                if device['max_input_channels'] > 0:
                    try:
                        # Test the device
                        sd.check_input_settings(device=i)
                        logger.info(f"Found working input device: {device['name']}")
                        return i
                    except Exception as e:
                        logger.warning(f"Device {i} ({device['name']}) failed test: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error finding working input device: {str(e)}")
            return None

    def record_audio(self, duration=10):
        """Record audio with better error handling"""
        try:
            logger.info(f"Recording audio for {duration} seconds...")
            
            # Try recording with the selected device
            try:
                recording = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    device=self.input_device
                )
                sd.wait()
            except Exception as e:
                logger.error(f"Failed to record with selected device: {str(e)}")
                # Try with default device
                try:
                    recording = sd.rec(
                        int(duration * self.sample_rate),
                        samplerate=self.sample_rate,
                        channels=self.channels
                    )
                    sd.wait()
                except Exception as e:
                    logger.error(f"Failed to record with default device: {str(e)}")
                    return None, None
            
            # Create a temporary file for the recording
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, recording, self.sample_rate)
            logger.info(f"Audio recorded and saved to {temp_file.name}")
            
            return recording, temp_file.name
            
        except Exception as e:
            logger.error(f"Error recording audio: {str(e)}")
            return None, None

    def text_to_speech(self, text):
        """Convert text to speech with better error handling"""
        try:
            if not text:
                logger.warning("Empty text provided for text-to-speech")
                return None

            logger.info(f"Converting text to speech: {text[:50]}...")
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Create a temporary file for the audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tts.save(temp_file.name)
            logger.info(f"Text-to-speech audio saved to {temp_file.name}")
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error in text-to-speech: {str(e)}")
            return None

    def play_audio(self, audio_file):
        """Play audio file with better error handling"""
        try:
            if not audio_file or not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return False

            logger.info(f"Playing audio file: {audio_file}")
            
            # Try pygame first
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                return True
            except Exception as e:
                logger.warning(f"Failed to play audio with pygame: {str(e)}")
                # Try using aplay as fallback
                try:
                    subprocess.run(['aplay', audio_file], check=True)
                    return True
                except Exception as e:
                    logger.error(f"Failed to play audio with aplay: {str(e)}")
                    return False
                
        except Exception as e:
            logger.error(f"Error playing audio: {str(e)}")
            return False

    def transcribe_with_whisper(self, audio_file_path):
        client = OpenAI(api_key=OPENAI_API_KEY)
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text

    def process_voice_input(self, duration=10):
        """Process voice input with better error handling"""
        try:
            # Record audio
            recording, audio_file = self.record_audio(duration)
            if recording is None or audio_file is None:
                logger.error("Failed to record audio")
                return None, None

            # For now, return a placeholder transcription
            # In a real application, you would use a speech-to-text service here
            transcription = self.transcribe_with_whisper(audio_file)
            
            return audio_file, transcription
            
        except Exception as e:
            logger.error(f"Error processing voice input: {str(e)}")
            return None, None

    def cleanup(self, audio_file):
        """Clean up temporary audio files"""
        try:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)
                logger.info(f"Cleaned up audio file: {audio_file}")
        except Exception as e:
            logger.error(f"Error cleaning up audio file: {str(e)}")

    def _init_apis(self):
        """Initialize API clients for Whisper."""
        try:
            # Initialize OpenAI client for Whisper
            openai.api_key = OPENAI_API_KEY
            if not openai.api_key:
                logger.warning("OpenAI API key not found. Whisper functionality will be disabled.")
                
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            
    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper API.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcribed text or None if failed
        """
        try:
            if not OPENAI_API_KEY:
                logger.error("OpenAI API key not found")
                return None
                
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, self.sample_rate)
                
                # Transcribe with Whisper
                with open(temp_file.name, 'rb') as audio_file:
                    response = openai.Audio.transcribe(
                        model='whisper-1',
                        file=audio_file
                    )
                    
            # Clean up temporary file
            os.unlink(temp_file.name)
            
            return response['text']
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return None
            
    def _generate_response(self, text: str) -> str:
        """Generate response using LLM.
        
        Args:
            text: Input text
            
        Returns:
            Generated response text
        """
        # TODO: Implement LLM response generation
        # For now, return a simple response
        return f"I heard you say: {text}. How can I help you with your exercise?" 

    def extract_opensmile_features(self, wav_path, output_csv=None):
        """
        Extract paralinguistic features from a WAV file using OpenSmile.
        Returns a pandas Series of features.
        """
        if output_csv is None:
            output_csv = wav_path.replace('.wav', '_features.csv')
        opensmile_bin = os.getenv(
            "OPENSMILE_BIN",
            os.path.join(os.path.dirname(__file__), "opensmile-master", "opensmile-master",
                         "build", "progsrc", "smilextract", "SMILExtract")
        )
        config_path = os.getenv(
            "OPENSMILE_CONFIG",
            os.path.join(os.path.dirname(__file__), "opensmile-master", "opensmile-master",
                         "config", "gemaps", "v01a", "GeMAPSv01a.conf")
        )
        cmd = [
            opensmile_bin,
            "-C", config_path,
            "-I", wav_path,
            "-O", output_csv
        ]
        try:
            subprocess.run(cmd, check=True)
            df = pd.read_csv(output_csv, sep=';')
            # Clean up the output CSV after reading
            os.remove(output_csv)
            return df.iloc[0]  # Return the features as a pandas Series
        except Exception as e:
            logger.error(f"OpenSmile feature extraction failed: {e}")
            return None 

    def process_and_save_feedback(self, user_id, audio_path=None, duration=5):
        from voice_emotion_detector import predict_emotion_from_csv

        # 1. Use provided audio file if available, else record live
        if audio_path is not None:
            audio_file = audio_path
            transcription = self.transcribe_with_whisper(audio_file)
        else:
            audio_file, transcription = self.process_voice_input(duration=duration)
            if audio_file is None:
                logger.error("No audio file recorded.")
                return None

        # 2. Extract features
        features = self.extract_opensmile_features(audio_file)
        if features is None:
            logger.error("No features extracted.")
            return None

        # 3. Save features to a temp CSV for emotion detector
        temp_csv = audio_file.replace('.wav', '_features.csv')
        predicted_emotion, scores = predict_emotion_from_csv(temp_csv)
        feedback = generate_feedback(predicted_emotion)
        save_feedback_postgres(user_id, predicted_emotion, feedback, transcription, audio_file)
        save_feedback_redis(user_id, predicted_emotion, feedback, transcription, audio_file)
        print("Predicted emotion:", predicted_emotion)
        print("Feedback:", feedback)
        return predicted_emotion, feedback, scores

def save_feedback_postgres(user_id, emotion, feedback, transcription, audio_path):
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "ltm_database"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            user_id TEXT,
            timestamp TIMESTAMP,
            emotion TEXT,
            feedback TEXT,
            transcription TEXT,
            audio_path TEXT
        )
    """)
    cur.execute("""
        INSERT INTO feedback (user_id, timestamp, emotion, feedback, transcription, audio_path)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (user_id, datetime.now(), emotion, feedback, transcription, audio_path))
    conn.commit()
    cur.close()
    conn.close()

def save_feedback_redis(user_id, emotion, feedback, transcription, audio_path):
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=0
    )
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "emotion": emotion,
        "feedback": feedback,
        "transcription": transcription,
        "audio_path": audio_path
    }
    r.rpush(f"user:{user_id}:feedback", json.dumps(feedback_data)) 

def generate_feedback(emotion):
    feedback_dict = {
        "happy": "Great energy! Keep it up!",
        "sad": "You sound a bit down. Remember, every effort counts!",
        "angry": "Try to relax and focus on your breathing.",
        "calm": "Nice calm tone! Keep going.",
        "fearful": "Take your time, you're doing well.",
        "disgust": "If something is bothering you, let us know.",
        "surprised": "Surprised? Let's keep the momentum!",
        "neutral": "Let's add some energy to your session!"
    }
    return feedback_dict.get(emotion, "Keep going, you're doing great!") 