import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import openai
from pose_detection import DetectionResult, DetectionMethod
from voice_processor import VoiceResponse
from emotion_detector import EmotionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserState:
    pose_quality: float
    emotion_valence: float
    emotion_arousal: float
    voice_confidence: float
    fatigue_level: float
    engagement_level: float

@dataclass
class FeedbackResponse:
    text: str
    audio: Optional[np.ndarray] = None
    emotion: Optional[str] = None
    confidence: float = 0.0
    action_type: str = "neutral"  # encourage, correct, rest, question

class AdaptiveFeedback:
    def __init__(self, config: Dict = None):
        """Initialize adaptive feedback system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'openai_model': 'gpt-4',
            'prompt_templates': {
                'encourage': "The user is showing signs of fatigue. Provide encouraging feedback.",
                'correct': "The user's form needs correction. Provide specific form feedback.",
                'rest': "The user needs a rest. Suggest a short break.",
                'question': "Ask the user how they're feeling."
            },
            'thresholds': {
                'fatigue': 0.7,
                'engagement': 0.3,
                'form_quality': 0.6
            }
        }
        
        # Initialize OpenAI
        self._init_openai()
        
        # Initialize state tracking
        self.user_state = UserState(
            pose_quality=1.0,
            emotion_valence=0.0,
            emotion_arousal=0.0,
            voice_confidence=1.0,
            fatigue_level=0.0,
            engagement_level=1.0
        )
        
        # Initialize feedback history
        self.feedback_history = []
        
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                logger.warning("OpenAI API key not found. LLM feedback will be disabled.")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            
    def update_state(self, 
                    pose_result: Optional[DetectionResult] = None,
                    voice_result: Optional[VoiceResponse] = None,
                    emotion_result: Optional[EmotionResult] = None):
        """Update user state with new observations.
        
        Args:
            pose_result: Pose detection result
            voice_result: Voice processing result
            emotion_result: Emotion detection result
        """
        try:
            # Update pose quality
            if pose_result:
                self.user_state.pose_quality = pose_result.confidence
                
            # Update emotion state
            if emotion_result:
                self.user_state.emotion_valence = emotion_result.valence
                self.user_state.emotion_arousal = emotion_result.arousal
                
            # Update voice confidence
            if voice_result:
                self.user_state.voice_confidence = voice_result.confidence
                
            # Calculate fatigue level
            self.user_state.fatigue_level = self._calculate_fatigue()
            
            # Calculate engagement level
            self.user_state.engagement_level = self._calculate_engagement()
            
        except Exception as e:
            logger.error(f"Failed to update state: {e}")
            
    def _calculate_fatigue(self) -> float:
        """Calculate fatigue level from current state."""
        # Combine multiple factors
        factors = [
            (1 - self.user_state.pose_quality) * 0.4,  # Poor form
            (1 - self.user_state.emotion_arousal) * 0.3,  # Low energy
            (1 - self.user_state.voice_confidence) * 0.3  # Weak voice
        ]
        return min(sum(factors), 1.0)
        
    def _calculate_engagement(self) -> float:
        """Calculate engagement level from current state."""
        # Combine multiple factors
        factors = [
            self.user_state.pose_quality * 0.4,  # Good form
            self.user_state.emotion_valence * 0.3,  # Positive emotion
            self.user_state.voice_confidence * 0.3  # Clear voice
        ]
        return min(sum(factors), 1.0)
        
    def generate_feedback(self) -> Optional[FeedbackResponse]:
        """Generate adaptive feedback based on current state.
        
        Returns:
            FeedbackResponse containing feedback text and metadata
        """
        try:
            # Determine action type
            action_type = self._determine_action_type()
            
            # Generate feedback text
            feedback_text = self._generate_feedback_text(action_type)
            
            # Create response
            response = FeedbackResponse(
                text=feedback_text,
                action_type=action_type,
                confidence=0.8  # Placeholder confidence
            )
            
            # Add to history
            self.feedback_history.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate feedback: {e}")
            return None
            
    def _determine_action_type(self) -> str:
        """Determine appropriate action type based on state."""
        if self.user_state.fatigue_level > self.config['thresholds']['fatigue']:
            return "rest"
        elif self.user_state.pose_quality < self.config['thresholds']['form_quality']:
            return "correct"
        elif self.user_state.engagement_level < self.config['thresholds']['engagement']:
            return "encourage"
        else:
            return "question"
            
    def _generate_feedback_text(self, action_type: str) -> str:
        """Generate feedback text using LLM.
        
        Args:
            action_type: Type of feedback to generate
            
        Returns:
            Generated feedback text
        """
        try:
            if not openai.api_key:
                return self._get_fallback_feedback(action_type)
                
            # Prepare prompt
            prompt = self.config['prompt_templates'][action_type]
            
            # Add context
            context = {
                'pose_quality': self.user_state.pose_quality,
                'fatigue_level': self.user_state.fatigue_level,
                'engagement_level': self.user_state.engagement_level,
                'emotion': 'positive' if self.user_state.emotion_valence > 0 else 'negative'
            }
            
            # Call OpenAI
            response = openai.ChatCompletion.create(
                model=self.config['openai_model'],
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(context)}
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate feedback text: {e}")
            return self._get_fallback_feedback(action_type)
            
    def _get_fallback_feedback(self, action_type: str) -> str:
        """Get fallback feedback when LLM is unavailable."""
        fallback_responses = {
            'encourage': "You're doing great! Keep up the good work!",
            'correct': "Let's focus on maintaining proper form.",
            'rest': "Take a short break to recover your energy.",
            'question': "How are you feeling? Do you need any adjustments?"
        }
        return fallback_responses.get(action_type, "Keep going!") 