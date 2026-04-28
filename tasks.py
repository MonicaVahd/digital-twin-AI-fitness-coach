from crewai import Task
from agents import recommendation_agent, feedback_agent

# Task for generating workout plans
recommendation_task = Task(
    description="Analyze user habits, past injuries, and preferences. Suggest a diverse, adaptive workout plan with varying intensity levels. Provide alternative exercises for users who may struggle with certain movements.",
    agent=recommendation_agent,
    expected_output="A structured workout plan with exercises tailored to the user's needs."
)

# Task for monitoring posture and movement
feedback_task = Task(
    description="Analyze real-time workout data and provide feedback to correct posture and movement.",
    agent=feedback_agent,
    expected_output="Text/voice-based real-time feedback messages on workout form."
)
