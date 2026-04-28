import random
from crew import fitness_crew
from storage import store_ltm, store_stm, get_ltm, get_stm

workouts = ["Strength Training, Yoga, HIIT", "Pilates, Cardio, Core Training", "Cycling, Weightlifting, Kickboxing"]
feedbacks = ["Incorrect posture detected", "Great form! Keep it up!", "Slow down and engage core more"]

# Store random workout history & feedback
store_ltm("user1", random.choice(workouts))
store_stm("user1", random.choice(feedbacks))

# Retrieve stored data
user_history = get_ltm("user1")
live_feedback = get_stm("user1")

# Print Retrieved Data
print("\n Retrieved LTM Data (User History):", user_history)
print(" Retrieved STM Data (Live Feedback):", live_feedback)

# Run the Multi-Agent System
print("\n Running Multi-Agent System...")
result = fitness_crew.kickoff()

# Print Results
print("\n System Response:", result)
