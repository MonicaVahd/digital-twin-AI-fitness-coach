from crew import fitness_crew
from storage import store_ltm, store_stm, get_ltm, get_stm

def run_multi_agent_system(user_id):
    """
    Main function to run the multi-agent system.
    
    - Fetches user data from LTM
    - Generates a personalized workout plan
    - Monitors real-time workout performance
    - Stores updates for future adaptation
    """
    
    # Step 1: Store Sample Data in LTM (Past Workouts)
    store_ltm(user_id, "Strength Training, Yoga, HIIT")

    # Step 2: Store Sample Data in STM (Live Feedback)
    store_stm(user_id, "Incorrect posture detected")

    # Step 3: Retrieve Stored Data
    user_history = get_ltm(user_id)
    live_feedback = get_stm(user_id)

    # Print Retrieved Data
    print("\n Retrieved LTM Data (User History):", user_history)
    print(" Retrieved STM Data (Live Feedback):", live_feedback)

    # Step 4: Run the Multi-Agent System
    print("\n Running Multi-Agent System...")
    result = fitness_crew.kickoff()
    
    # Print Results
    print("\n System Response:", result)


if __name__ == "__main__":
    # Run the system for a sample user
    run_multi_agent_system(user_id="user1")
