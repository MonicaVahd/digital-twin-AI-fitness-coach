from crewai import Crew
from agents import orchestration_agent, recommendation_agent, feedback_agent
from tasks import recommendation_task, feedback_task

# Define the Crew (team of agents)
fitness_crew = Crew(
    agents=[orchestration_agent, recommendation_agent, feedback_agent],
    tasks=[recommendation_task, feedback_task],
    verbose=True
)
