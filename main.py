from agents.agent import BrainstormingAgent
from cognition_base.database import Base, engine
from cognition_base.models import Experiment

# create tables
Base.metadata.create_all(bind=engine)


agent = BrainstormingAgent()

prompt = """You are a concise ML architecture-search assistant for 64x64 RGB EuroSAT classification (10 land-cover classes).

Start your search with an unconventional architecture to improve exploration. Try depthwise separable convolutions, residual connections, attention mechanisms, MLPs. Explore different combinations of these components to find the best architecture.
Explore existing experiments first to compare ideas.

After submitting an experiment, you will see the results, and an external agent will give you feedback.

This provides a feedback loop. Go around ~10-15 of these loops. Report a brief table of your experiments at the end, along with a short report. Feel free to take detours on architectures; no need to stay stuck on one."""

history, last_message = agent.call_agent(prompt)

print(history)

print("\n\nMessages:")
print(last_message)
