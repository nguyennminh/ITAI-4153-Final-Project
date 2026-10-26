"""Package marker. # PROVIDED"""

import os

# MLflow 3 prints a log line meant for AI coding agents; it is not about your project.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
