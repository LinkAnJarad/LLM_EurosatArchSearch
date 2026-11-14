from cognition_base.database import SessionLocal
from cognition_base.models import Experiment
import pandas as pd
from sqlalchemy import inspect

import seaborn as sns
import matplotlib.pyplot as plt

with SessionLocal() as db:
    all_experiments = db.query(Experiment).order_by(Experiment.created_at.asc()).all()

# convert to pandas dataframe
def object_as_dict(obj):
  return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}

df = pd.DataFrame([object_as_dict(e) for e in all_experiments])


# step graph and save

step_graph_y = list(df.val_f1)
step_graph_y = [
    max([step_graph_y[i]] + step_graph_y[:i]) for i in range(len(step_graph_y))
]
df["best_running_val_f1"] = step_graph_y

ax = sns.lineplot(data=df, x="id", y="best_running_val_f1", drawstyle="steps-post", marker="o")
plt.xlabel("Iteration")
plt.ylabel("Best Val F1 Score")
plt.title("Best Running Validation F1 Score Over Iterations")
plt.savefig("analysis/step_graph.png")