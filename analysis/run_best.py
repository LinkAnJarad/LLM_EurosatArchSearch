from cognition_base.database import SessionLocal
from cognition_base.models import Experiment
import pandas as pd
from sqlalchemy import func

from run.train import run
from models.skeleton import build_skeleton

import json

with SessionLocal() as db:
    best_experiment = db.query(Experiment).order_by(Experiment.val_f1.desc()).first()

best_spec = best_experiment.spec

# save best spec to json
with open("models/best_model_spec.json", "w") as f:
    json.dump(best_spec, f, indent=4)

model = build_skeleton(best_spec)

print("Training best model from experiment:", best_experiment.name)

results = run(
    model,
    epochs=20, # override config
    save_model=True
)

# save results to json
with open("models/best_model_results.json", "w") as f:
    json.dump(results, f, indent=4)

