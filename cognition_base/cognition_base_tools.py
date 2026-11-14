from sqlalchemy import String, cast
from sqlalchemy.orm import Session
from cognition_base.schemas import ExperimentBase
from cognition_base.models import Experiment

def unique_experiment_name(db: Session, name: str) -> str:
    """Return a name that is not yet used, appending _2, _3, ... if needed."""
    if not db.query(Experiment.id).filter(Experiment.name == name).first():
        return name

    suffix = 2
    while db.query(Experiment.id).filter(Experiment.name == f"{name}_{suffix}").first():
        suffix += 1
    return f"{name}_{suffix}"

# create a new experiment in the database
def create_experiment(db: Session, experiment: ExperimentBase):

    data = experiment.model_dump()
    data["name"] = unique_experiment_name(db, data["name"])

    db_experiment = Experiment(**data)
    db.add(db_experiment)
    db.commit()
    db.refresh(db_experiment)
    return db_experiment

# search experiments by name, spec, or description
def search_experiments(db: Session, query: str, on: str = "name", limit: int = 5):
    if on == "name":
        return db.query(Experiment).filter(Experiment.name.contains(query)).limit(limit).all()
    elif on == "spec":
        return db.query(Experiment).filter(cast(Experiment.spec, String).contains(query)).limit(limit).all()
    elif on == "description":
        return db.query(Experiment).filter(Experiment.description.contains(query)).limit(limit).all()
    else:
        raise ValueError("Invalid search field. Must be 'name', 'spec', or 'description'.")

def list_experiments(db: Session, limit: int = 10):
    return db.query(Experiment).limit(limit).all()