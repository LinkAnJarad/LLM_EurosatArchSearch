from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

class ExperimentBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="The name of the experiment")
    description: Optional[str] = Field(None, description="A brief description of the experiment")
    spec: dict[str, Any] = Field(..., description="Declarative architecture spec for models.BaseModel")
    num_params: int = Field(..., description="The number of parameters in the experiment")
    train_loss: float = Field(..., description="The training loss of the experiment")
    val_loss: float = Field(..., description="The validation loss of the experiment")
    train_acc: float = Field(..., description="The training accuracy of the experiment")
    val_acc: float = Field(..., description="The validation accuracy of the experiment")
    train_f1: float = Field(..., description="The training F1 score of the experiment")
    val_f1: float = Field(..., description="The validation F1 score of the experiment")


class AgentSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="The name of the experiment")
    description: Optional[str] = Field(None, description="A brief description of the experiment")
    spec: dict[str, Any] = Field(..., description="Declarative architecture spec for models.BaseModel")