from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from cognition_base.database import Base


class Experiment(Base):
    """
    Base model for an experiment in the database.
    """
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    # description with max length of 100 characters
    description: Mapped[str | None] = mapped_column(String(100), nullable=True)
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    num_params: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    train_loss: Mapped[float] = mapped_column(Float)
    val_loss: Mapped[float] = mapped_column(Float)
    train_acc: Mapped[float] = mapped_column(Float)
    val_acc: Mapped[float] = mapped_column(Float)
    train_f1: Mapped[float] = mapped_column(Float)
    val_f1: Mapped[float] = mapped_column(Float)

    __tablename__ = "experiments"

    def __repr__(self) -> str:
        return f"Experiment(id={self.id}, name={self.name}, description={self.description}, spec={self.spec!r}, num_params={self.num_params}, created_at={self.created_at!r}, train_loss={self.train_loss}, val_loss={self.val_loss}, train_acc={self.train_acc}, val_acc={self.val_acc}, train_f1={self.train_f1}, val_f1={self.val_f1})"

    
