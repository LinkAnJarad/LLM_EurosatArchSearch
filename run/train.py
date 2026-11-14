import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from eurosat_loader.loaders import generate_eurosat_dataloader
from utils import num_params
from .config import Config, build_optimizer, build_scheduler

from tqdm import tqdm

from models.dummy import DummyModel

from .config import TEST


def _metrics(labels: torch.Tensor, logits: torch.Tensor) -> tuple[float, float]:
    """Accuracy and macro-F1 from a full epoch of logits."""
    target = labels.cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(target, pred), f1_score(target, pred, average="macro")


def train_step(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_fn,
    device="cuda",
) -> dict:
    """One training epoch; returns epoch loss / accuracy / macro-F1."""
    model.train()

    epoch_loss = 0.0
    all_logits, all_labels = [], []

    for image, label in tqdm(train_loader, desc="train", leave=False):

        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad(set_to_none=True)

        preds = model(image)

        batch_loss = loss_fn(preds, label)

        batch_loss.backward()

        optimizer.step()

        epoch_loss += batch_loss.item()

        all_logits.append(preds.detach())
        all_labels.append(label)

        scheduler.step()

    epoch_loss /= len(train_loader)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    epoch_acc, epoch_f1 = _metrics(labels, logits)

    return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}


def test_step(
    model,
    test_loader,
    loss_fn,
    device="cuda",
) -> dict:
    """Evaluate on validation or test; returns loss / accuracy / macro-F1."""
    model.eval()

    epoch_loss = 0.0
    all_logits, all_labels = [], []

    with torch.inference_mode():

        for image, label in tqdm(test_loader, desc="eval", leave=False):

            image = image.to(device)
            label = label.to(device)

            preds = model(image)

            epoch_loss += loss_fn(preds, label).item()

            all_logits.append(preds)
            all_labels.append(label)

    epoch_loss /= len(test_loader)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    epoch_acc, epoch_f1 = _metrics(labels, logits)

    return {"loss": epoch_loss, "acc": epoch_acc, "f1": epoch_f1}


train_loader = generate_eurosat_dataloader(split="train", batch_size=Config.BATCH_SIZE)
val_loader = generate_eurosat_dataloader(split="validation", batch_size=Config.BATCH_SIZE)
test_loader = generate_eurosat_dataloader(split="test", batch_size=Config.BATCH_SIZE)

loss_fn = nn.CrossEntropyLoss()


def run(
    model: nn.Module,
    epochs: int = Config.EPOCHS,
    device: Optional[str] = None,
    loaders: Optional[tuple] = None,
    name: Optional[str] = None,
    save_model: Optional[bool] = False,
) -> dict:
    """Train + evaluate one architecture

    loaders: optional (train, val, test)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if loaders is None:
        loaders = (train_loader, val_loader, test_loader)
    train_dl, val_dl, test_dl = loaders

    model = model.to(device)
    name = name or model.__class__.__name__  # label for the results table
    params = num_params(model)

    optimizer = build_optimizer(model.parameters())
    # Scheduler is stepped once per batch in train_step, so T_max is total
    # optimizer steps across all epochs. Use train_dl, not the global default,
    # so custom/subset loaders get the correct schedule length.
    scheduler = build_scheduler(optimizer, total_steps=epochs * len(train_dl))

    history: list = []
    best: dict = {"epoch": 0, "val": None, "weights": None}

    for epoch in range(1, epochs + 1):
        train = train_step(model, train_dl, optimizer, scheduler, loss_fn, device)
        val = test_step(model, val_dl, loss_fn, device)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train,
            "val": val,
        }
        history.append(row)

        if best["val"] is None or val["loss"] < best["val"]["loss"]:
            best = {"epoch": epoch, "val": val, "weights": copy.deepcopy(model.state_dict())}

        print(
            f"{name} epoch {epoch}/{epochs} lr={row['lr']:.2e} "
            f"train={train} val={val}"
        )

    model.load_state_dict(best["weights"])  # report metrics of the best-val model
    train_best = history[best["epoch"] - 1]["train"]

    if save_model:
        torch.save(model.state_dict(), f"models/{name}.pt")

    results = {
        "model": name,
        "params": params,
        "epochs": epochs,
        "best_epoch": best["epoch"],
        "train_loss": train_best["loss"],
        "train_acc": train_best["acc"],
        "train_f1": train_best["f1"],
        "val_loss": best["val"]["loss"],
        "val_acc": best["val"]["acc"],
        "val_f1": best["val"]["f1"],
        "test": test_step(model, test_dl, loss_fn, device),
        "history": history,
    }

    print(
        f"{name} best epoch {results['best_epoch']} params={params:,} "
        f"train(acc={results['train_acc']:.4f}, f1={results['train_f1']:.4f}, "
        f"loss={results['train_loss']:.4f}) "
        f"val(acc={results['val_acc']:.4f}, f1={results['val_f1']:.4f}, "
        f"loss={results['val_loss']:.4f})"
    )
    return results


if __name__ == "__main__":
    results = run(DummyModel(), epochs=2 if TEST else Config.EPOCHS)
    print(f"test={results['test']}")
