from typing import Callable, Optional

from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


DEFAULT_DATASET_ID = "blanchon/EuroSAT_RGB"

SPLITS = ("train", "validation", "test")

# Channel statistics computed on the EuroSAT RGB train split.
EUROSAT_MEAN = (0.444411, 0.428012, 0.428117)
EUROSAT_STD = (0.240878, 0.222481, 0.224839)


class _EuroSATDataset(Dataset):
    """Wrapper around an Hugging Face EuroSAT split for PyTorch DataLoaders."""

    def __init__(self, dataset: HFDataset, transform: Optional[Callable] = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        sample = self.dataset[index]

        image = sample["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = int(sample["label"])
        return image, label


def build_transform(train: bool) -> transforms.Compose:
    """ToTensor + normalization, with light augmentation for the train split.

    Affine and colour-jitter magnitudes are deliberately small: EuroSAT tiles are
    already 64x64 crops, so strong warps or colour shifts would destroy context.
    """
    steps: list[Callable] = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(EUROSAT_MEAN, EUROSAT_STD)]
    return transforms.Compose(steps)


def generate_eurosat_dataloader(
    split: str = "train",
    batch_size: int = 64,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> DataLoader:
    """Build a DataLoader for one EuroSAT split: train, validation or test."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}.")

    hf_split = load_dataset(dataset_id, split=split)
    dataset = _EuroSATDataset(hf_split, transform=build_transform(train=split == "train"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" if shuffle is None else shuffle,
        num_workers=num_workers,
    )
