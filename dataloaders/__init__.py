from dataloaders.base import AbstractDataset
from dataloaders.stemp import STemp

DATASETS: dict[str, type[AbstractDataset]] = {
    "stemp": STemp,
}
