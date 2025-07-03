from dataloaders.base import AbstractDataset
from dataloaders.shakespeare import Shakespeare
from dataloaders.stemp import STemp

DATASETS: dict[str, type[AbstractDataset]] = {
    "stemp": STemp,
    "shakespeare": Shakespeare,
}
