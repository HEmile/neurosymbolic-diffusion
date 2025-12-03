from typing import Tuple

import numpy as np
import torch


class WCDataSet(torch.utils.data.Dataset):

    def __init__(self, N: int, base_path="data/", type="train"):
        super().__init__()
        self.N = N
        self.maps = torch.tensor(np.load(base_path + f"warcraft_shortest_path/{N}x{N}/{type}_maps.npy"), dtype=torch.float)
        self.paths = torch.tensor(np.load(base_path + f"warcraft_shortest_path/{N}x{N}/{type}_shortest_paths.npy"))\
            .reshape(-1, self.N * self.N)
        weights = np.load(base_path + f"warcraft_shortest_path/{N}x{N}/{type}_vertex_weights.npy")\
            .reshape(-1, self.N * self.N)
        self.weights = torch.tensor(weights, dtype=torch.float)
        self.num_classes = len(np.unique(self.weights))
        self.num_maps = self.maps.shape[0]

        # Apply transformations on maps
        self.maps = self.maps.permute(0, 3, 1, 2)
        mean, std = (
            torch.mean(self.maps, dim=(0, 2, 3), keepdim=True),
            torch.std(self.maps, dim=(0, 2, 3), keepdim=True),
        )
        self.maps = (self.maps - mean) / std

    def __len__(self):
        return self.num_maps

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.maps[item], self.paths[item], self.weights[item]

def get_datasets(N, basepath="data/"):
    train_dataset = WCDataSet(N, base_path=basepath, type="train")
    val_dataset = WCDataSet(N, base_path=basepath, type="val")
    test_dataset = WCDataSet(N, base_path=basepath, type="test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    N = 12
    maps = np.load("warcraft_shortest_path/12x12/train_maps.npy")
    paths = np.load("warcraft_shortest_path/12x12/train_shortest_paths.npy")
    weights = np.load("warcraft_shortest_path/12x12/train_vertex_weights.npy")

    print(maps.shape)
    print(paths.shape)
    print(weights.shape)

    print(np.unique(weights))

    print(paths[0])