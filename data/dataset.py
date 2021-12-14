from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        X = self.data[index, :]
        y = self.data[index, :]
        return X, y

    def __len__(self):
        return len(self.data)
