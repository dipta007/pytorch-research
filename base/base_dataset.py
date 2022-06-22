import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, paths, fields=["x", "y"]):
        super().__init__()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collate_fn(self, batch):
        """
        batch = [
            [(text1, text1_len), (text2, text2_len), ..., target],
            [(text1, text1_len), (text2, text2_len), ..., target],
        ]
        last one must be the target, if there is no pass -1 as target (i.e. unsupervised)
        """
        # ? sorting
        # batch.sort(key=lambda x: x[0][1], reverse=True)

        # #? zipping to collate inside batch
        # values = list(zip(*batch))
        # x = (tuple(zip(*x)) for x in values[:-1])
        # y = values[-1]

        # #? padding and make tensor
        # x = [self.tokenizer(x[0]), torch.tensor(x[1]) for x in x]
        # y = torch.tensor(y)

        # #? make dictionary
        # ret = {x: y for x, y in zip(self.fields, x)}
        # ret['target'] = y
        ret = {}

        return ret
