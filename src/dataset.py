import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import config
# 1.定义dataset
class TranslationDataset(Dataset):
    def __init__(self,data_path):
        # print(data_path)
        self.data = pd.read_json(data_path,lines=True,orient='records').to_dict(orient="records")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'],dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'],dtype=torch.long)
        return input_tensor,target_tensor

def get_dataloader(train=True):
    data_path = config.PROCESS_DATA_DIR /('indexed_train.jsonl' if train else 'indexed_test.jsonl')
    dataset = TranslationDataset(data_path)
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    print(f'batch个数：{len(train_dataloader)}')
    test_dataloader = get_dataloader(train=False)
    print(f'batch个数：{len(test_dataloader)}')
    for inputs,targets in train_dataloader:
        print(inputs.shape)     #inputs.shape:[batch_size, seq_len]
        print(targets.shape)    #tarhets.shape:[batch_size]
        break