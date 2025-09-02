import time
from itertools import chain

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import nn
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel
import dataset


def train_one_epoch(dataloader,model, optimizer, loss_function, device):
    model.train()
    epoch_total_loss = 0
    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)  # [batch_size, seq_len]
        targets = targets.to(device)  # [batch_size, seq_len]
        optimizer.zero_grad()

        decoder_input = targets[:,0:-1]
        # 源序列pad
        src_pad_mask = (inputs==model.src_embedding.padding_idx)
        # 目标序列pad
        tgt_pad_mask = (decoder_input==model.tgt_embedding.padding_idx)
        # tgt_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)

        decoder_output = model(inputs,decoder_input,src_pad_mask,tgt_pad_mask,tgt_mask) # [batch_size, seq_len-1, en_vocab_size]
        decoder_targets = targets[:,1:] # [batch_size, seq_len-1, en_vocab_size]

        # 计算损失
        loss = loss_function(decoder_output.reshape(-1,decoder_output.shape[-1]), decoder_targets.reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_total_loss += loss.item()

    return epoch_total_loss


def train():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab_txt')
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_id,
                             en_padding_idx =en_tokenizer.pad_token_id).to(device)
    dataloader = dataset.get_dataloader()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float('inf')
    for epoch in range(1, 1 + config.EPOCHS):
        print(f"=========== epoch:{epoch} ===========")
        avg_loss = train_one_epoch(dataloader,model, optimizer, loss_function, device)
        print(f"loss:{avg_loss:.4f}")
        writer.add_scalar("Loss", avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print("模型保存成功")
        else:
            print("模型无需保存！")


if __name__ == '__main__':
    train()
