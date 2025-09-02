import math

import torch
from torch import nn
import config


class PositionEncoding(nn.Module):
    def __init__(self, dim_model, max_len=100):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model, dtype=torch.float)
        for pos in range(max_len):
            for _2i in range(0, dim_model, 2):
                pe[pos, _2i] = math.sin(pos / math.pow(10000.0, _2i / dim_model))
                # if _2i+1<dim_model:
                pe[pos, _2i + 1] = math.cos(pos / math.pow(10000.0, _2i / dim_model))

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x [batch_size, seq_len, d_model]
        seq_len = x.shape[1]
        pe_part = self.pe[:seq_len]  # [seq_len, d_model]
        x = x + pe_part
        return x + pe_part


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_idx, en_padding_idx):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=zh_vocab_size,
                                          embedding_dim=config.DIM_MODEL,
                                          padding_idx=zh_padding_idx)
        self.tgt_embedding = nn.Embedding(num_embeddings=en_vocab_size,
                                          embedding_dim=config.DIM_MODEL,
                                          padding_idx=en_padding_idx)

        self.position_encoding = PositionEncoding(dim_model=config.DIM_MODEL)

        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
                                          nhead=config.NUM_HEADS,
                                          num_encoder_layers=config.NUM_ENCODER_LAYERS,
                                          num_decoder_layers=config.NUM_DECODER_LAYERS,
                                          batch_first=True)
        self.linear = nn.Linear(in_features=config.DIM_MODEL, out_features=en_vocab_size)

    def encode(self, src, src_key_padding_mask):
        # src.shape:[batch_size,seq_len
        src_embed = self.src_embedding(src)  # [batch_size, seq_len, d_model]
        src_embed = self.position_encoding(src_embed)  # [batch_size, seq_len, d_model]
        memory = self.transformer.encoder(src=src_embed, src_key_padding_mask=src_key_padding_mask)
        return memory  # [batch_size, seq_len, d_model]

    def decode(self, tgt, memory, tgt_mask, memory_pad_mask, tgt_pad_mask):
        tgt_embed = self.tgt_embedding(tgt)
        tgt_embed = self.position_encoding(tgt_embed)  # [batch_size, seq_len, d_model]
        output = self.transformer.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=memory_pad_mask)
        # [batch_size, seq_len, d_model]
        output = self.linear(output)  # [batch_size, seq_len, en_vocab_size]
        return output

    def forward(self, src, tgt, src_key_padding_mask, tgt_pad_mask,tgt_mask):
        memory = self.encode(src, src_key_padding_mask)
        output = self.decode(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_pad_mask=src_key_padding_mask,
                             tgt_pad_mask=tgt_pad_mask)

        return output
