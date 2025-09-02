import torch
from torch import nn

from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationModel


def predict_batch(input_tensor, model, zh_tokenizer, en_tokenizer, device):
    """
    批量预测
    :param input_tensor: [batch_size, seq_len]
    :param model:
    :param zh_tokenizer:
    :param en_tokenizer:
    :param device:
    :return:一批英文句子 e.g.:[[14,212],[2132,21,323,123,122],[213,221,54],...]
    """
    model.eval()

    with torch.no_grad():
        # 编码
        src_pad_mask = (input_tensor==zh_tokenizer.pad_token_id)
        memory = model.encode(input_tensor,src_pad_mask)  # [batch_size, seq_len, d_model]
        batch_size = input_tensor.shape[0]

        # 解码
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_id, device=device)
        #[batch_size, 1]

        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(config.SEQ_LEN):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            tgt_pad_mask = decoder_input==model.tgt_embedding.padding_idx
            decoder_outputs = model.decode(decoder_input,memory,tgt_mask,src_pad_mask,tgt_pad_mask)
            # decoder_outputs: [batch_size, tgt_len, vocab_size]
            last_decoder_output = decoder_outputs[:,-1,:] # [batch_size, 1, vocab_size]
            predict_indexes = torch.argmax(last_decoder_output, dim=-1)  # [batch_size]
            # 处理每个时间步的预测结果
            for i in range(batch_size):
                if is_finished[i]:
                    continue
                else:
                    if predict_indexes[i].item() == en_tokenizer.eos_token_id:
                        is_finished[i] = True
                        continue
                    else:
                        generated[i].append(predict_indexes[i].item())
                if all(is_finished):
                    break

            decoder_input = torch.cat([decoder_input,predict_indexes.unsqueeze(1)],dim=1)
        return generated


def predict(user_input, model, zh_tokenizer, en_tokenizer, device):
    # 处理数据
    index_list = zh_tokenizer.encode(user_input, config.SEQ_LEN)
    input_tensor = torch.tensor([index_list]).to(device)
    batch_result = predict_batch(input_tensor,model, zh_tokenizer, en_tokenizer, device)
    result = batch_result[0]
    return en_tokenizer.decode(result)
    # return result


def run_predict():
    # 准备资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab_txt')

    # 准备模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
                             en_vocab_size=en_tokenizer.vocab_size,
                             zh_padding_idx=zh_tokenizer.pad_token_id,
                             en_padding_idx=en_tokenizer.pad_token_id).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    print("欢迎使用中英翻译！（输入q或者quit退出）")
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("程序已退出！")
            break
        if user_input.strip() == '':
            print('请输入中文')
            continue
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f"英文：{result}")


if __name__ == '__main__':
    run_predict()
