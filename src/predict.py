import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TrainslationEncoder, TrainslationDecoder


def predict_batch(input_tensor, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    """
    批量预测
    :param input_tensor: [batch_size, seq_len]
    :param encoder:
    :param decoder:
    :param zh_tokenizer:
    :param en_tokenizer:
    :param device:
    :return:一批英文句子 e.g.:[[14,212],[2132,21,323,123,122],[213,221,54],...]
    """
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        encoder_outputs, context_vector = encoder(input_tensor)  # [batch_size, decoder_hidden_size]
        batch_size = context_vector.shape[0]
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_id, device=device)
        decoder_hidden = context_vector.unsqueeze(0)

        generated = [[] for _ in range(batch_size)]
        is_finished = [False for _ in range(batch_size)]
        for t in range(config.SEQ_LEN):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden,encoder_outputs)
            # decoder_output: [batch_size, 1, vocab_size]
            predict_indexes = torch.argmax(decoder_output, dim=-1)  # [batch_size, 1]
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

            decoder_input = predict_indexes
        return generated


def predict(user_input, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    # 处理数据
    index_list = zh_tokenizer.encode(user_input, config.SEQ_LEN)
    input_tensor = torch.tensor([index_list]).to(device)
    batch_result = predict_batch(input_tensor, encoder, decoder, zh_tokenizer, en_tokenizer, device)
    result = batch_result[0]
    return en_tokenizer.decode(result)
    # return result


def run_predict():
    # 准备资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab_txt')

    # 准备模型
    encoder = TrainslationEncoder(vocab_size=zh_tokenizer.vocab_size, padding_index=zh_tokenizer.pad_token_id).to(
        device)
    decoder = TrainslationDecoder(vocab_size=en_tokenizer.vocab_size, padding_index=en_tokenizer.pad_token_id).to(
        device)
    encoder.load_state_dict(torch.load(config.MODELS_DIR / 'encoder.pt'))
    decoder.load_state_dict(torch.load(config.MODELS_DIR / 'decoder.pt'))
    print("欢迎使用中英翻译！（输入q或者quit退出）")
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("程序已退出！")
            break
        if user_input.strip() == '':
            print('请输入中文')
            continue
        result = predict(user_input, encoder, decoder, zh_tokenizer, en_tokenizer, device)
        print(f"英文：{result}")


if __name__ == '__main__':
    run_predict()
