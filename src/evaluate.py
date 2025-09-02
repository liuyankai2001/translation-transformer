import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from dataset import get_dataloader
import config
from model import TrainslationEncoder, TrainslationDecoder
from tokenizer import ChineseTokenizer, EnglishTokenizer
from predict import predict_batch


def evaluate(dataloader, encoder, decoder, zh_tokenizer, en_tokenizer, device):
    references = [] # [batch_size, 1, seq_len]
    predictions = [] # [batch_size, seq_len]
    special_tokens = [en_tokenizer.eos_token_id,en_tokenizer.sos_token_id,en_tokenizer.pad_token_id]
    for inputs,targets in tqdm(dataloader,desc="评估"):
        inputs = inputs.to(device) # [batch_size, seq_len]
        targets = targets.tolist() # [batch_size, seq_len]
        batch_result = predict_batch(inputs,encoder,decoder,zh_tokenizer,en_tokenizer,device)
        predictions.extend(batch_result)
        # references.extend([[target] for target in targets])
        references.extend([[[index for index in target if index not in special_tokens]] for target in targets])
    return corpus_bleu(references,predictions)


def run_evaluate():
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

    dataloader = get_dataloader(train=False)
    bleu = evaluate(dataloader,encoder,decoder,zh_tokenizer,en_tokenizer,device)
    print(f"bleu:{bleu}")


if __name__ == '__main__':
    run_evaluate()