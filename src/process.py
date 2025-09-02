import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizer import EnglishTokenizer,ChineseTokenizer
import config


def process():
    print("开始处理数据")
    # 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt',sep='\t',header=None,usecols=[0,1],names=['en','zh'])
    df.dropna(inplace=True)
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]
    train_df,test_df = train_test_split(df,test_size=0.2)
    # 构建词表
    ChineseTokenizer.built_vocab(train_df['zh'].tolist(),config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    EnglishTokenizer.built_vocab(train_df['en'].tolist(),config.PROCESS_DATA_DIR / 'en_vocab_txt')
    # 构建tokenizer对象
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'zh_vocab_txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'en_vocab_txt')

    # 计算最大长度
    # zh_len = train_df['zh'].apply(lambda x:len(zh_tokenizer.tokenize(x))).quantile(0.95)
    # en_len = train_df['en'].apply(lambda x:len(en_tokenizer.tokenize(x))).quantile(0.95)
    # print(f"{zh_len = },{en_len = }")

    # 构建训练集并进行保存
    train_df['zh'] = train_df['zh'].apply(lambda x:zh_tokenizer.encode(x,config.SEQ_LEN,add_sos_eos=False))
    train_df['en'] = train_df['en'].apply(lambda x:en_tokenizer.encode(x,config.SEQ_LEN,add_sos_eos=True))
    train_df.to_json(config.PROCESS_DATA_DIR / 'indexed_train.jsonl',lines=True,orient='records')
    # 构建测试集并进行保存
    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=False))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, config.SEQ_LEN, add_sos_eos=True))
    test_df.to_json(config.PROCESS_DATA_DIR / 'indexed_test.jsonl', lines=True, orient='records')
    print("数据处理完成")


if __name__ == '__main__':
    process()