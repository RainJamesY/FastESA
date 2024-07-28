from fastNLP.io import CSVLoader
from fastNLP import Vocabulary
from fastNLP import Const
import numpy as np
import fitlog
import pickle
import os
from fastNLP import cache_results
from fastNLP_module import StaticEmbedding


# 加入了加载CLUE2020的方法

@cache_results(_cache_fp='cache/clue2020', _refresh=True)
def load_clue2020(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True, train_clip=False,
                  char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path, 'train.char.bmes{}'.format('_clip' if train_clip else ''))
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')
    # predict_path = os.path.join(path, 'predict.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)
    # predict_bundle = loader.load(predict_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    # datasets['predict'] = predict_bundle.datasets['predict']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    # datasets['predict'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    # datasets['predict'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    # print(len(datasets['predict']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {'char': char_vocab, 'label': label_vocab, 'bigram': bigram_vocab}

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/clue2020_pre', _refresh=True)
def load_clue2020_predict(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                          train_clip=False,
                          char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0):
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams

    train_path = os.path.join(path, 'train.char.bmes{}'.format('_clip' if train_clip else ''))
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')
    predict_path = os.path.join(path, 'predict.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)
    predict_bundle = loader.load(predict_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']
    datasets['predict'] = predict_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['predict'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')
    datasets['predict'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary()

    print(datasets.keys())
    print(len(datasets['dev']))
    print(len(datasets['test']))
    print(len(datasets['train']))
    print(len(datasets['predict']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test'], datasets['predict']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test'], datasets['predict']])
    label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'], datasets['predict'],
                                 field_name='chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'], datasets['predict'],
                                   field_name='bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'], datasets['predict'],
                                  field_name='target', new_field_name='target')

    vocabs = {'char': char_vocab, 'label': label_vocab, 'bigram': bigram_vocab}
    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq)
        embeddings['bigram'] = bigram_embedding

    return datasets, vocabs, embeddings


@cache_results(_cache_fp='cache/load_yangjie_rich_pretrain_word_list', _refresh=False)
def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list


def deal_input_data(path):
    from urllib import parse
    a = '. O\n. O\n. O\n\n'
    b = [',', '.', '。', '“', '”', '，', '‘', '’', '?', '？', ':', '：']
    c = '\n'
    with open('./CLUE2020/test.char.bmes', 'w', encoding='utf-8') as f:

        f.close()
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for sequence in lines:
            s = sequence.encode()
            ss = s.decode('utf-8').replace('\\x', '%')
            for i in ss:
                with open('./CLUE2020/test.char.bmes', 'a+', encoding='utf-8') as f:
                    un = parse.unquote(i)
                    f.write(un + ' O' + '\n')
                f.close()
    f.close()

def recognize(label_list, raw_chars):
    """
    根据模型预测的label_list，找出其中的实体
    label_lsit: array
    raw_chars: list of raw_char
    return: entity_list: list of tuple(ent_text, ent_type)
    -------------
    ver: 20210303
    by: changhongyu
    """
    if len(label_list.shape) == 2:
        label_list = label_list[0]
    elif len(label_list) > 2:
        raise ValueError('please check the shape of input')

    assert len(label_list.shape) == 1
    assert len(label_list) == len(raw_chars)

    # 其实没有必要写这个
    # 但是为了将来可能适应bio的标注模式还是把它放在这里了
    starting_org = False  ###
    starting_name = False  ###
    starting_type = False  ###
    ent_type = None
    ent_stance = None
    ent_text = ''
    entity_list = []

    for i, label in enumerate(label_list):
        if label in [0, 1, 2]:
            ent_text = ''
            ent_type = None
            continue
        # begin
        elif label == 5 or label == 12 or label == 18:
            ent_type = 'ORG'
            starting_org = True
            ent_text += raw_chars[i]

            if label == 5:
                ent_stance = 'P'
            if label == 12:
                ent_stance = 'M'
            if label == 18:
                ent_stance = 'N'

        elif label == 10 or label == 20 or label == 28:
            ent_type = 'NAME'
            starting_name = True
            ent_text += raw_chars[i]

            if label == 10:
                ent_stance = 'P'
            if label == 20:
                ent_stance = 'M'
            if label == 28:
                ent_stance = 'N'

        elif label == 16 or label == 23 or label == 24:
            ent_type = 'TYPE'
            starting_type = True
            ent_text += raw_chars[i]

            if label == 16:
                ent_stance = 'M'
            if label == 23:
                ent_stance = 'P'
            if label == 24:
                ent_stance = 'N'

        # middle
        elif label == 9 or label == 14 or label == 27:
            if starting_org:
                ent_text += raw_chars[i]
        elif label == 4 or label == 8 or label == 26:
            if starting_name:
                ent_text += raw_chars[i]
        elif label == 3 or label == 7 or label == 21:
            if starting_type:
                ent_text += raw_chars[i]

        # end
        elif label == 6 or label == 13 or label == 17:
            if starting_org:
                ent_text += raw_chars[i]
                starting_org = False
        elif label == 11 or label == 19 or label == 29:
            if starting_name:
                ent_text += raw_chars[i]
                starting_name = False
        elif label == 15 or label == 22 or label == 25:
            if starting_type:
                ent_text += raw_chars[i]
                starting_type = False

        else:
            ent_text = ''
            ent_type = None
            ent_stance = None
            continue

        if not (starting_org or starting_name or starting_type) and len(ent_text):
            # 判断实体已经结束，并且提取到的实体有内容
            entity_list.append((ent_text, ent_type, ent_stance))

    return entity_list

if __name__ == '__main__':
    # pass
    datasets, vocabs, embeddings = load_clue2020_predict('./CLUE2020')
    # for i in datasets:
    #     print(i)
