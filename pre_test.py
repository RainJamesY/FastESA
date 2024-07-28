import fitlog

use_fitlog = False
if not use_fitlog:
    fitlog.debug()
fitlog.set_log_dir('logs')
load_dataset_seed = 100
fitlog.add_hyper(load_dataset_seed, 'load_dataset_seed')
fitlog.set_rng_seed(load_dataset_seed)
import sys

sys.path.append('../')
from load_data import *
import argparse
from paths import *
from fastNLP.core import Trainer
# from trainer import Trainer
from fastNLP.core import Callback
from V1.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric, AccuracyMetric
from fastNLP.core.callback import WarmupCallback, GradientClipCallback, EarlyStopCallback
from fastNLP import FitlogCallback
# from fitlogcallback import FitlogCallback
# from my_fitlog_callback import FitlogCallback
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
# from models import LSTM_SeqLabel,LSTM_SeqLabel_True
import fitlog
from fastNLP import logger
from utils import get_peking_time
from V1.add_lattice import equip_chinese_ner_with_lexicon
# from load_data import load_toy_ner

import traceback
import warnings
import sys
from utils import print_info
from fastNLP.embeddings import BertEmbedding
# from fastNLP_module import BertEmbedding
from V1.models import BERT_SeqLabel

import torch
from fastNLP.core.predictor import Predictor

parser = argparse.ArgumentParser()
# performance inrelevant
parser.add_argument('--update_every', type=int, default=1)
parser.add_argument('--status', choices=['train', 'test'], default='test')
parser.add_argument('--use_bert', type=int, default=0)
parser.add_argument('--only_bert', type=int, default=0)
parser.add_argument('--fix_bert_epoch', type=int, default=20)
parser.add_argument('--after_bert', default='mlp', choices=['lstm', 'mlp'])
parser.add_argument('--msg', default='11266')
parser.add_argument('--train_clip', default=False, help='是不是要把train的char长度限制在200以内')
parser.add_argument('--device', default='0')
parser.add_argument('--debug', default=0, type=int)
parser.add_argument('--gpumm', default=False, help='查看显存')
parser.add_argument('--see_convergence', default=False)
parser.add_argument('--see_param', default=False)
parser.add_argument('--test_batch', default=-1)
parser.add_argument('--seed', default=1080956, type=int)
parser.add_argument('--test_train', default=False)
parser.add_argument('--number_normalized', type=int, default=0,
                    choices=[0, 1, 2, 3], help='0不norm，1只norm char,2norm char和bigram，3norm char，bigram和lattice')
parser.add_argument('--lexicon_name', default='yj', choices=['lk', 'yj'])
parser.add_argument('--use_pytorch_dropout', type=int, default=0)

parser.add_argument('--char_min_freq', default=1, type=int)
parser.add_argument('--bigram_min_freq', default=1, type=int)
parser.add_argument('--lattice_min_freq', default=1, type=int)
parser.add_argument('--only_train_min_freq', default=True)
parser.add_argument('--only_lexicon_in_train', default=False)

parser.add_argument('--word_min_freq', default=1, type=int)

# hyper of training
parser.add_argument('--early_stop', default=25, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=10, type=int)
parser.add_argument('--optim', default='sgd', help='sgd|adam')
parser.add_argument('--lr', default=6e-4, type=float)
parser.add_argument('--bert_lr_rate', default=0.05, type=float)
parser.add_argument('--embed_lr_rate', default=1, type=float)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--init', default='uniform', help='norm|uniform')
parser.add_argument('--self_supervised', default=False)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--norm_embed', default=True)
parser.add_argument('--norm_lattice_embed', default=True)

parser.add_argument('--warmup', default=0.1, type=float)

# hyper of model
# parser.add_argument('--use_bert',type=int,default=1)
parser.add_argument('--model', default='transformer', help='lstm|transformer')
parser.add_argument('--lattice', default=1, type=int)
parser.add_argument('--use_bigram', default=1, type=int)
parser.add_argument('--hidden', default=-1, type=int)
parser.add_argument('--ff', default=3, type=int)
parser.add_argument('--layer', default=1, type=int)
parser.add_argument('--head', default=8, type=int)
parser.add_argument('--head_dim', default=20, type=int)
parser.add_argument('--scaled', default=False)
parser.add_argument('--ff_activate', default='relu', help='leaky|relu')

parser.add_argument('--k_proj', default=False)
parser.add_argument('--q_proj', default=True)
parser.add_argument('--v_proj', default=True)
parser.add_argument('--r_proj', default=True)

parser.add_argument('--attn_ff', default=False)

# parser.add_argument('--rel_pos', default=False)
parser.add_argument('--use_abs_pos', default=False)
parser.add_argument('--use_rel_pos', default=True)
# 相对位置和绝对位置不是对立的，可以同时使用
parser.add_argument('--rel_pos_shared', default=True)
parser.add_argument('--add_pos', default=False)
parser.add_argument('--learn_pos', default=False)
parser.add_argument('--pos_norm', default=False)
parser.add_argument('--rel_pos_init', default=1)
parser.add_argument('--four_pos_shared', default=True, help='只针对相对位置编码，指4个位置编码是不是共享权重')
parser.add_argument('--four_pos_fusion', default='ff_two', choices=['ff', 'attn', 'gate', 'ff_two', 'ff_linear'],
                    help='ff就是输入带非线性隐层的全连接，'
                         'attn就是先计算出对每个位置编码的加权，然后求加权和'
                         'gate和attn类似，只不过就是计算的加权多了一个维度')

parser.add_argument('--four_pos_fusion_shared', default=True, help='是不是要共享4个位置融合之后形成的pos')

# parser.add_argument('--rel_pos_scale',default=2,help='在lattice且用相对位置编码时，由于中间过程消耗显存过大，'
#                                                  '所以可以使4个位置的初始embedding size缩小，'
#                                                  '最后融合时回到正常的hidden size即可')

parser.add_argument('--pre', default='')
parser.add_argument('--post', default='an')

over_all_dropout = -1
parser.add_argument('--embed_dropout_before_pos', default=False)
parser.add_argument('--embed_dropout', default=0.5, type=float)
parser.add_argument('--gaz_dropout', default=0.5, type=float)
parser.add_argument('--output_dropout', default=0.3, type=float)
parser.add_argument('--pre_dropout', default=0.5, type=float)
parser.add_argument('--post_dropout', default=0.3, type=float)
parser.add_argument('--ff_dropout', default=0.15, type=float)
parser.add_argument('--ff_dropout_2', default=-1, type=float, help='FF第二层过完后的dropout，之前没管这个的时候是0')
parser.add_argument('--attn_dropout', default=0, type=float)
parser.add_argument('--embed_dropout_pos', default='0')
parser.add_argument('--abs_pos_fusion_func', default='nonlinear_add',
                    choices=['add', 'concat', 'nonlinear_concat', 'nonlinear_add', 'concat_nonlinear', 'add_nonlinear'])

parser.add_argument('--dataset', default='clue', help='ontonotes|weibo|resume|ontonotes|msra')
# parser.add_argument('--debug',default=1)


args = parser.parse_args()
if args.ff_dropout_2 < 0:
    args.ff_dropout_2 = args.ff_dropout

if over_all_dropout > 0:
    args.embed_dropout = over_all_dropout
    args.output_dropout = over_all_dropout
    args.pre_dropout = over_all_dropout
    args.post_dropout = over_all_dropout
    args.ff_dropout = over_all_dropout
    args.attn_dropout = over_all_dropout

if args.lattice and args.use_rel_pos:
    args.train_clip = False

# fitlog.commit(__file__,fit_msg='绝对位置用新的了')
fitlog.set_log_dir('logs')
now_time = get_peking_time()
logger.add_file('log/{}'.format(now_time), level='info')
if args.test_batch == -1:
    args.test_batch = args.batch // 2
fitlog.add_hyper(now_time, 'time')
if args.debug:
    # args.dataset = 'toy'
    pass

if args.device != 'cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

refresh_data = True

raw_dataset_cache_name = os.path.join('cache', args.dataset +
                                      '_trainClip:{}'.format(args.train_clip)
                                      + 'bgminfreq_{}'.format(args.bigram_min_freq)
                                      + 'char_min_freq_{}'.format(args.char_min_freq)
                                      + 'word_min_freq_{}'.format(args.word_min_freq)
                                      + 'only_train_min_freq{}'.format(args.only_train_min_freq)
                                      + 'number_norm{}'.format(args.number_normalized)
                                      + 'load_dataset_seed{}'.format(load_dataset_seed))

datasets, vocabs, embeddings = load_clue2020(clue_2020_ner_path, yangjie_rich_pretrain_unigram_path,
                                             yangjie_rich_pretrain_bigram_path,
                                             _refresh=refresh_data,
                                             index_token=False,
                                             train_clip=args.train_clip,
                                             _cache_fp=raw_dataset_cache_name,
                                             char_min_freq=args.char_min_freq,
                                             bigram_min_freq=args.bigram_min_freq,
                                             only_train_min_freq=args.only_train_min_freq
                                             )

if args.gaz_dropout < 0:
    args.gaz_dropout = args.embed_dropout

args.hidden = args.head_dim * args.head
args.ff = args.hidden * args.ff

w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                              _refresh=refresh_data,
                                              _cache_fp='cache/{}'.format(args.lexicon_name)
                                              )

cache_name = os.path.join('cache', (args.dataset + '_lattice' + '_only_train#{}' +
                                    '_trainClip#{}' + '_norm_num#{}'
                                    + 'char_min_freq{}' + 'bigram_min_freq{}' + 'word_min_freq{}' + 'only_train_min_freq{}'
                                    + 'number_norm{}' + 'lexicon_{}' + 'load_dataset_seed_{}')
                          .format(args.only_lexicon_in_train,
                                  args.train_clip, args.number_normalized, args.char_min_freq,
                                  args.bigram_min_freq, args.word_min_freq, args.only_train_min_freq,
                                  args.number_normalized, args.lexicon_name, load_dataset_seed))

datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(datasets, vocabs, embeddings,
                                                              w_list, yangjie_rich_pretrain_word_path,
                                                              _refresh=refresh_data,
                                                              _cache_fp=cache_name,
                                                              only_lexicon_in_train=args.only_lexicon_in_train,
                                                              word_char_mix_embedding_path=yangjie_rich_pretrain_char_and_word_path,
                                                              number_normalized=args.number_normalized,
                                                              lattice_min_freq=args.lattice_min_freq,
                                                              only_train_min_freq=args.only_train_min_freq)

avg_seq_len = 0
avg_lex_num = 0
avg_seq_lex = 0
train_seq_lex = []
dev_seq_lex = []
test_seq_lex = []
train_seq = []
dev_seq = []
test_seq = []
for k, v in datasets.items():
    max_seq_len = 0
    max_lex_num = 0
    max_seq_lex = 0
    max_seq_len_i = -1
    for i in range(len(v)):
        if max_seq_len < v[i]['seq_len']:
            max_seq_len = v[i]['seq_len']
            max_seq_len_i = i
        # max_seq_len = max(max_seq_len,v[i]['seq_len'])
        max_lex_num = max(max_lex_num, v[i]['lex_num'])
        max_seq_lex = max(max_seq_lex, v[i]['lex_num'] + v[i]['seq_len'])

        avg_seq_len += v[i]['seq_len']
        avg_lex_num += v[i]['lex_num']
        avg_seq_lex += (v[i]['seq_len'] + v[i]['lex_num'])
        if k == 'train':
            train_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
            train_seq.append(v[i]['seq_len'])
            if v[i]['seq_len'] > 200:
                print('train里这个句子char长度已经超了200了')
                print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
            else:
                if v[i]['seq_len'] + v[i]['lex_num'] > 400:
                    print('train里这个句子char长度没超200，但是总长度超了400')
                    print(''.join(list(map(lambda x: vocabs['char'].to_word(x), v[i]['chars']))))
        if k == 'dev':
            dev_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
            dev_seq.append(v[i]['seq_len'])
        if k == 'test':
            test_seq_lex.append(v[i]['lex_num'] + v[i]['seq_len'])
            test_seq.append(v[i]['seq_len'])

max_seq_len = max(*map(lambda x: max(x['seq_len']), datasets.values()))
# show_index = 4
# exit(1208)

for k, v in datasets.items():
    if args.lattice:
        v.set_input('lattice', 'bigrams', 'seq_len', 'target')
        v.set_input('lex_num', 'pos_s', 'pos_e')
        v.set_target('target', 'seq_len')
        v.set_pad_val('lattice', vocabs['lattice'].padding_idx)
    else:
        v.set_input('chars', 'bigrams', 'seq_len', 'target')
        v.set_target('target', 'seq_len')

from utils import norm_static_embedding

# print(embeddings['char'].embedding.weight[:10])
if args.norm_embed > 0:
    print('embedding:{}'.format(embeddings['char'].embedding.weight.size()))
    print('norm embedding')
    for k, v in embeddings.items():
        norm_static_embedding(v, args.norm_embed)

if args.norm_lattice_embed > 0:
    print('embedding:{}'.format(embeddings['lattice'].embedding.weight.size()))
    print('norm lattice embedding')
    for k, v in embeddings.items():
        norm_static_embedding(v, args.norm_embed)

mode = {}
mode['debug'] = args.debug
mode['gpumm'] = args.gpumm
if args.debug or args.gpumm:
    fitlog.debug()
dropout = collections.defaultdict(int)
dropout['embed'] = args.embed_dropout
dropout['gaz'] = args.gaz_dropout
dropout['output'] = args.output_dropout
dropout['pre'] = args.pre_dropout
dropout['post'] = args.post_dropout
dropout['ff'] = args.ff_dropout
dropout['ff_2'] = args.ff_dropout_2
dropout['attn'] = args.attn_dropout

torch.backends.cudnn.benchmark = False
fitlog.set_rng_seed(args.seed)
torch.backends.cudnn.benchmark = False

fitlog.add_hyper(args)

model = Lattice_Transformer_SeqLabel(embeddings['lattice'], embeddings['bigram'], args.hidden,
                                     len(vocabs['label']),
                                     args.head, args.layer, args.use_abs_pos, args.use_rel_pos,
                                     args.learn_pos, args.add_pos,
                                     args.pre, args.post, args.ff, args.scaled, dropout, args.use_bigram,
                                     mode, device, vocabs,
                                     max_seq_len=max_seq_len,
                                     rel_pos_shared=args.rel_pos_shared,
                                     k_proj=args.k_proj,
                                     q_proj=args.q_proj,
                                     v_proj=args.v_proj,
                                     r_proj=args.r_proj,
                                     self_supervised=args.self_supervised,
                                     attn_ff=args.attn_ff,
                                     pos_norm=args.pos_norm,
                                     ff_activate=args.ff_activate,
                                     abs_pos_fusion_func=args.abs_pos_fusion_func,
                                     embed_dropout_pos=args.embed_dropout_pos,
                                     four_pos_shared=args.four_pos_shared,
                                     four_pos_fusion=args.four_pos_fusion,
                                     four_pos_fusion_shared=args.four_pos_fusion_shared,
                                     bert_embedding=None,
                                     )

if 1:
    model_path = './clue/best_Lattice_Transformer_SeqLabel_f_2023-02-05-14-00-53-135014'
    states = torch.load(model_path).state_dict()
    states.pop('bigram_embed.words_to_words')
    states.pop('bigram_embed.embedding.weight')
    states.pop('lattice_embed.words_to_words')
    states.pop('lattice_embed.embedding.weight')
    model.load_state_dict(states, strict=False)

    predictor = Predictor(model)  # 这里的model是加载权重之后的model

    # for d in vocabs['label']:
    #     print(d)

    test_label_list = predictor.predict(datasets['test'][:])['pred']  # 预测结果
    test_raw_char = datasets['test'][:]['raw_chars']  # 原始文字
    #

    for i in test_label_list:
        print(i)
    for i in test_raw_char:
        print(i)

    entity_list = []
    for i, j in zip(test_label_list, test_raw_char):
        entity = recognize(i, j)
        if len(entity) != 0:
            entity_list.append(entity)
    for i in entity_list:
        print(i)
