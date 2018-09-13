import numpy as np
import ujson as json
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import gradcheck
import sys
import os
import argparse
import logging
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as mfm
from tqdm import tqdm

from baseline_model import BaselineModel
from util import load_data, save_checkpoint, load_checkpoint, print_model_param, sents2idx, token2sents
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Thin.ttc"
prop = mfm.FontProperties(fname=font_path)
# mpl.rcParams['font.family']=font_name
# mpl.rcParams['axes.unicode_minus']=False

parser = argparse.ArgumentParser(
    description='SER Baseline Model'
)
parser.add_argument('--name', default='', help='additional name of the current run')
parser.add_argument('--log_file', default='output.log', help='path for log file.')

# data
parser.add_argument('-dp', '--data_path', default='data/')
parser.add_argument('--train_data', default='train.json')
parser.add_argument('--dev_data', default='dev.json')
parser.add_argument('--test_data', default='test.json')
parser.add_argument('--word_emb_file', default='word_emb_file')
parser.add_argument('--word_dict', default='word_dictionary')

# training
parser.add_argument('-lr', type=float, default=0.01)
parser.add_argument('-e', '--epoches', type=int, default=22)
parser.add_argument('-bs', '--batch_size', type=int, default=8)
parser.add_argument('--check_num', type=int, default=3)
parser.add_argument('-t', '--task', type=int, default=1) # train 1, evaluate 2, multi_evaluate 4
TRAIN=1
EVALUATE=2
MULTI_EVALUTE=4
DRAW_ATTENTION = 8

parser.add_argument('--lstm', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--sent_emb_dim', type=int, default=300)


# model
parser.add_argument('-md', '--model_dir', default='model_dir', help='path to store saved models')
parser.add_argument('--save_all', dest="save_best_only", action='store_false',
                    help='save all models in addition to the best.')
parser.add_argument('--do_not_save', action='store_true', help='don\'t save any model')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
parser.add_argument('--model_path', default='model_dir/epoch', help='model path that you want to evaluate')
parser.add_argument('--draw', type=bool, default=True, help='draw figure if task == 4 (multi_evaluate)')
parser.add_argument('--whole_output_directory', default='whole_output')
parser.add_argument('--whole_output_file', default='whole_output')

parser.add_argument('--big', type=bool, default=True)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--max', type=bool, default=True) # if false, use mean
parser.add_argument('--abs', type=bool, default=True) # use abs+tanh, if false, then use Sigmoid





args = parser.parse_args()
if args.name != '':
    args.model_dir = args.model_dir + '_' + args.name
    args.log_file = os.path.dirname(args.log_file) + 'output_' + args.name + '.log'
    args.model_path = args.model_dir+'/epoch'

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch.setFormatter(formatter)
log.addHandler(ch)

def main():
    log.info('[program starts.]')
    opt = vars(args)

    with open(opt['data_path']+opt['word_emb_file'], "r") as f:
        word_emb = np.array(json.load(f), dtype=np.float32)
    with open(opt['data_path']+opt['word_dict'], "r") as f:
        word_dictionary = json.load(f)

    train_data = load_data(opt['data_path']+opt['train_data'], dict=word_dictionary) # 131
    dev_data = load_data(opt['data_path']+opt['dev_data'], dict=word_dictionary)     # 135
    test_data = load_data(opt['data_path']+opt['test_data'], dict=word_dictionary)   # 120
    raw_train_data = load_data(opt['data_path']+opt['train_data'])
    raw_dev_data = load_data(opt['data_path']+opt['dev_data'])
    raw_test_data = load_data(opt['data_path']+opt['test_data'])

    model = BaselineModel(word_emb, opt)

    # small test
    if not opt['big']:
        train_data = train_data[0:16]
        dev_data = dev_data[0:16]
        test_data = test_data[0:16]
        raw_train_data = raw_train_data[0:16]
        raw_dev_data = raw_dev_data[0:16]
        raw_test_data = raw_test_data[0:16]

    # raw_train_data = [{'target': [1, 1], 'paragraph': [['我', '討厭', '吃', '香蕉', '。'], ['我', '喜歡', '吃', '鳳梨','。']], 'hypothesis': [['我', '喜歡', '吃', '香蕉'], ['我','討厭', '吃', '鳳梨']], 'id': 'test-000001'}]

    train_data.extend(dev_data)

    print(len(train_data))
    if opt['task'] & TRAIN:
        train(opt, train_data, model)
    if opt['task'] & EVALUATE:
        evaluate(opt, dev_data, model)
    if opt['task'] & MULTI_EVALUTE:
        train_f1, train_em = multi_evaluate(opt, train_data, model, draw=opt['draw'], data_type='train')
        dev_f1, dev_em = multi_evaluate(opt, test_data, model, draw=opt['draw'], data_type='test')
        if opt['draw']:
            plt.legend()
            plt.xlabel('epoches')
            plt.ylabel('score')
            plt.savefig(opt['model_dir']+'/'+opt['name']+'figure.png')
    if opt['task'] & DRAW_ATTENTION:
        draw_attention(opt, test_data, model, raw_test_data)


    #whole_output(opt, train_data, raw_train_data, model, 'tmp')
    # print("\n\n===== testing set======")
    #whole_output(opt, test_data, raw_test_data, model, 'tmp')



def train(opt, train_data, model):
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    batch_size = opt['batch_size']
    model_dir = opt['model_dir']
    check_num = opt['check_num']

    if not opt['save']:
        check_num = 1000

    batch_num = [i for i in range(0, len(train_data), batch_size)]
    for epoch in range(opt['epoches']):
        total_loss = 0
        print("epoch: ", epoch, file=sys.stderr)
        random.shuffle(train_data)

        for i in tqdm(batch_num):
            batch = train_data[i:i+batch_size]
            model.zero_grad()
            loss = model.loss_func(batch)
            total_loss += loss
            loss.backward()
            optimizer.step()

        total_loss /= len(train_data)
        print("total_loss: ", total_loss, file=sys.stderr)

        if epoch % check_num == 0:
            model.save_checkpoint(model_dir + '/epoch'+str(epoch), epoch)


def evaluate(opt, data, model, model_path=None):
    if model_path == None:
        model_path = opt['model_path']+opt['epoches']
    model.load_checkpoint(model_path)
    result = model.evaluate(data)
    return result

def multi_evaluate(opt, data, model, draw=False, data_type='dev'):
    epoches = opt['epoches']
    check_num = opt['check_num']
    model_path = opt['model_path']
    f1_list = []
    em_list = []

    for i in range(0, epoches, check_num):
        cur_model_path = model_path+str(i)
        result = evaluate(opt, data, model, model_path=cur_model_path)
        f1_list.append(result['f1'])
        em_list.append(result['em'])

    if draw:
        x_axis = [i for i in range(0, epoches, check_num)]
        plt.title(opt['name']+" f1/em score")
        plt.plot(x_axis, f1_list, label=data_type+'f1')
        plt.plot(x_axis, em_list, label=data_type+'em')

    return f1_list, em_list

def whole_output(opt, data, raw_data, model, output_name):
    model_path = opt['model_path']
    log.info('Prediction')
    #model.load_checkpoint(model_path)
    prediction = model.predict(data)
    for p,d in zip(prediction, raw_data):
        log.critical(d["id"])
        log.critical("是非題：\t"+token2sents(d["hypothesis"]))
        log.critical("課文段落：\t"+token2sents(d["paragraph"]))
        log.critical("正確證據：\t"+token2sents(d["paragraph"], d['target']))
        log.critical("分析證據：\t"+token2sents(d["paragraph"], p))
        log.critical("---------------------------")
    return

def draw_attention(opt, data, model, raw_data):
    # data stores index, raw_data stores string
    model_path = opt['model_dir']+'/epoch'+str(opt['epoches'])
    model.load_checkpoint(model_path)
    scores = model.get_score(data)
    for i, score in enumerate(scores):
        hypo = raw_data[i]['hypothesis']
        para = raw_data[i]['paragraph']
        hypo = [''.join(h) for t, h in enumerate(hypo)]
        para = [''.join(p) for t, p in enumerate(para)]
        fig, ax = plt.subplots()
        score_data = score.tolist()
        im = ax.imshow(score_data)
        ax.set_xticks(np.arange(len(para)))
        ax.set_yticks(np.arange(len(hypo)))
        ax.set_xticklabels(para, fontproperties=prop)
        ax.set_yticklabels(hypo, fontproperties=prop)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

        for k in range(len(hypo)):
            for j in range(len(para)):
                text = ax.text(j, k, round(score_data[k][j],2), ha="center", va="center", color="w")

        id = data[i]['id']
        ax.set_title(id)
        fig.tight_layout()

        plt.savefig(opt['model_dir']+'/'+id+'.png')
        plt.clf()

# python3 main.py -t 2 --model_path model_dir/epoch5
# python3 main.py -t 1 --name NAME -e EPOCH
# python3 main.py -t 4  -e 15 --draw t --name NAME
# python3 main.py -t 8 --name NAME -e 39 --big t
# time python3 main.py -t 0 --big t --name abs_tanh_large

if __name__ == '__main__':
    main()
