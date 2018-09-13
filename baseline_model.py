import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

class BaselineModel(nn.Module):
    def __init__(self, word_emb, opt):
        super(BaselineModel, self).__init__()
        self.opt = opt
        self.word_emb = nn.Embedding(517015, 300)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(word_emb))
        self.word_emb_dim = 300
        self.sent_emb_dim = opt['sent_emb_dim']
        self.dropout = opt['dropout'] # actually this doesn't work
        if opt['lstm']:
            self.word2sent_lstm = nn.LSTM(self.word_emb_dim, self.sent_emb_dim//2, num_layers=1, bidirectional=True, batch_first=True)
            self.hidden = self.init_hidden()

        self.dot_layer = nn.Linear(self.sent_emb_dim, self.sent_emb_dim) # add weight in score.dot(score)

    def init_hidden(self):
        return (torch.randn(2, 1, self.sent_emb_dim // 2),
                torch.randn(2, 1, self.sent_emb_dim // 2))

    def _idx2emb(self, batch, word_emb):
        hypo_res = []
        para_res = []
        for instance in batch:
            hypo_instance = []
            para_instance = []
            for sent in instance["paragraph"]:
                sent_res = self.word_emb(torch.LongTensor(sent))
                para_instance.append(sent_res)
            for sent in instance["hypothesis"]:
                sent_res = self.word_emb(torch.LongTensor(sent))
                hypo_instance.append(sent_res)
            para_res.append(para_instance)
            hypo_res.append(hypo_instance)
        return hypo_res, para_res

    def _get_sent_emb(self, batch, lstm=False):
        res = []
        for instance in batch:
            tmp = []
            for sent in instance:
                if lstm:    tmp.append(self._sent_emb_lstm(sent)) # use BiLSTM
                else:   tmp.append(torch.sum(sent, dim=0)/len(sent)) # Just simply sum all word vectors. Have to divided by length or else longer vector will have higher score.
            tmp = torch.stack(tmp)
            res.append(tmp)
        return res

    def _sent_emb_lstm(self, sent):
        '''
        sent: [[0.23323,...], [0.1682,...], [0.9481,...]] => ['I', 'am', 'happy']
        output: tensor([0.4398, 0.4348, 0.3911]) (sentence embedding)
        '''
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.word2sent_lstm(sent.view(1, len(sent), self.word_emb_dim), self.hidden)
        return lstm_out[0][-1]


    def _fuse(self, hypo, para, max=True, return_score_matrix=False, abs=True):
        '''
        output:
            [ tensor1d ]

        '''
        instance_num = len(para)
        res = []
        for i in range(instance_num):
            score_tensor = torch.matmul(hypo[i], torch.t(para[i])) # m * n
            if abs:
                score_tensor = torch.abs(score_tensor)
            if not return_score_matrix:
                if max:
                    score_tensor = torch.max(score_tensor, dim=0)[0] # max
                else:
                    score_tensor = torch.mean(score_tensor, dim=0) # mean
            res.append(score_tensor)
        return res

    def get_score(self, batch):
        '''
        Almost same with forward, but stop at _fuse().
        '''
        hypo_word_emb, para_word_emb = self._idx2emb(batch, self.word_emb)
        hypo_sent_emb = self._get_sent_emb(hypo_word_emb, lstm=self.opt['lstm'])
        para_sent_emb = self._get_sent_emb(para_word_emb, lstm=self.opt['lstm'])
        score_matrix = self._fuse(hypo_sent_emb, para_sent_emb, return_score_matrix=True)

        return score_matrix

    def score2prob(self, scores):
        '''
        output:
            [ tensor1d ]
        '''
        prob = []
        activate_func = nn.Sigmoid()
        for instance in scores:
            #prob.append(activate_func(instance))
            prob.append(torch.tanh(instance))
        return prob

    def loss_func(self, batch):
        '''
        'backward' this to backprop.
        Use NLLL.
        '''
        loss = torch.FloatTensor([0])
        probs = self.forward(batch)
        for i, instance in enumerate(probs):
            target = batch[i]["target"]
            for j, prob in enumerate(instance):
                if target[j] == 1:  loss -= torch.log(instance[j:j+1])
                else:   loss -= torch.log(1-instance[j:j+1])
        return loss

    def predict(self, batch):
        '''
        input batch and output the prediction(0/1 vector)
        '''
        res_list = []
        probs = self.forward(batch)
        for instance in probs:
            instance_list = [1  if i >= 0.5 else 0 for i in instance]
            res_list.append(instance_list)
        return res_list

    def evaluate(self, batch):
        '''
        input batch and output the em and f1 score
        '''
        predict_labels = self.predict(batch)
        ground_truths = [instance["target"] for instance in batch]
        em = 0
        f1 = 0
        instance_num = len(predict_labels)
        for i in range(instance_num):
            candi_num = len(predict_labels[i])
            same = [1 if (predict_labels[i][j] == ground_truths[i][j] == 1) else 0 for j in range(candi_num)]
            same = sum(same)
            # Compute precision and recall
            try:    precision = same/sum(predict_labels[i])
            except: precision = 0
            recall = same/sum(ground_truths[i])

            # Compute EM and F1
            if predict_labels[i] == ground_truths[i]:   em += 1
            # precision+recall might be zero
            try:    f1 += 2 * precision * recall / (precision+recall)
            except: pass
        em /= instance_num
        f1 /= instance_num

        return {"em":em, "f1":f1}


    def forward(self, batch):
        '''
        batch:
            [{"hypothesis":[[1,2,3], [4,4], [5,6,7]],
              "paragraph": [[2,8], [8,7], [1], [6,6,3,9]],
              "target": [1, 0, 0],
              "id": "testing-1"}
            ]

        hypo_word_emb:
            batch[   instance[    sentence:tensor2d   ]]
            [[tensor2d, tensor2d], [t2d, t2d]]

        hypo_sent_emb:
            batch[   sents:tensor2d      ]
            [   tensor2d, tensor2d   ]

        '''
        hypo_word_emb, para_word_emb = self._idx2emb(batch, self.word_emb)
        hypo_sent_emb = self._get_sent_emb(hypo_word_emb, lstm=self.opt['lstm'])
        para_sent_emb = self._get_sent_emb(para_word_emb, lstm=self.opt['lstm'])
        para_score = self._fuse(hypo_sent_emb, para_sent_emb)
        para_prob = self.score2prob(para_score)

        return para_prob


    def load_checkpoint(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state['state'])

    def save_checkpoint(self, filename, epoch):
        model_state = self.state_dict()
        params = {
            'state': model_state,
        }
        path = filename.split('/')[:-1]
        if not os.path.exists(''.join(path)):
            os.makedirs(''.join(path))
        try:
            torch.save(params, filename)
        except:
            print("Fail to save...", file=sys.stderr)
