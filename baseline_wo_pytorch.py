import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BaselineModel(nn.Module):
    def __init__(self, word_emb):
        super(BaselineModel, self).__init__()
        self.word_emb = nn.Embedding(517015, 300)
        self.word_emb.weight = nn.Parameter(torch.from_numpy(word_emb))
        self.word_emb_dim = 300
        self.sent_emb_dim = 300
        self.word2sent_lstm = nn.LSTM(self.word_emb_dim, self.sent_emb_dim//2, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden = self.init_hidden()

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
                sent_res = []
                for word in sent:
                    # # TODO: deal with OOV
                    sent_res.append(word_emb[word])
                para_instance.append(sent_res)
            for sent in instance["hypothesis"]:
                sent_res = []
                for word in sent:
                    # # TODO: deal with OOV
                    sent_res.append(word_emb[word])
                hypo_instance.append(sent_res)

            para_res.append(para_instance)
            hypo_res.append(hypo_instance)
        return hypo_res, para_res

    def _get_sent_emb(self, batch, lstm=False):
        res = []
        for instance in batch:
            tmp = []
            for sent in instance:
                # Have to normalize embedding vector or else longer vector will have higher score
                # Just simply sum all word vectors
                if lstm:
                    tmp.append(self._sent_emb_lstm(sent)/len(sent))
                else:
                    tmp.append(np.sum(sent, axis=0)/len(sent))
            torch.stack(tmp)
            res.append(tmp)
        torch.stack(tmp)
        return res

    def _sent_emb_lstm(self, sent):
        '''
        sent: [0.23323, 0.1682 ,0.9481] => ['I', 'am', 'happy']
        output: tensor([0.4398, 0.4348, 0.3911]) (sentence embedding)
        '''
        self.hidden = self.init_hidden()
        #emb = self.word_emb(sent).view(len(sent),1,-1)
        lstm_out, self.hidden = self.word2sent_lstm(sent.view(1, len(sent), self.sent_emb_dim), self.hidden)
        return lstm_out[0][-1]


    def _fuse(self, hypo, para):
        instance_num = len(para)
        res = []
        for i in range(instance_num):
            max_score_list = []
            hypo_cur = hypo[i] # Every sent in hypo
            for cand in para[i]: # Compute the score of each sent in para (candidate)
                max_score_list.append(np.max(np.dot(hypo_cur, cand)))
            res.append(max_score_list)
        return res

    def forward(self, batch):
        '''
        batch:
        [{"hypothesis":[[1,2,3], [4,4], [5,6,7]],
          "paragraph": [[2,8], [8,7], [1], [6,6,3,9]],
          "target": [1, 0, 0],
          "id": "testing-1"}
        ]
        '''
        hypo_word_emb, para_word_emb = self._idx2emb(batch, self.word_emb)
        hypo_sent_emb = self._get_sent_emb(hypo_word_emb)
        para_sent_emb = self._get_sent_emb(para_word_emb)
        #para_score = self._fuse(hypo_sent_emb, para_sent_emb)

        '''
        hypo_sent_emb:
        tensor



        '''
        # for p in para_matrix:
        #     for hypo_matrix:

        return hypo_sent_emb
