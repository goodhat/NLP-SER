import sys
import numpy as np
import ujson as json
import copy


def load_data(path, dict=None):
    #data = pd.read_json(path, lines=True)
    with open(path, "r") as f:
        data = json.load(f)
    if dict != None:
        data = sents2idx(data, dict)
    return data

def save_checkpoint(model, filename, epoch):
    model_state = model.state_dict()
    params = {
        'state': model_state,
    }
    try:
        torch.save(params, filename)
    except:
        print("Fail to save...", file=sys.stderr)

def load_checkpoint(filename):
    state = torch.load(weights_file)
    self.seen = state['seen']
    self.load_state_dict(state['weights'])

def print_model_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print (name, param.size())

def sents2idx(data, to_idx):
    res = copy.deepcopy(data)
    for i, instance in enumerate(data):
        hypo_instance = []
        para_instance = []
        for sent in instance["paragraph"]:
            sent_res = []
            for token in sent:
                try:
                    sent_res.append(to_idx[token])
                except:
                    sent_res.append(1) # OOV
            para_instance.append(sent_res)
        for sent in instance["hypothesis"]:
            sent_res = []
            for token in sent:
                try:
                    sent_res.append(to_idx[token])
                except:
                    sent_res.append(1) # OOV
            hypo_instance.append(sent_res)
        res[i]["paragraph"] = para_instance
        res[i]["hypothesis"] = hypo_instance
    return res

def token2sents(token_list, labels=None):
    tmp = []
    for i,sent in enumerate(token_list):
        if labels!=None and labels[i] == 0:
            continue
        tmp.append(''.join(sent))
    res = ''.join(tmp)
    return res
