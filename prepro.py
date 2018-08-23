import numpy as np
import ujson as json
from tqdm import tqdm


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)

def get_embedding(limit=-1, emb_file=None, size=517015, vec_size=300):
    print("Generating word embedding...")
    embedding_dict = {}
    if emb_file is not None:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                embedding_dict[word] = vector
            print("{} tokens have corresponding word embedding vector".format(
                len(embedding_dict)))
    NULL = "--NULL--"
    OOV = "--OOV--"

    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token] for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    # use token2idx_dict to get the index of word and look up emb_mat with the index to get the word vector
    return emb_mat, token2idx_dict


def prepro(emb_file, data_path):
    emb_mat, word2idx_dict = get_embedding(emb_file = emb_file)
    save(data_path+"word_emb_file", emb_mat, message="word embedding")
    save(data_path+"word_dictionary", word2idx_dict, message="word dictionary")

    return

emb_file = "/Users/goodhat/NTU/thirdDown/summerIntern/SERproject/task.CKIP.WordEmbedding/Glove_CNA_ASBC_300d.vec"
data_path = "data/"
prepro(emb_file, data_path)
