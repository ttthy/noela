import os
import io
import re
import json
from collections import Counter, OrderedDict
import numpy as np
from dataclasses import dataclass, field
from typing import List
from ure.utils.data_utils import get_from_json
from ure.utils.tiktok import *


LOWER       = False  # make all characters lowercase
DIGIT_0     = False  # replace all digit by zero `0`
CHARACTER   = False
ADD_PAD_UNK = False
MIN_FREQ    = 1
UNK_TOKEN   = "#UNK#"  # string represent for unknown token
PAD_TOKEN   = "#PAD#"  # string represent for padding
BRACKETS    = {"-LCB-": "{", "-LRB-": "(", "-LSB-": "[", "-RCB-": "}", "-RRB-": ")", "-RSB-": "]"}


def load_vocab_from_file(voc_path, lower, digit_0, 
add_pad_unk=ADD_PAD_UNK, min_freq=MIN_FREQ, use_char=False, char_path=None):
    print("Loading vocabulary from saved file <{}".format(voc_path))
    word_vocab = Vocabulary.load(voc_path, lower=lower, digit_0=digit_0, add_pad_unk=add_pad_unk, min_freq=min_freq)
    
    char_vocab = None
    if use_char and char_path is not None:
        print("Loading character vocabulary from <{}".format(char_path))
        char_vocab = Vocabulary.load(char_path)
        return word_vocab, char_vocab
    return word_vocab


def load_vocab(data, lower, digit_0, add_pad_unk=ADD_PAD_UNK, min_freq=MIN_FREQ, use_char=False):
    print("Loading vocabulary from raw data...")
    word_vocab = Vocabulary.load(data, lower=lower, digit_0=digit_0, add_pad_unk=add_pad_unk, min_freq=min_freq)
    char_vocab = None
    if use_char:
        print("Loading character vocabulary from raw data...")
        char_raw_sentences = [
            list(w)
            for s in data
            for w in s
        ]
        char_vocab = Vocabulary.load(char_raw_sentences)
    return word_vocab, char_vocab



def load_pretrained_embeddings(embedding_file, word_dim, word_to_id,
                                digit_0=DIGIT_0, lower=LOWER):
    print("Loading pre-trained embeddings from < {}".format(embedding_file))
    embeddings = OrderedDict()
    with open(embedding_file, "r") as f:
        for i, line in enumerate(f):
            word = line.rstrip().split()[0]
            word = Vocabulary.normalise_string(word, digit_0=digit_0, lower=lower)
            # skip word that not in this dataset
            if word not in word_to_id: continue
            vec = line.rstrip().split()[1:]
            n = len(vec)
            if n != word_dim:
                print("Not same word dimensionality! -- \
                    line No{}, word: {}, len {}".format(i, word, n))
                continue
            embeddings[word] = np.asarray(vec, "f")
            if i < 3:
                print("Word IDs: {}:id:{} {}".format(word, word_to_id[word],  vec[-3:]))
    print("Pre-trained word embeddings: {} x {}".format(len(embeddings), word_dim))
    return embeddings


def create_mapping_from_counter(sorted_counter, min_freq=1):
    print("Found {} unique items in ({} in total)".format(
        len(sorted_counter), sum([v for k, v in sorted_counter])))
    id2item = []
    freq = []
    for k, v in sorted_counter:
        if v >= min_freq:
            id2item += [k]
            freq += [v]
    item2id = {k: i for i, k in enumerate(id2item)}
    return id2item, freq, item2id


def get_counter_from_file(filepath, lower=LOWER, digit_0=DIGIT_0):
    # for testing or prediction
    f = io.open(filepath, "r", encoding="utf-8", errors="ignore")
    # config = f.readline().rstrip()
    # skip the first line for config: lower/digit_0
    # config = json.loads(config)
    # print(config)
    counter = Counter()
    for line in f:
        line = line.strip()
        cols = line.split("\t")
        assert 0 < len(cols) < 3, """invalid vocabulary file: {}\n\
        The Vocabulary class only accept file with the following format 
        Word-i-th\\t[Frequency]""".format(filepath)
        # Token has been normalised before writing in file
        token = cols[0].strip()
        token = Vocabulary.normalise_string(
            token, lower, digit_0
        )
        counter[token] = int(cols[1]) if len(cols) == 2 else 1
    f.close()
    return counter


def get_counter_from_sentences(raw_sentences, lower=LOWER, digit_0=DIGIT_0):
        # for training
        # get statistics from training samples
        counter = Counter(
            [
                Vocabulary.normalise_string(tok, lower, digit_0)
                for s in raw_sentences
                for tok in s
            ]
        )
        return counter


def get_counter_from_list(s, lower=LOWER, digit_0=DIGIT_0):
    counter = Counter(
        [
            Vocabulary.normalise_string(tok, lower, digit_0)
            for tok in s
        ]
    )
    return counter


class Vocabulary(object):
    unk_token = UNK_TOKEN
    pad_token = PAD_TOKEN

    def __init__(self, lower=LOWER, digit_0=DIGIT_0, add_pad_unk=ADD_PAD_UNK, min_freq=MIN_FREQ):
        self.stoi = {}
        self.itos = []
        self.freq = []
        self.pad_id = -1
        self.unk_id = -1
        self.lower = lower
        self.digit_0 = digit_0
        self.add_pad_unk = add_pad_unk
        self.min_freq = min_freq

    @staticmethod
    def normalise_string(token, lower=LOWER, digit_0=DIGIT_0):
        if token in [Vocabulary.unk_token, Vocabulary.pad_token, "<s>", "</s>"]:
            return token
        elif token in BRACKETS:
            token = BRACKETS[token]
        elif digit_0:
            token = re.sub(r"\d", "0", token)
        if lower:
            return token.lower()
        return token
    
    def __load_from_counter(self, counter):
        pad_freq = counter.pop(self.pad_token, self.min_freq)
        unk_freq = counter.pop(self.unk_token, self.min_freq)
        
        sorted_counter = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        prefix = []
        if self.add_pad_unk:
            prefix += [(self.pad_token, pad_freq), (self.unk_token, unk_freq)]
        sorted_counter = prefix + sorted_counter
        (self.itos, self.freq, self.stoi) = create_mapping_from_counter(sorted_counter, self.min_freq)
        if self.add_pad_unk:
            self.pad_id = self.stoi[self.pad_token]
            self.unk_id = self.stoi[self.unk_token]

    @staticmethod
    def load(arg, lower=DIGIT_0, digit_0=DIGIT_0, add_pad_unk=ADD_PAD_UNK, min_freq=MIN_FREQ):
        vocab = Vocabulary(lower, digit_0, add_pad_unk, min_freq)
        if isinstance(arg, list) or isinstance(arg, tuple):
            if isinstance(arg[0], list) or isinstance(arg[0], tuple):
                counter = get_counter_from_sentences(arg, lower, digit_0)
            elif isinstance(arg[0], str):
                counter = get_counter_from_list(arg, lower, digit_0)
            else:
                raise Exception(
                    """Argument is invalid! Neither List or Tuple! \
                        Load only from raw sentences or vocab file... """
                )
        elif os.path.isfile(arg):
            print("\nLoading from vocabulary file <{}".format(arg))
            counter = get_counter_from_file(arg, lower, digit_0)
        else:
            raise Exception(
                "Argument is invalid! \
            Load only from raw sentences or vocab file... "
            )
        vocab.__load_from_counter(counter)
        return vocab

    def save(self, path):
        f = io.open(path, "w", encoding="utf-8", errors="ignore")
        s = json.dumps({"lower": self.lower, "digit_0": self.digit_0})
        f.write(s + "\n")
        for item, value in zip(self.itos, self.freq):
            f.write("{}\t{}\n".format(item, value))
        f.close()

    def size(self):
        return len(self.itos)
    
    def norm(self, token):
        return Vocabulary.normalise_string(token, self.lower, self.digit_0)

    def get_id(self, token):
        tok = self.norm(token)
        return self.stoi.get(tok, self.unk_id)

    def get_str(self, _id):
        return self.itos[_id]

    def __repr__(self):
        last_idx = -1 if self.add_pad_unk else 0
        least_freq = []
        for x in reversed(range(len(self.itos))):
            if x != self.pad_id and x != self.unk_id:
                least_freq.append(self.itos[x])
            if len(least_freq) == 5:
                break
        s = """
        Vocabulary size: {} ({} in total)
        Most freq      : {} \t {}
        Least freq     : {} \t {}
        Normalisation  : lower   = {}, digit_0  = {}
                         {}
        """.format(
            self.size(), sum(self.freq),
            self.freq[2], ", ".join(self.itos[2:7]),
            # freq[-1]=#UNK#
            self.min_freq, ", ".join(least_freq),
            self.lower, self.digit_0, 
            'UNK[{}] = {}, PAD[{}]  = {}'.format(
                self.unk_id, self.unk_token, self.pad_id, self.pad_token) 
                if self.add_pad_unk else 'No PAD/UNK'
        )
        return s

@dataclass
class Template:
    weight: float
    rid: str
    template: str


@dataclass
class Relation:
    rid: str
    _type: str
    label: str
    description: str
    

class ItemVocabulary(object):
    def __init__(self):
        self.idx2i = {}
        self.i2item = []
        self.size = 0

    @staticmethod
    def load_relations(path):
        predefined_relation = get_from_json(path)
        prel = ItemVocabulary()
        for i, r in enumerate(predefined_relation):
            prel.i2item.append(r)
            prel.idx2i[r['label']] = i
        prel.size = len(prel.i2item)
        return prel

    @staticmethod
    def load_predefined_relations(path):
        predefined_relation = get_from_json(path)
        prel = ItemVocabulary()
        for i, (rid, rel) in enumerate(predefined_relation.items()):
            r = Relation(rid=rid, _type=rel["type"],
                         label=rel["label"], description=rel["description"])
            prel.i2item.append(r)
            prel.idx2i[rid] = i
        prel.size = len(prel.i2item)
        return prel
    
    @staticmethod
    def load_templates(path):
        selected_templates = get_from_json(path)
        templs = ItemVocabulary()
        for i, temp in enumerate(selected_templates):
            t = Template(rid=temp["rid"], weight=temp["weight"],
                         template=temp["template"])
            templs.i2item.append(t)
        templs.size = len(templs.i2item)
        return templs

    def __len__(self):
        return self.size

    def get_id(self, idx):
        return self.idx2i[idx]

    def get_item(self, _id):
        return self.i2item[_id]

    def __repr__(self):
        s = "Vocabulary size: {}".format(self.size)
        return s
    

def load_vocabs(config):
    tik("load_vocas")

    relation_vocab = Vocabulary.load(config["path_data_relation"])
    vocas = {"relation": relation_vocab}

    # Vocabulary
    if "path_vocab" in config and config["path_vocab"]:
        vocab = Vocabulary.load(
            config["path_vocab"],
            add_pad_unk=config["add_pad_unk"],
            lower=config["lowercase"],  digit_0=config["digit_0"],
            min_freq=config["min_freq"])
        config["model"]["vocab_size"] = vocab.size()
        vocas["word"] = vocab

        # Check pretrained_word_embeddings, load if there is
        if "path_pretrained_word" in config and config["path_pretrained_word"]:
            print("\nPath_pretrained_word", config["path_pretrained_word"])
            config["model"]["word_vocab"] = vocab
            config["model"]["pretrained_word_embs"] = load_pretrained_embeddings(
                config["path_pretrained_word"],
                config["model"]["word_dim"],
                word_to_id=vocab.stoi,
                lower=config["lowercase"],
                digit_0=config["digit_0"])
    
    if "path_etype" in config and config["path_etype"]:
        print("\nLoad entity type vocabulary from", config["path_etype"])
        etype_voc = Vocabulary.load(
            config["path_etype"],
            add_pad_unk=False,
            lower=False, digit_0=False, min_freq=1
        )
        vocas["etype"] = etype_voc
        config["model"]["n_etype"] = vocas["etype"].size()

    if "path_predefined_relation" in config and config["path_predefined_relation"]:
        predefined_relation = ItemVocabulary.load_relations(
            config["path_predefined_relation"])
        vocas["predefined_relation"] = predefined_relation
    if "path_template" in config and config["path_template"]:
        selected_templates = ItemVocabulary.load_templates(
            config["path_template"])
        vocas["selected_templates"] = selected_templates
    tok("load_vocas")

    return vocas


def test_vocab_class():
    ss = "List giving the size of context for each item in the batch and used to compute a context_mask. If context_mask or context_sizes are not given, context is assumed to have fixed size. 56789"
    # sentence splitting: create list of sentences containing list of words
    raw_sentences = [[t for t in s.split()] for s in ss.split(". ")]
    print ("List of sentences:")
    print(raw_sentences, "\n")
    # get vocabuary from list of sentences
    print ("Loading vocabulary from list of sentences")
    vocab = Vocabulary.load(raw_sentences,
                            lower=True, digit_0=True)
    # print details of current vocabulary: size, most/least freq words, normalisation
    print("Current vocabulary details:", vocab)
    print("First 2 words and 2 characters:", vocab.itos[:2])
    # try the normalisation: replace all digits with 0
    # expected results: 56789 --> 00000
    print ("Get ID and Token of `56789`: {} and {} respectively".format(
        vocab.get_id("56789"), vocab.itos[vocab.get_id("56789")]))
    char_raw_sentences = [list(t) for s in raw_sentences for t in s]
    print ("List of characters:")
    print(char_raw_sentences, "\n")
    print ("Load character vocabulary...")
    vocab = Vocabulary.load(char_raw_sentences,
                            lower=True, digit_0=False)
    print(vocab)
    print ("List of unique characters:")
    print(vocab.itos)
    
    checkpoint_dir = "checkpoints/temp"
    p = os.path.join(checkpoint_dir, "a.txt")
    print ("Trying to save vocabulary to file: {}".format(p))
    if not os.path.isdir(checkpoint_dir):
        print("Invalid directory! Make directory {}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir)
    vocab.save(p)
    print ("Saving success!!!")
    print ("Loading vocabulary from saved file + add UNK word")
    vocab = Vocabulary.load(p, add_pad_unk=True)
    print ("Vocabulary details:")
    print(vocab)
    print("First 2 words and 2 characters:", vocab.itos[:2])
    print ("Congratulations! Finish vocabulary testing!")


def test_vocab_from_file():
    path = 'data/nyt/vocab.freq'
    word_vocab = load_vocab_from_file(path, lower=True, digit_0=True, add_pad_unk=True, min_freq=5)
    print(word_vocab)


if __name__ == "__main__":
    # test_vocab_class()
    # test_vocab_from_file()
    vocas = load_vocabs({
        "path_vocab": "./data/nyt/vocab.freq",
        "add_pad_unk": True,
        "lowercase": True,
        "digit_0": True,
        "min_freq": 1,
        "path_pretrained_word": "../../data/glove.6B.50d.txt",
        "path_relation": "./data/nyt/dict.relation",
        "path_predefined_relation": "./data/relations.json",
        "path_template": "./data/selected_templates.json",
        "model": {
            "word_dim": 50
        }
    })
    print (vocas)
    print(vocas['predefined_relation'].get_item("P19"))
    print(vocas['selected_templates'].get_item(0))
    print_time()
