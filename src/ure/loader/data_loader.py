import collections
import json
import os
import random

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from ure.loader.vocabulary import load_vocabs
from ure.utils.nn_utils import (to_cuda, make_equal_len, flatten_3Dto2D_lists, to_long_tensor, to_float_tensor)
from ure.utils.data_utils import (read_tsv, tokenize, get_position_wrt_entities, get_mask, find_mentions_position)


class BatchDataLoader(object):
    def __init__(self, path, vocas, max_len, batch_size,
                 max_position, mask_entity=False, is_training=True, 
                 n_cand=(3, 10),
                 parse_func="parse_line"):
        
        self.vocas = vocas
        self.max_len = max_len
        self.batch_size = batch_size
        self.max_position = max_position
        self.mask_entity = mask_entity
        self.is_training = is_training

        self.n_cand = n_cand

        # Loading data from file
        self.path = path
        self.parse_func = parse_func
        self.data = self.read_data()
        self.size = len(self.data)

        # After loading data from path
        self.data_loader = self.build_data_loader()

    def __repr__(self):
        s= """
        DataLoader configuration:
            - Max length:   {}
            - Batch size:   {}
            - Max position: {}
            - Mask entity:  {}
            - Is training:  {}
            - N candidates: {}
            - Path:         {}
            - Parse func:   {}
            - Data size:    {}
        """.format(
            self.max_len, self.batch_size, self.max_position, self.mask_entity,
            self.is_training,
            self.n_cand, self.path, self.parse_func, self.size)
        return s

    def build_data_loader(self):
        data_ids = TensorDataset(torch.arange(self.size))
        if self.is_training:
            sampler = RandomSampler(data_ids)
        else:
            sampler = SequentialSampler(data_ids)
        return DataLoader(
            data_ids, sampler=sampler, batch_size=self.batch_size)

    def __len__(self):
        return self.size

    def __iter__(self):
        for b in self.data_loader:
            # Data loader returns list of ID list
            yield (self.get_batch(b[0]))

    def process_batch(self, batch):
        raise NotImplementedError("Need to implement `process_batch` in specific dataset!")

    def get_batch(self, batch_ids):
        # Always padding with 0
        batch = [self.data[idx] for idx in batch_ids]
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        batch["batch_size"] = len(batch_ids)
        # print("keys in batch", batch.keys())
        return self.process_batch(batch)

    def process_raw_item(self, batch):
        raise NotImplementedError("Need to implement `process_raw_item` in specific dataset!")

    def read_data(self):
        raw_data = read_tsv(self.path, self.parse_func)
        data = []
        count_valid = 0
        for i, item in enumerate(raw_data):
            item = self.process_raw_item(item)
            if item is not None:
                data.append(item)
                count_valid += 1

                if count_valid % 1000 == 0:
                    print("load %d items" % count_valid, end="\r")
        print("Load {}/{} instances".format(count_valid, i))
        return data


class NYT(BatchDataLoader):
    def __init__(self, path, vocas, max_len, batch_size,
                 max_position, mask_entity=False, is_training=True,
                 n_cand=(3, 10),
                 parse_func="parse_line"):
        super(NYT, self).__init__(path, vocas, max_len, batch_size,
                                  max_position, mask_entity, is_training, 
                                  n_cand, parse_func)

    def process_batch(self, batch):
        # TODO N-pos, neg, "head_mention", "tail_mention",
        # "pos_templates", "neg_templates",
        batch["input_ids"].extend(flatten_3Dto2D_lists(
            batch.pop("positive_bag_input_ids")))
        batch["pos_wrt_head"].extend(flatten_3Dto2D_lists(
            batch.pop("pos_wrt_head_posbag")))
        batch["pos_wrt_tail"].extend(flatten_3Dto2D_lists(
            batch.pop("pos_wrt_tail_posbag")))
        batch["pcnn_mask"].extend(flatten_3Dto2D_lists(
            batch.pop("positive_pcnn_mask")))

        if self.is_training:
            batch["input_ids"].extend(flatten_3Dto2D_lists(
                batch.pop("negative_bag_input_ids")))
            batch["pos_wrt_head"].extend(flatten_3Dto2D_lists(
                batch.pop("pos_wrt_head_negbag")))
            batch["pos_wrt_tail"].extend(flatten_3Dto2D_lists(
                batch.pop("pos_wrt_tail_negbag")))
            batch["pcnn_mask"].extend(flatten_3Dto2D_lists(
                batch.pop("negative_pcnn_mask")))

        batch["lengths"] = to_long_tensor([len(x) for x in batch["input_ids"]])

        for k in ["input_ids", "pos_wrt_head", "pos_wrt_tail", "pcnn_mask"]:
            batch[k], masks = make_equal_len(batch[k])
            if k == "input_ids":
                batch["masks"] = to_float_tensor(masks)
            batch[k] = to_long_tensor(batch[k])

        batch["candidate2relation"] = to_long_tensor(batch["candidate2relation"])
        batch["pos"] = to_long_tensor(batch["pos"])
        batch["neg"] = to_long_tensor(batch["neg"])
        return batch
    
    def process_raw_item(self, item):
        if len(set(
            range(item["head_pos"][0],
                    item["head_pos"][1])
        ).intersection(
            range(item["tail_pos"][0],
                    item["tail_pos"][1]))) > 0:
            # Invalid if head and tail overlap
            return None

        # TODO Check entity type
        # head_etype = item["head_etype"] if self.vocas["etype"].get_id(
        #     item["head_etype"]) != self.vocas["etype"].unk_id else "MISC"
        # tail_etype = item["tail_etype"] if self.vocas["etype"].get_id(
        #     item["tail_etype"]) != self.vocas["etype"].unk_id else "MISC"
        # head_etype_subj = head_etype + "-SUBJ"
        # tail_etype_obj = tail_etype + "-OBJ"

        sentence = item["sentence"]
        tokens, head_pos, tail_pos = tokenize(
            sentence, item["head_pos"], item["tail_pos"],
            mask_entity=self.mask_entity)
        # mask_entity=self.mask_entity, head_mask=head_etype_subj, tail_mask=tail_etype_obj)

        n_tokens = len(tokens)
        if n_tokens > self.max_len:
            # Invalid if length > max_len
            return None
        if item["head_pos"][1] > n_tokens or item["tail_pos"][1] > n_tokens:
            return None
        relation = item["relation"]

        # TODO Entity ids
        head_mention, tail_mention, = item["head_mention"], item["tail_mention"]
        # head_ent_id = self.vocas["entity"].get_id(head_ent)
        # if head_ent_id == self.vocas["entity"].unk_id:
        #     head_ent_id = i * 2
        # tail_ent_id = self.vocas["entity"].get_id(tail_ent)
        # if tail_ent_id == self.vocas["entity"].unk_id:
        #     tail_ent_id = i * 2 + 1

        try:
            relation_id = self.vocas["relation"].get_id(relation)
        except:
            relation_id = 0

        (pos_wrt_head, pos_wrt_tail) = get_position_wrt_entities(
            self.max_position, head_pos, tail_pos, n_tokens)
        if len(pos_wrt_head) != len(pos_wrt_tail) == n_tokens:
            print("sentence {}".format(sentence))
            print("head_pos {}, tail_pos {}".format(item["head_pos"], item["tail_pos"]))
            print("poshead {} != tail {} != n_tokens{}".format(
                len(pos_wrt_head), len(pos_wrt_tail), n_tokens))
            print("head_pos {}, tail_pos {}".format(head_pos, tail_pos))
            print("tokens {}".format(tokens))
            return None

        mask = get_mask(self.max_position, head_pos, tail_pos, n_tokens)

        selected_templates = self.vocas["selected_templates"]
        positive_bags = []
        candidate2relation = []
        for t in item["pos_template_ids"]:
            temp = selected_templates.get_item(t)
            positive_bags.append(temp.template
                                    .replace('[X]', head_mention)
                                    .replace('[Y]', tail_mention))
            candidate2relation.append(self.vocas["predefined_relation"].get_id(temp.rid))

        positive_bags = [
            selected_templates.get_item(t).template
            .replace('[X]', head_mention)
            .replace('[Y]', tail_mention)
            for t in item["pos_template_ids"]
        ]
        # print('POS')
        # for t in item["pos_template_ids"]:
        #     print(t, selected_templates.get_item(t).template)
        # input()
        negative_bags = [
            selected_templates.get_item(t).template
            .replace('[X]', head_mention)
            .replace('[Y]', tail_mention)
            for t in item["neg_template_ids"]
        ]
        # print('NEG')
        # for t in item["neg_template_ids"]:
        #     print(t, selected_templates.get_item(t).template)
        # input()

        # Separate now so later will keep only positive bag
        # May randomly generate negative bag while batching
        pos_wrt_head_posbag = []
        pos_wrt_tail_posbag = []
        positive_bag_input_ids = []
        positive_mask = []
        for x in positive_bags:
            _head_pos, _tail_pos = find_mentions_position(
                x, head_mention, tail_mention)
            tids = [self.vocas["word"].get_id(t) for t in x.split()]
            n_tids = len(tids)
            (_pos_wrt_head, _pos_wrt_tail) = get_position_wrt_entities(
                self.max_position, _head_pos, _tail_pos, n_tids)
            _mask = get_mask(self.max_position, _head_pos, _tail_pos, n_tids)
            pos_wrt_head_posbag.append(_pos_wrt_head)
            pos_wrt_tail_posbag.append(_pos_wrt_tail)
            positive_bag_input_ids.append(tids)
            positive_mask.append(_mask)

        pos_wrt_head_negbag = []
        pos_wrt_tail_negbag = []
        negative_bag_input_ids = []
        negative_mask = []
        for x in negative_bags:
            _head_pos, _tail_pos = find_mentions_position(
                x, head_mention, tail_mention)
            tids = [self.vocas["word"].get_id(t) for t in x.split()]
            n_tids = len(tids)
            (_pos_wrt_head, _pos_wrt_tail) = get_position_wrt_entities(
                self.max_position, _head_pos, _tail_pos, n_tids)
            _mask = get_mask(self.max_position, _head_pos, _tail_pos, n_tids)
            pos_wrt_head_negbag.append(_pos_wrt_head)
            pos_wrt_tail_negbag.append(_pos_wrt_tail)
            negative_bag_input_ids.append(tids)
            negative_mask.append(_mask)

        # TODO input_ids
        input_ids = [self.vocas["word"].get_id(t) for t in tokens]
        item["relation"] = relation
        item["relation_id"] = relation_id
        # TODO current inference only uses positive bag relations
        item["candidate2relation"] = candidate2relation
        item["input_ids"] = input_ids
        item["tokens"] = tokens
        item["pos_wrt_head"] = pos_wrt_head
        item["pos_wrt_tail"] = pos_wrt_tail
        item["pcnn_mask"] = mask
        item["positive_bags"] = positive_bags
        item["negative_bags"] = negative_bags
        item["positive_bag_input_ids"] = positive_bag_input_ids
        item["negative_bag_input_ids"] = negative_bag_input_ids
        item["pos_wrt_head_posbag"] = pos_wrt_head_posbag
        item["pos_wrt_tail_posbag"] = pos_wrt_tail_posbag
        item["pos_wrt_head_negbag"] = pos_wrt_head_negbag
        item["pos_wrt_tail_negbag"] = pos_wrt_tail_negbag
        item["positive_pcnn_mask"] = positive_mask
        item["negative_pcnn_mask"] = negative_mask
        return item


class TACRED(BatchDataLoader):

    def __init__(self, path, vocas, max_len, batch_size,
                 max_position, mask_entity=False, is_training=True,
                 n_cand=(3, 10), parse_func="parse_line_w_bags"):
        super(TACRED, self).__init__(path, vocas, max_len, batch_size,
                                  max_position, mask_entity, is_training,
                                  n_cand, parse_func)

    def process_batch(self, batch):
        batch["lengths"] = to_long_tensor([len(x) for x in batch["input_ids"]])

        for k in ["input_ids", "pos_wrt_head", "pos_wrt_tail", "pcnn_mask"]:
            batch[k], masks = make_equal_len(batch[k])
            if k == "input_ids":
                batch["masks"] = to_float_tensor(masks)
            batch[k] = to_long_tensor(batch[k])

        # batch["relation_dist"] = to_float_tensor(batch["relation_dist"])
        if self.is_training:
            batch["pos_masks"] = to_float_tensor(batch["pos_masks"])
            batch["pos"] = to_long_tensor(batch["pos"])
            batch["neg"] = to_long_tensor(batch["neg"])
        return batch

    def process_raw_item(self, item):
        if len(set(
            range(item["head_pos"][0],
                  item["head_pos"][1])
        ).intersection(
            range(item["tail_pos"][0],
                  item["tail_pos"][1]))) > 0:
            # Invalid if head and tail overlap
            return None

        # TODO Check entity type
        head_etype = item["head_etype"]
        head_etype_subj = "{}-SUBJ".format(head_etype)
        tail_etype = item["tail_etype"]
        tail_etype_obj = "{}-OBJ".format(tail_etype)
        head_etype_subj = head_etype_subj if self.vocas["word"].get_id(
            head_etype_subj) != self.vocas["word"].unk_id else "MISC-SUBJ"
        tail_etype_obj = tail_etype_obj if self.vocas["word"].get_id(tail_etype_obj) != self.vocas["word"].unk_id else "MISC-OBJ"
        
        sentence = item["sentence"]
        tokens, head_pos, tail_pos = tokenize(
            sentence, item["head_pos"], item["tail_pos"],
            mask_entity=self.mask_entity, 
            head_mask=head_etype_subj,
            tail_mask=tail_etype_obj)

        n_tokens = len(tokens)
        if n_tokens > self.max_len:
            # Invalid if length > max_len
            return None
        if item["head_pos"][1] > n_tokens or item["tail_pos"][1] > n_tokens:
            return None
        relation = item["relation"]
        # TODO Entity ids
        head_mention, tail_mention, = item["head_mention"], item["tail_mention"]
        # head_ent_id = self.vocas["entity"].get_id(head_ent)
        # if head_ent_id == self.vocas["entity"].unk_id:
        #     head_ent_id = i * 2
        # tail_ent_id = self.vocas["entity"].get_id(tail_ent)
        # if tail_ent_id == self.vocas["entity"].unk_id:
        #     tail_ent_id = i * 2 + 1

        try:
            relation_id = self.vocas["relation"].get_id(relation)
        except:
            print("Except!!!!")
            return None

        (pos_wrt_head, pos_wrt_tail) = get_position_wrt_entities(
            self.max_position, head_pos, tail_pos, n_tokens)
        if len(pos_wrt_head) != len(pos_wrt_tail) or len(pos_wrt_head) != n_tokens:
            print("sentence {}".format(sentence))
            print("head_pos {}, tail_pos {}".format(item["head_pos"], item["tail_pos"]))
            print("poshead {} != tail {} != n_tokens{}".format(
                len(pos_wrt_head), len(pos_wrt_tail), n_tokens))
            print("head_pos {}, tail_pos {}".format(head_pos, tail_pos))
            print("tokens {}".format(tokens))
            return None

        mask = get_mask(self.max_position, head_pos, tail_pos, n_tokens)
        # assert len(item["pos_template_ids"]) == 41, "wrong # of relations!!!!"
        
        # TODO input_ids
        input_ids = [self.vocas["word"].get_id(t) for t in tokens]
        item["relation_id"] = relation_id
        item["input_ids"] = input_ids
        item["tokens"] = tokens
        item["pos_wrt_head"] = pos_wrt_head
        item["pos_wrt_tail"] = pos_wrt_tail
        item["pcnn_mask"] = mask
        # if inputs are scores
        # item["relation_dist"] = item.pop("pos_template_ids")
        # sorted_indices = np.argsort(item['relation_dist'])[::-1]
        # item["pos"] = sorted_indices[:self.n_cand[0]].tolist()
        # item["neg"] = sorted_indices[-self.n_cand[1]:].tolist()

        if self.is_training:
            item["pos"] = item.pop("pos_template_ids")
            item["neg"] = item.pop("neg_template_ids")
            last_pos_id = item["pos"][-1]
            n_pos, n_neg = self.n_cand
            item["pos_masks"] = [1]*len(item["pos"])
            # Get positive bags
            if len(item["pos"]) > n_pos:
                item["pos"] = item["pos"][:n_pos]
            else:
                while len(item["pos"]) < n_pos:
                    item["pos"] += [last_pos_id]
                    item["pos_masks"] += [0]
            # Get neg bags
            last_neg_id = item["neg"][-1]
            if len(item["neg"]) > n_neg:
                flip = np.random.randint(0, 2)
                if flip == 0:
                    item["neg"] = item["neg"][:n_neg]
                elif flip == 1:
                    item["neg"] = item["neg"][len(item["neg"]) - n_neg:]
            else:
                while len(item["neg"]) < n_neg:
                    item["neg"] += [last_neg_id]

        return item


def test_nyt():
    config = {
        "path_train": "./data/nyt/sample.tsv",
        "path_vocab": "./data/nyt/vocab.freq",
        "add_pad_unk": True,
        "lowercase": True,
        "digit_0": True,
        "min_freq": 1,
        "max_len": 100,
        # "path_pretrained_word": "../../data/glove.6B.50d.txt",
        "path_pretrained_word": None,
        "path_data_relation": "./data/nyt/dict.relation",
        "path_predefined_relation": "./data/relations.json",
        "path_template": "./data/selected_templates.json",
        "model": {
            "word_dim": 50,
            "max_position": 100,
            "mask_entity": False
        },
        "training": {
            "batch_size": 2
        },
    }
    vocas = load_vocabs(config)
    train_data = NYT(
        config["path_train"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        mask_entity=config["model"]["mask_entity"],
        is_training=False,
        parse_func="parse_line")
    for _, batch_data in enumerate(train_data):
        # TODO check if correct
        cur_bsize = len(batch_data)
        input()


if __name__ == "__main__":
    # test_nyt()
    config = {
        "path_train": "./data/tacred/dev.txt.filtered",
        "path_vocab": "../ure/data/TACRED/vocab.freq",
        "add_pad_unk": True,
        "lowercase": True,
        "digit_0": True,
        "min_freq": 1,
        "max_len": 100,
        # "path_pretrained_word": "../../data/glove.6B.50d.txt",
        "path_pretrained_word": None,
        "path_data_relation": "../ure/data/TACRED/dict.relation",
        "path_predefined_relation": "./data/tacred/relations.json",
        "model": {
            "word_dim": 50,
            "max_position": 100,
            "mask_entity": True
        },
        "training": {
            "batch_size": 2
        },
    }
    vocas = load_vocabs(config)
    print(vocas)
    train_data = TACRED(
        config["path_train"],
        vocas=vocas,
        max_len=config["max_len"],
        batch_size=config["training"]["batch_size"],
        max_position=config["model"]["max_position"],
        mask_entity=config["model"]["mask_entity"],
        is_training=False,
        parse_func="parse_line_w_weight")
    for _, batch_data in enumerate(train_data):
        # TODO check if correct
        cur_bsize = len(batch_data)
        print(batch_data)
        head_et = batch_data["head_etype"][0] + '-SUBJ'
        hid = vocas["word"].get_id(head_et)
        print(head_et, hid, vocas["word"].get_str(hid))
        input()
