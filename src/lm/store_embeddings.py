import os
import argparse
import torch
import numpy as np
import json
import pdb
# GPT2
from lm.configuration_gpt2 import (GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
                                   GPT2Config)
from lm.modeling_gpt2 import (GPT2_PRETRAINED_MODEL_ARCHIVE_MAP,
                              GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Model,
                              GPT2PreTrainedModel, load_tf_weights_in_gpt2)
from lm.tokenization_gpt2 import GPT2Tokenizer

# BERT
from lm.configuration_bert import (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig)
from lm.modeling_bert import (BERT_PRETRAINED_MODEL_ARCHIVE_MAP, BertForMaskedLM,
                              BertForNextSentencePrediction, BertModel)
from lm.tokenization_bert import BertTokenizer

from ure.utils.data_utils import read_raw_tsv, get_from_json, get_raw_predefined_relations
from ure.utils.nn_utils import (get_gpu_memory, current_memory_usage, to_cuda,
                                make_equal_len, to_long_tensor, to_float_tensor)

import h5py

MASK = "[MASK]"
PAD  = "[PAD]"
CLS  = "[CLS]"
SEP  = "[SEP]"
EOT = "<|endoftext|>"


def extract_output_from_offset(features, offsets, time_dimension=1, mode="mean"):
    """ Indexing features from a 2D tensor
    
    Arguments:
        features {torch.tensor} -- [batch_size, seq_len, feature_dim]
        offsets {[torch.tensor]} -- [batch_size, batch_size]
    
    Returns:
        {torch.tensor} -- [batch_size, feature_dim]
    """
    if time_dimension == 1:
        batch_size, seq_len, _ = features.size()
    else:
        seq_len, batch_size, _ = features.size()

    # Create word position ranges
    # [seq_len]
    ranges = to_long_tensor(torch.arange(seq_len))
    # [batch_size, seq_len]
    ranges = ranges.unsqueeze(0).expand(batch_size, seq_len)
    # Filter out irrelevant positions
    # [batch_size, seq_len]
    idx = (torch.ge(ranges, offsets[:, 0].unsqueeze(-1)) &
           torch.lt(ranges, offsets[:, 1].unsqueeze(-1))).float()

    if mode == "max":
        # [batch_size, dim]
        output = (idx.unsqueeze(time_dimension) * features).max(dim=time_dimension, keepdim=False)
    else:
        # [batch_size, dim]
        output = torch.bmm(idx.unsqueeze(time_dimension), features).squeeze(time_dimension)
        if mode == "sum":
            pass
        elif mode == "mean":
            # [batch_size]
            idx = torch.sum(idx, dim=time_dimension).unsqueeze(-1)
            # [batch_size, dim]
            output = torch.div(output, idx)
        else:
            raise NotImplementedError("Mode <{}> is invalid!".format(mode))

    return output


def load_samples(rel_path):
    with open(rel_path) as f:
        rels = json.load(f)
    return rels


class RandomRunner(object):
    def __init__(self, model_dir, rel_path):
        self.templates = self._processSamples(rel_path)
        self.template2id = dict([(t['relation'], i) for i, t in enumerate(self.templates)])
        self.n_templates = len(self.templates)

    def _processSamples(self, rel_path):
        return load_samples(rel_path)

    def convert_inputs(self, item):
        # valid_rels = item["valid_rels"]
        predicted_probs = [0]*self.n_templates
        # predicted_probs[np.random.choice(valid_rels)] = 1
        predicted_probs[np.random.randint(self.n_templates)] = 1
        return predicted_probs

    def scoreBatch(self, batch_data):
        return batch_data


class BERTEntityMentionRunner(object):
    def __init__(self, model_dir, rel_path, pooling_mode="mean"):
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(model_dir, "vocab.txt"), do_lower_case=True)
        model = BertModel.from_pretrained(model_dir)
        self.pooling_mode = pooling_mode
        self.model = to_cuda(model)
        self.model.eval()
        self.model.summary()
        self.mask_id = self.tokenizer.convert_tokens_to_ids([MASK])[0]
        self.pad_id = self.tokenizer.convert_tokens_to_ids([PAD])[0]
        self.cls_id = self.tokenizer.convert_tokens_to_ids([CLS])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids([SEP])[0]
        self.samples = self._process_samples(rel_path)
        self.sample_embeddings = self.__samples2embeddings()
        self.sample2id = dict([(t['relation'], i) for i, t in enumerate(self.samples)])
        self.freeze_params()
        self.n_samples = len(self.samples)

    def freeze_params(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def get_sequence_output(self, inputs):
        # [batch_size, seq_len, hidden_size]
        sequence_output, _ = self.model(inputs["input_ids"],
                                       attention_mask=inputs["masks"],
                                       token_type_ids=inputs["token_type_ids"])
        return sequence_output

    def get_entity_pooling_embeddings(self, inputs, sequence_output=None):
        head_positions = inputs["head_position"]
        tail_positions = inputs["tail_position"]
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.model(inputs["input_ids"],
                                           attention_mask=inputs["masks"],
                                           token_type_ids=inputs["token_type_ids"])
        head_embeddings = extract_output_from_offset(
            sequence_output, head_positions, mode=self.pooling_mode)
        tail_embeddings = extract_output_from_offset(
            sequence_output, tail_positions, mode=self.pooling_mode)
        return head_embeddings, tail_embeddings

    def get_entity_start_embeddings(self, inputs, sequence_output=None):
        head_positions = inputs["head_position"]
        tail_positions = inputs["tail_position"]
        batch_size, _ = head_positions.size()
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.model(inputs["input_ids"],
                                           attention_mask=inputs["masks"],
                                           token_type_ids=inputs["token_type_ids"])
        head_embeddings = sequence_output[torch.arange(batch_size), head_positions[:, 0]]
        tail_embeddings = sequence_output[torch.arange(batch_size), tail_positions[:, 0]]
        return head_embeddings, tail_embeddings

    def get_entity_end_embeddings(self, inputs, sequence_output=None):
        head_positions = inputs["head_position"]
        tail_positions = inputs["tail_position"]
        batch_size, _ = head_positions.size()
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.model(inputs["input_ids"],
                                           attention_mask=inputs["masks"],
                                           token_type_ids=inputs["token_type_ids"])
        head_embeddings = sequence_output[torch.arange(batch_size), head_positions[:, 1]]
        tail_embeddings = sequence_output[torch.arange(batch_size), tail_positions[:, 1]]
        return head_embeddings, tail_embeddings

    def _process_samples(self, rel_path):
        raw_samples = load_samples(rel_path)

        for temp in raw_samples:
            # TODO change if more than 1 template
            sample = temp["sample"]
            pos_head = temp["head_position"]
            pos_tail = temp["tail_position"]
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                ent_0 = self.tokenizer.encode(temp["tail"], add_special_tokens=False)
                ent_1 = self.tokenizer.encode(temp["head"], add_special_tokens=False)
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                ent_0 = self.tokenizer.encode(temp["head"], add_special_tokens=False)
                ent_1 = self.tokenizer.encode(temp["tail"], add_special_tokens=False)
                rev = False
            sent_0 = [self.cls_id] + self.tokenizer.encode(
                                sample[:pos_min[0]], add_special_tokens=False)
            sent_1 = self.tokenizer.encode(
                sample[pos_min[1]:pos_max[0]], add_special_tokens=False)
            sent_2 = self.tokenizer.encode(
                sample[pos_max[1]:], add_special_tokens=False) + [self.sep_id]
            temp["tokens"] = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
            if rev:
                tail_tok_pos = len(sent_0)
                tail_tok_pos = (tail_tok_pos, tail_tok_pos+len(ent_0))
                temp["tail_tok_pos"] = tail_tok_pos
                head_tok_pos = tail_tok_pos[1] + len(sent_1)
                head_tok_pos = (head_tok_pos, head_tok_pos+len(ent_1))
                temp["head_tok_pos"] = head_tok_pos
            else:
                head_tok_pos = len(sent_0)
                head_tok_pos = (head_tok_pos, head_tok_pos+len(ent_0))
                temp["head_tok_pos"] = head_tok_pos
                tail_tok_pos = head_tok_pos[1] + len(sent_1)
                tail_tok_pos = (tail_tok_pos, tail_tok_pos+len(ent_1))
                temp["tail_tok_pos"] = tail_tok_pos
            temp["reverse_entities"] = rev
        return raw_samples

    def tokenize(self, sentence, head_position, tail_position,
                 mask_entity=False,
                 head_mask="#UNK#", tail_mask='#UNK#'):

        if head_position[0] > tail_position[0]:
            pos_min = tail_position
            pos_max = head_position
            rev = True
        else:
            pos_min = head_position
            pos_max = tail_position
            rev = False
        piece_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
        piece_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        piece_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        
        # Get new entity positions based on sub-words
        if rev:
            tail_position = len(piece_0)+1
            tail_position = [tail_position, tail_position + len(ent_0)]
            head_position = tail_position[1] + len(piece_1)
            head_position = [head_position, head_position + len(ent_1)]
        else:
            head_position = len(piece_0) + 1
            head_position = [head_position, head_position + len(ent_0)]
            tail_position = head_position[1] + len(piece_1)
            tail_position = [tail_position, tail_position + len(ent_1)]

        tokens = ["CLS"] + piece_0 + ent_0 + piece_1 + ent_1 + piece_2 + ["SEP"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0]*len(tokens)

        # pdb.set_trace()
        return {
            "input_ids": token_ids,
            "token_type_ids": token_type_ids,
            "head_position": head_position, "tail_position": tail_position,
            "length": len(token_ids)
        }

    def __sample2inputs(self, sample_item):
        length = len(sample_item["tokens"])
        return {
            "input_ids": sample_item["tokens"],
            "token_type_ids": [0]*length,
            "head_position": sample_item["head_tok_pos"], 
            "tail_position": sample_item["tail_tok_pos"],
            "length": length
        }

    def process_batch(self, batch):
        # Always padding with 0
        batch_size = len(batch)
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        batch["batch_size"] = batch_size
        for k in ["input_ids", "token_type_ids"]:
            if k in batch:
                batch[k], masks = make_equal_len(batch[k])
                if k == "input_ids":
                    batch["masks"] = to_float_tensor(masks)
                batch[k] = to_long_tensor(batch[k])

        for k in ["head_position", "tail_position"]:
            batch[k] = to_long_tensor(batch[k])           
        return batch

    def get_batch_embeddings(self, batch_data):
        outputs = self.model(batch_data["input_ids"])[0]
        head_embeddings, tail_embeddings = self.get_entity_pooling_embeddings(batch_data)
        pair_embeddings = torch.cat([head_embeddings, tail_embeddings], dim=-1)
        return pair_embeddings

    def __samples2embeddings(self):
        batch = []
        for t in self.samples:
            batch.append(self.__sample2inputs(t))
        batch_data = self.process_batch(batch)
        return self.get_batch_embeddings(batch_data)

    def convert_inputs(self, items):
        batch = []
        for item in items:
            tokenized_sentence = self.tokenize(
                item['sentence'], item["head_position"], item["tail_position"])
            batch.append(tokenized_sentence)
        return batch

    def scoreBatch(self, batch_data):
        batch_data = self.process_batch(batch_data)
        batch_size = batch_data["batch_size"]
        pair_embeddings = self.get_batch_embeddings(batch_data)
        # import pdb; pdb.set_trace()
        # [N, D]
        samples = self.sample_embeddings
        similarity_scores = torch.nn.functional.cosine_similarity(
            pair_embeddings.unsqueeze(1).repeat(1, self.n_samples, 1), 
            samples.unsqueeze(0).repeat(batch_size, 1, 1),
            dim=-1).detach().data.cpu().numpy().tolist()
        return similarity_scores

    def forward(self, batch_data):
        batch_data = self.process_batch(batch_data)
        batch_size = batch_data["batch_size"]
        pair_embeddings = self.get_batch_embeddings(batch_data)
        return pair_embeddings.detach().data.cpu().numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rel", default="data/tacred/relations.json")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model name, e.g., gpt2/bertmaskedlm/bertnextsent")
    parser.add_argument("--path", type=str, help="Model directory")
    # parser.add_argument("--char_pos", action="store_true", help="Character-based position")
    # parser.add_argument("--drop_instance", action="store_true", 
    #                     help="drop invalid relation instance")
    # parser.add_argument("--etype_filter", action="store_true", 
    #                     help="filtering valid relations based on entity types")
    parser.add_argument("--pool_mode", type=str, default="mean", help="mean, sum, max")
    parser.add_argument("--inpath", type=str, default="./data/tacred", 
                        help="Data directory")
    parser.add_argument("--outpath", type=str, help="Data embeddings")
    parser.add_argument("--reloutpath", type=str, default=None, 
                        help="Relation embeddings output")

    args = parser.parse_args()
    
    if args.model == "random":
        runner = RandomRunner(args.path, args.rel)
    elif args.model == 'bertmention':
        runner = BERTEntityMentionRunner(args.path, args.rel, args.pool_mode)

    print("Finding top templates from BERT.....")
    batchsize = 100
    with torch.no_grad():
        filepath = args.inpath
        outfile = args.outpath
        print("Reading from < {}\nWriting to > {}".format(filepath, outfile))

        raw_data = read_raw_tsv(filepath)
        datasize = len(raw_data)
        # with h5py.File(outfile, "w") as writer:
        #     for i in range(0, datasize, batchsize):
        #         start_idx = i*batchsize
        #         print(start_idx, end='\r')
        #         batch = raw_data[i:min(datasize, i+batchsize)]
        #         batch_data = runner.convert_inputs(batch)
        #         batch_embeddings = runner.forward(batch_data)

        #         for batch_id, item in enumerate(batch):
        #             gold_relation = item['relation']
        #             ds = writer.create_dataset(
        #                 "{}".format(start_idx + batch_id),
        #                 (1536,), dtype='float32',
        #                 data=batch_embeddings[batch_id]
        #             )
        #     print("{}....loaded!!!!!".format(datasize))

        if args.reloutpath is not None:
            with h5py.File(args.reloutpath, "w") as writer:
                sample_embeddings = runner.sample_embeddings.detach().cpu().numpy()
                for sample_id in range(runner.n_samples):
                    ds = writer.create_dataset(
                        "{}".format(sample_id),
                        (1536,), dtype='float32',
                        data=sample_embeddings[sample_id]
                    )

    print("Finish BERT data loop!")

