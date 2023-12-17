import os
import torch
import torch.nn as nn
import ure.utils.nn_utils as nn_utils
from ure.nn.base import BaseModel
from ure.nn.module import EmbedLayer, LSTMLayer
from lm.modeling_bert import BertModel, BertPreTrainedModel
from lm.tokenization_bert import BertTokenizer


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
    ranges = nn_utils.to_long_tensor(torch.arange(seq_len))
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


class BertBase(BaseModel):
    def __init__(self, config):
        super(BertBase, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(config["bert_dir"], "vocab.txt"), do_lower_case=config["lowercase"])
        self.bert = BertModel.from_pretrained(config["bert_dir"])

    def freeze_params(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_sequence_output(self, inputs):
        # [batch_size, seq_len, hidden_size]
        sequence_output, _ = self.bert(inputs["input_ids"],
                                       attention_mask=inputs["masks"],
                                       token_type_ids=inputs["token_type_ids"])
        return sequence_output

    def get_entity_pooling_embeddings(self, inputs, sequence_output=None):
        head_positions =  inputs["head_pos"]
        tail_positions = inputs["tail_pos"]
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.bert(inputs["input_ids"],
                                attention_mask=inputs["masks"],
                                token_type_ids=inputs["token_type_ids"])        
        head_embeddings = extract_output_from_offset(
            sequence_output, head_positions, mode=self.pooling_mode)
        tail_embeddings = extract_output_from_offset(
            sequence_output, tail_positions, mode=self.pooling_mode)
        return head_embeddings, tail_embeddings

    def get_entity_start_embeddings(self, inputs, sequence_output=None):
        head_positions = inputs["head_pos"]
        tail_positions = inputs["tail_pos"]
        batch_size, _ = head_positions.size()
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.bert(inputs["input_ids"],
                                           attention_mask=inputs["masks"],
                                           token_type_ids=inputs["token_type_ids"])
        head_embeddings = sequence_output[torch.arange(batch_size), head_positions[:, 0]]
        tail_embeddings = sequence_output[torch.arange(batch_size), tail_positions[:, 0]]
        return head_embeddings, tail_embeddings

    def get_entity_end_embeddings(self, inputs, sequence_output=None):
        head_positions = inputs["head_pos"]
        tail_positions = inputs["tail_pos"]
        batch_size, _ = head_positions.size()
        # [batch_size, seq_len, hidden_size]
        if sequence_output is None:
            sequence_output, _ = self.bert(inputs["input_ids"],
                                           attention_mask=inputs["masks"],
                                           token_type_ids=inputs["token_type_ids"])
        head_embeddings = sequence_output[torch.arange(batch_size), head_positions[:, 1]]
        tail_embeddings = sequence_output[torch.arange(batch_size), tail_positions[:, 1]]
        return head_embeddings, tail_embeddings
   

    def tokenize(self, sentence, head_pos, tail_pos, 
                mask_entity=False, 
                head_mask="#UNK#", tail_mask='#UNK#'):
                
        if head_pos[0] > tail_pos[0]:
            pos_min = tail_pos
            pos_max = head_pos
            rev = True
        else:
            pos_min = head_pos
            pos_max = tail_pos
            rev = False
        piece_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
        piece_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        piece_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        ent_0  = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        ent_1  = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        if mask_entity:
            if mask_entity == "add_marker":
                head_marks = ["[unused0]", "[unused1]"]
                tail_marks = ["[unused2]", "[unused3]"]
                if rev:
                    ent_0 = [tail_marks[0]] + ent_0 + [tail_marks[1]]
                    ent_1 = [head_marks[0]] + ent_1 + [head_marks[1]]
                else:
                    ent_0 = [head_marks[0]] + ent_0 + [head_marks[1]]
                    ent_1 = [tail_marks[0]] + ent_1 + [tail_marks[1]]
            else:
                if rev:
                    ent_0 = [tail_mask]*len(ent_0)
                    ent_1 = [head_mask]*len(ent_1)
                else:
                    ent_0 = [head_mask]*len(ent_0)
                    ent_1 = [tail_mask]*len(ent_1)

        # Get new entity positions based on sub-words
        if rev:
            tail_pos = len(piece_0)+1
            tail_pos = [tail_pos, tail_pos + len(ent_0)]
            head_pos = tail_pos[1] + len(piece_1)
            head_pos = [head_pos, head_pos + len(ent_1)]
        else:
            head_pos = len(piece_0) + 1
            head_pos = [head_pos, head_pos + len(ent_0)]
            tail_pos = head_pos[1] + len(piece_1)
            tail_pos = [tail_pos, tail_pos + len(ent_1)]

        tokens = ["CLS"] +  piece_0 + ent_0 + piece_1 + ent_1 + piece_2 + ["SEP"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0]*len(tokens)

        return {
            "tokens": tokens, "input_ids": token_ids,
            "token_type_ids": token_type_ids, 
            "head_pos": head_pos, "tail_pos": tail_pos,
            "length": len(token_ids)
        }


class BertEntityStart(BertBase):
    def __init__(self, config):
        super(BertEntityStart, self).__init__(config)
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(2*config["bert_hidden"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        head_mention_embeddings, tail_mention_embeddings = self.get_entity_start_embeddings(inputs)
        rel_embeddings = torch.cat([head_mention_embeddings, tail_mention_embeddings], dim=-1)
        return self.fc(rel_embeddings)


class BertEntityStartEType(BertBase):
    def __init__(self, config):
        super(BertEntityStartEType, self).__init__(config)
        self.etype_embeds = EmbedLayer(
            num_embeddings=config["n_etype"],
            embedding_dim=config["etype_dim"],
            # dropout=config['word_dropout'],
        )
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(2*config["bert_hidden"] + 2*config["etype_dim"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']
        self.cached_mention_embeddings = {}

    def get_cached_embeddings(self, inputs):
        rel_embeddings = self.get_rel_embeddings(inputs)
        if inputs['data_partition'] in self.cached_mention_embeddings:
            self.cached_mention_embeddings[inputs['data_partition']].append(rel_embeddings)
        else:
            self.cached_mention_embeddings[inputs['data_partition']] =[rel_embeddings]
            
    def get_rel_embeddings(self, inputs):
        head_mention_embeddings, tail_mention_embeddings = self.get_entity_start_embeddings(inputs)
        head_etype_embs = self.etype_embeds(inputs["head_etype_id"])
        tail_etype_embs = self.etype_embeds(inputs["tail_etype_id"])
        rel_embeddings = torch.cat([head_mention_embeddings, head_etype_embs,
                                    tail_mention_embeddings, tail_etype_embs], dim=-1)
        return rel_embeddings

    def forward(self, inputs):
        if inputs["data_partition"] in self.cached_mention_embeddings:
            # import pdb; pdb.set_trace()
            rel_embeddings = self.cached_mention_embeddings[inputs["data_partition"]][inputs['idx']]
        else:
            rel_embeddings = self.get_rel_embeddings(inputs)
        rel_embeddings = self.fc(rel_embeddings)
        return rel_embeddings


class BertEntityConcat(BertBase):
    def __init__(self, config):
        super(BertEntityConcat, self).__init__(config)
        self.pooling_mode = config["entity_pooling"]
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(6*config["bert_hidden"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        sequence_output = self.get_sequence_output(inputs)
        head_mention_embeddings, tail_mention_embeddings = self.get_entity_pooling_embeddings(inputs, sequence_output=sequence_output)
        head_start_embeddings, tail_start_embeddings = self.get_entity_start_embeddings(inputs, sequence_output=sequence_output)
        head_end_embeddings, tail_end_embeddings = self.get_entity_end_embeddings(inputs, sequence_output=sequence_output)
        rel_embeddings = torch.cat(
            [head_start_embeddings, head_mention_embeddings, head_end_embeddings,
             tail_start_embeddings, tail_mention_embeddings, tail_end_embeddings], dim=-1)
        rel_embeddings = self.dropout(rel_embeddings)
        rel_embeddings = self.fc(rel_embeddings)
        return rel_embeddings


class BertEntityConcatEType(BertBase):
    def __init__(self, config):
        super(BertEntityConcatEType, self).__init__(config)
        self.pooling_mode = config["entity_pooling"]
        self.etype_embeds = EmbedLayer(
            num_embeddings=config["n_etype"],
            embedding_dim=config["etype_dim"],
            # dropout=config['word_dropout'],
        )
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(6*config["bert_hidden"] + 2*config["etype_dim"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        sequence_output = self.get_sequence_output(inputs)
        head_mention_embeddings, tail_mention_embeddings = self.get_entity_pooling_embeddings(
            inputs, sequence_output=sequence_output)
        head_start_embeddings, tail_start_embeddings = self.get_entity_start_embeddings(
            inputs, sequence_output=sequence_output)
        head_end_embeddings, tail_end_embeddings = self.get_entity_end_embeddings(
            inputs, sequence_output=sequence_output)
        head_etype_embs = self.etype_embeds(inputs["head_etype_id"])
        tail_etype_embs = self.etype_embeds(inputs["tail_etype_id"])
        rel_embeddings = torch.cat(
            [head_start_embeddings, head_mention_embeddings, head_end_embeddings, head_etype_embs,
             tail_start_embeddings, tail_mention_embeddings, tail_end_embeddings, tail_etype_embs], dim=-1)
        rel_embeddings = self.dropout(rel_embeddings)
        rel_embeddings = self.fc(rel_embeddings)
        return rel_embeddings


class BertMentionPooling(BertBase):
    def __init__(self, config):
        super(BertMentionPooling, self).__init__(config)
        self.pooling_mode = config["entity_pooling"]
        self.hidden_size = 2*config["bert_hidden"]
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(2*config["bert_hidden"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']
    
    def forward(self, inputs):
        head_embeddings, tail_embeddings = self.get_entity_pooling_embeddings(inputs)
        rel_embeddings = torch.cat([head_embeddings, tail_embeddings], dim=-1)
        rel_embeddings = self.fc(rel_embeddings)
        return rel_embeddings


class BertMentionPoolingEType(BertBase):
    def __init__(self, config):
        super(BertMentionPoolingEType, self).__init__(config)
        self.pooling_mode = config["entity_pooling"]
        self.etype_embeds = EmbedLayer(
            num_embeddings=config["n_etype"],
            embedding_dim=config["etype_dim"],
            # dropout=config['word_dropout'],
        )
        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(2*config["bert_hidden"] + 2*config["etype_dim"], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        head_mention_embeddings, tail_mention_embeddings = self.get_entity_pooling_embeddings(inputs)
        head_etype_embs = self.etype_embeds(inputs["head_etype_id"])
        tail_etype_embs = self.etype_embeds(inputs["tail_etype_id"])
        rel_embeddings = torch.cat([head_mention_embeddings, head_etype_embs,
                                    tail_mention_embeddings, tail_etype_embs], dim=-1)
        rel_embeddings = self.fc(rel_embeddings)
        return rel_embeddings


class EType(BaseModel):
    def __init__(self, config):
        super(EType, self).__init__(config)
        self.etype_embeds = EmbedLayer(
            num_embeddings=config["n_etype"],
            embedding_dim=config["etype_dim"],
            # dropout=config['word_dropout'],
        )
        self.hidden_size = config["etype_dim"]

    def forward(self, inputs):
        head_etype_embs = self.etype_embeds(inputs["head_etype_id"])
        tail_etype_embs = self.etype_embeds(inputs["tail_etype_id"])
        rel_embeddings = torch.cat([head_etype_embs, tail_etype_embs], dim=-1)
        rel_embeddings = self.linear_transform(rel_embeddings)
        return rel_embeddings

class EmbedBagEncoder(BaseModel):
    def __init__(self, config):
        super(EmbedBagEncoder, self).__init__()
        self.num_embeddings = config['vocab_size']
        self.embedding_dim = config['word_dim']
        self.embbag = nn.EmbeddingBag(
            num_embeddings=config['vocab_size'], 
            embedding_dim=config['word_dim'], 
            max_norm=config['max_norm'] if 'max_norm' in config else None, 
            norm_type=config['norm_type'] if 'norm_type' in config else 2.,
            scale_grad_by_freq=True, 
            mode=config['embbag_mode'])

        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['word_dim'], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']
        self.reset_parameters(config['pretrained_word_embs'], config['word_vocab'])

    def reset_parameters(self, pretrained=None, mapping=None):
        if pretrained is None:
            nn.init.orthogonal_(self.embbag.weight)
        else:
            print('Loading pre-trained word embeddings!')
            self.load_pretrained(pretrained, mapping)

    def load_pretrained(self, pretrained, voca):
        """
        Args:
            pretrained: (dict) keys are words, values are vectors
            mapping: (dict) keys are words, values are unique ids
            
        Returns: updates the embedding matrix with pre-trained embeddings
        """
        weights = nn.init.normal_(torch.empty((self.num_embeddings, self.embedding_dim)))
        for word, word_id in voca.stoi.items():
            word = voca.norm(word)
            if word in pretrained:
                weights[word_id, :] = torch.from_numpy(pretrained[word])
        self.embbag.weight.data = weights

    def forward(self, inputs, offsets=None):
        output = self.embbag(inputs["input_ids"], offsets)
        return self.fc(output)


class LSTMEncoder(BaseModel):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__(config)

        self.embed = EmbedLayer(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['word_dim'],
            dropout=config['word_dropout'],
            padding_idx=config['padding_idx'],
            scale_grad_by_freq=True,
            freeze=config['freeze_word_emb'],
            pretrained=config['pretrained_word_embs'],
            mapping=config['word_vocab']
        )
        
        self.max_position = config['max_position']
        self.pos_embs = EmbedLayer(
            num_embeddings=2 * self.max_position + 1,
            embedding_dim=config['position_dim']
        )

        n_lstm_layers = 2
        self.lstm = LSTMLayer(
            config['word_dim'], config['lstm_dim'],
            n_lstm_layers, bidirectional=False, dropout=config['dropout'])
        
        # Affine & Attention
        self.linear_fc = nn.Linear(config['lstm_dim'], config['fc_dim'])
        self.linear_position = nn.Linear(config['position_dim']*2, config['fc_dim'], bias=False)
        self.linear_last_hidden = nn.Linear(config['lstm_dim'], config['fc_dim'], bias=False)
        self.linear_attention = nn.Linear(config['fc_dim'], 1)
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        # Sort inputs based on lengths: longest first, required by torch.LSTM
        inputs['lengths'], sorted_idx = inputs['lengths'].sort(descending=True)
        inverse_indx = torch.argsort(sorted_idx)

        input_embs = self.embed(inputs['input_ids'][sorted_idx])
        # LSTM, Att
        output, last_hidden = self.lstm(input_embs, inputs['lengths'])
        # if self.direction == 2:
        #     last_hidden = torch.cat([last_hidden[-2], last_hidden[-1]], dim=1)
        #print (last_hidden.shape)
        last_hidden = last_hidden[-1, :, :]
        # Reverse back to original indices
        output = output[inverse_indx]
        last_hidden = last_hidden[inverse_indx]
        inputs['lengths'] = inputs['lengths'][inverse_indx]
        
        # [B, W, 2*lstm+2pos]
        # output = torch.cat([output, pos_head_embs, pos_tail_embs], dim=-1)

        # [B, W, 2*lstm+2pos] -> [B, W, hiddim]
        output = self.linear_fc(output)

        # pos embs
        pos_head_embs = self.pos_embs(inputs['pos_wrt_head'] + self.max_position)
        pos_tail_embs = self.pos_embs(inputs['pos_wrt_tail'] + self.max_position)
        position = self.linear_position(torch.cat([pos_head_embs, pos_tail_embs], dim=-1))

        # Query: [B, 2*lstm] -> [B, 1, hiddim]
        last_hidden = self.linear_last_hidden(last_hidden).unsqueeze(1)
        # [B, W, hiddim][B, 1, hiddim] -> [B, W, hiddim] matmul [hiddim] -> [B, W]
        att_scores = self.linear_attention(torch.tanh(output + position + last_hidden)).squeeze()
        att_scores = torch.where(
            inputs['masks'].byte(), att_scores,
            nn_utils.to_cuda(torch.empty(att_scores.shape).fill_(-1e7)))

        # [B, 1, W]
        #att_scores = torch.sigmoid(att_scores).unsqueeze(1)
        att_scores = torch.softmax(att_scores, dim=1).unsqueeze(1)

        # [B, 1, W] [B, W, D] -> [B, D]
        output = torch.matmul(att_scores, output).squeeze(1)
        return output


class PCNNEncoder(BaseModel):
    def __init__(self, config):
        super(PCNNEncoder, self).__init__(config)
        self.embed = EmbedLayer(
            num_embeddings=config['vocab_size'],
            embedding_dim=config['word_dim'],
            dropout=config['word_dropout'],
            padding_idx=config['padding_idx'],
            scale_grad_by_freq=True,
            freeze=config['freeze_word_emb'],
            pretrained=config['pretrained_word_embs'],
            mapping=config['word_vocab']
        )
        self.convs = nn.ModuleList(
            [nn.Conv1d(config['word_dim'], config['n_filters'], kernel_size=3, padding=1)
             for i in range(3)]
        )

        self.fc = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(config['n_filters'], config['fc_dim']),
            nn.ReLU()
        )
        self.hidden_size = config['fc_dim']

    def forward(self, inputs):
        input_embs = self.embed(inputs['input_ids'])
        input_embs = input_embs.permute(0, 2, 1)
        conv_output = []
        for i, conv in enumerate(self.convs):
            conved = conv(input_embs).permute(0, 2, 1)
            mask = (inputs['pcnn_mask'] == (i+1)).float().unsqueeze(dim=2)
            conved = conved * mask - (1 - mask) * 1e7
            # max pooling
            pooled = torch.tanh(torch.max(conved, 1)[0])
            conv_output.append(pooled)

        conv_output = (conv_output[0] + conv_output[1] + conv_output[2]) / 3
        output = self.fc(conv_output)
        # output = conv_output
        return output
