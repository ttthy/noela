import torch
import torch.nn as nn
from ure.utils.nn_utils import to_cuda


class EmbedLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, dropout=0.0,
                 padding_idx=None, scale_grad_by_freq=True,
                 freeze=False, pretrained=None, mapping=None):
        """
        Args:
            num_embeddings: (tensor) number of unique items
            embedding_dim: (int) dimensionality of vectors
            dropout: (float) dropout rate
            trainable: (bool) train or not
            pretrained: (dict) pretrained embeddings
            mapping: (dict) mapping of items to unique ids
        """
        super(EmbedLayer, self).__init__()
        self.padding_idx = padding_idx
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.freeze = freeze
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=padding_idx,
                                      scale_grad_by_freq=scale_grad_by_freq)

        self.dropout = nn.Dropout3d(dropout)
        self.reset_parameters(pretrained, mapping)

    def reset_parameters(self, pretrained=None, mapping=None):
        if pretrained is not None:
            print('Loading pre-trained word embeddings!')
            self.load_pretrained(pretrained, mapping)
            self.embedding.weight.requires_grad = not self.freeze
        else:
            nn.init.orthogonal_(self.embedding.weight.data)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

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
        self.embedding.weight.data = weights

    def forward(self, xs):
        """
        Args:
            xs: (tensor) batchsize x word_ids

        Returns: (tensor) batchsize x word_ids x dimensionality
        """
        # max_len = xs.shape[1]
        # if self.dropout > 0 and self.training:
        #     probs = to_cuda(torch.empty(max_len).uniform_(0, 1))
        #     xs = torch.where(probs > self.dropout, xs, to_cuda(torch.empty_like(xs).fill_(self.padding_idx)))
        embeds = self.embedding(xs)
        embeds = self.dropout(embeds)

        return embeds


class LSTMLayer(nn.Module):
    def __init__(self, input_size, rnn_size, num_layers, bidirectional, dropout):
        """
        Wrapper for LSTM encoder
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        Returns: outputs, last_outputs
        - **outputs** of shape `(batch, seq_len, hidden_size)`:
          tensor containing the output features `(h_t)`
          from the last layer of the LSTM, for each t.
        - **last_outputs** of shape `(batch, hidden_size)`:
          tensor containing the last output features
          from the last layer of the LSTM, for each t=seq_len.
        """
        super(LSTMLayer, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=rnn_size // 2 if bidirectional else rnn_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # forget gate bias = 1
        # # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/3
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        # define output feature size
        self.feature_size = rnn_size


    def forward(self, embeds, lengths):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            embs (): word embeddings
            lengths (): the lengths of each sentence
        Returns: the logits for each class
        """
        # pack_padded_sequence so that
        # padded items in the sequence won"t be shown to the LSTM
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths, batch_first=True, enforce_sorted=False
        )

        lstm_outputs, (last_hidden, _) = self.lstm(packed_inputs)

        # undo the packing operation
        # padding with inf to avoid its effect to attention or other layers
        # lstm_outputs : [batch_size, seq_len, 2*lstm_dim]
        lstm_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_outputs, batch_first=True, padding_value=0
        )
        # Last hidden output from nn.LSTM contains (last_fw t=seq_len, last_bw t=0)
        
        # Last forward, t=seq_len
        # last_hidden_h = extract_output_from_timestep(
        #     lstm_outputs, lengths - torch.ones_like(lengths), lstm_outputs.device, time_dimension=1)
        
        # Last backward t=0
        # last_hidden_h = extract_output_from_timestep(
        #     lstm_outputs, torch.zeros_like(lengths), embeds.device, time_dimension=1)
        
        lstm_outputs = lstm_outputs.contiguous()
        
        return lstm_outputs, last_hidden


class Highway(nn.Module):
    """Highway network"""

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = nn.functional.sigmoid(self.fc1(x))
        return torch.mul(t, nn.functional.relu(self.fc2(x))) + torch.mul(1-t, x)
