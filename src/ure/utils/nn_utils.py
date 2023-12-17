import argparse
import inspect
import math
import numbers
import os
import random
import re
import subprocess
from collections import OrderedDict
from itertools import chain

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable


################################## utils for GPU ############################
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_gpu_memory():
    t = torch.cuda.get_device_properties(0).total_memory*1e-9
    c = torch.cuda.memory_cached(0)*1e-9
    a = torch.cuda.memory_allocated(0)*1e-9
    f = c-a  # free inside cache
    return 'Total={:.5f}, Used={:.5f}, Allocate={:.5f} > Free={:.5f} GB'.format(t, c, a, f)


def current_memory_usage():
    '''Returns current memory usage (in MB) of a current process'''
    out = subprocess.Popen(['ps', '-p', str(os.getpid()), '-o', 'rss'],
                           stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    mem = float(out[1].strip()) / 1024
    return mem


def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


################################## utils for training ############################

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        s = s.split(",")
        method = s[0]
        optim_params = {}
        for x in s[1:]:
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adabound":
        optim_fn = AdaBound
    elif method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = optim.Adam
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception("Unknown optimization method: '%s'" % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception("Unexpected parameters: expected '%s', got '%s'" % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


################################## utils for model ############################
def group_model_params(model):
    # Number of trainable parameters
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    # Not apply weight decay on bias
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01
        },
        {
            "params": [
                p
                for n, p in param_optimizer
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        },
    ]
    return optimizer_grouped_parameters


def max_pooling(vec, axis=1):
    # return the argmax as a python int
    max_features, _ = torch.max(vec, axis)
    return max_features


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data.item()

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = torch.autograd.Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask


def extract_output_from_timestep(features, indices, device, time_dimension=1):
    """ Indexing features from a 2D tensor
    
    Arguments:
        features {torch.tensor} -- [batch_size, seq_len, feature_dim]
        indices {torch.tensor} -- [batch_size]
    
    Returns:
        {torch.tensor} -- [batch_size, feature_dim]


    >>> timeit.timeit(arange, number=1000000)
    9.636487066978589
    >>> timeit.timeit(gather, number=1000000)
    11.31074752798304

    idx_exp = indices.view(-1, 1).expand(len(indices), features.size(2))
    idx_exp = idx_exp.unsqueeze(time_dimension)
    # idx_exp = idx_exp.to(device)
    idx_outputs = features.gather(time_dimension, torch.autograd.Variable(idx_exp)).squeeze(time_dimension)
    return idx_outputs

    Arange is faster but the time_dimension is fixed
    """
    return features[torch.arange(features.size(0)).to(device), indices]


def extract_output_from_offset(features, offsets, device, time_dimension=1):
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
    idx = offset_to_mask(offsets, batch_size, seq_len, device, features.dtype)
    # [batch_size, dim]
    output = torch.bmm(idx.unsqueeze(time_dimension), features).squeeze(time_dimension)
    # [batch_size]
    idx = torch.sum(idx, dim=time_dimension).unsqueeze(-1)
    # [batch_size, dim]
    output = torch.div(output, idx)

    return output


def offset_to_mask(offsets, batch_size, seq_len, device, dtype):
    # Create word position ranges
    # [seq_len]
    ranges = torch.arange(seq_len).long()
    # [batch_size, seq_len]
    ranges = ranges.unsqueeze(0).expand(batch_size, seq_len).to(device)
    # Filter out irrelevant positions
    # [batch_size, seq_len]
    idx = (torch.ge(ranges, offsets[0].unsqueeze(-1)) &
           torch.le(ranges, offsets[1].unsqueeze(-1))).type(dtype).to(device)

    return idx


def length_to_mask(length, dtype, max_len=None):
    """Length to Mask
    
    Arguments:
        length {torch.tensor} -- Sequence length [batch_size]
    
    Keyword Arguments:
        max_len {int} --  (default: {None})
        dtype {torch.dtype} -- torch.long (default: {None})
    
    Returns:
        mask {torch.tensor} -- [batch_size, max_len] mask out 0
    """
    assert len(length.size()) == 1, "Length should be 1D!"
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, dtype=length.dtype, device=length.device).expand(
        length.size(0), max_len) < length.unsqueeze(-1)
    return mask.type(dtype)


def dropout_by_replace_mask(input_ids, unk_id):
    # probabilities
    probs = torch.empty(input_ids.size()).uniform_(0, 1)
    # applying word dropout
    input_ids = torch.where(
        probs > 0.02, input_ids,
        torch.empty(input_ids.size()).fill_(unk_id).long().to(input_ids.device))
    return input_ids


################################# batching ############################
def make_equal_len(lists, fill_in=0, to_right=True):
    lens = [len(l) for l in lists]
    max_len = max(1, max(lens))
    if to_right:
        if fill_in is None:
            eq_lists = [l + [l[-1].copy() if isinstance(l[-1], list) else l[-1]] * (max_len - len(l)) for l in lists]
        else:
            eq_lists = [l + [fill_in] * (max_len - len(l)) for l in lists]
        mask = [[1.] * l + [0.] * (max_len - l) for l in lens]
    else:
        if fill_in is None:
            eq_lists = [[l[0].copy() if isinstance(l[0], list) else l[0]] * (max_len - len(l)) + l for l in lists]
        else:
            eq_lists = [[fill_in] * (max_len - len(l)) + l for l in lists]
        mask = [[0.] * (max_len - l) + [1.] * l for l in lens]

    return eq_lists, mask
    

def flatten_3Dto2D_lists(lists):
    # From 3D to 2D
    return [x for l in lists for x in l]


def to_long_tensor(lists):
    return Variable(to_cuda(torch.LongTensor(lists)), requires_grad=False)


def to_float_tensor(lists):
    return Variable(to_cuda(torch.FloatTensor(lists)), requires_grad=False)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, numbers.Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)
# Compute log sum exp in a numerically stable way for the forward algorithm


# def log_sum_exp(vec, m_size):
#     """
#     calculate log of exp sum
#     args:
#         vec (batch_size, vanishing_dim, hidden_dim) : input tensor
#         m_size : hidden_dim
#     return:
#         batch_size, hidden_dim
#     """
#     _, idx = torch.max(vec, 1)  # B * 1 * M
#     max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
#     return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)),
#                                                             1)).view(-1, m_size)  # B * M
