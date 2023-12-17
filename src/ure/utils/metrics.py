from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from collections import Counter

import operator


def compare_metrics(met_name, current, best):
    if met_name in ["bcubed", "F1"]:
        return current["F1"] > best["F1"]
    elif met_name == "vmeasure":
        return current["V"] > best["V"]
    elif met_name == "ari":
        return current["ARI"] > best["ARI"]
    elif met_name == "accuracy":
        return current["ACC"] > best["ACC"]
    elif met_name == "micro_f1":
        return current["micro_f1"] > best["micro_f1"]
    elif met_name == "macro_f1":
        return current["macro_f1"] > best["macro_f1"]
    else:
        raise NotImplementedError("`{}` not implemented!".format(met_name))


def bcubed(gold, pred, na_id=-1):

    def correctness(gold, pred, na_id=0):
        # remove NA
        gp = [(x, y) for x, y in zip(gold, pred) if x != na_id]
        go = [x for x, _ in gp]
        pr = [y for _, y in gp]

        # compute 'correctness'
        l = len(pr)
        assert(len(go) == l)
        go = torch.IntTensor(go)
        pr = torch.IntTensor(pr)
        gc = ((go.unsqueeze(0) - go.unsqueeze(1)) == 0).int()
        pc = ((pr.unsqueeze(0) - pr.unsqueeze(1)) == 0).int()
        # print('gc', gc.shape)
        # print('pc', pc.shape)
        c = gc * pc
        return c, gc, pc


    def precision(c, gc, pc):
        pcsum = pc.sum(1)
        total = torch.where(pcsum > 0, pcsum.float(), torch.ones(pcsum.shape))
        return ((c.sum(1).float() / total).sum() / gc.shape[0]).item()


    def recall(c, gc, pc):
        gcsum = gc.sum(1)
        total = torch.where(gcsum > 0, gcsum.float(), torch.ones(gcsum.shape))
        return ((c.sum(1).float() / total).sum() / pc.shape[0]).item()

    c, gc, pc = correctness(gold, pred, na_id)
    prec = precision(c, gc, pc)
    rec = recall(c, gc, pc)
    return {'P': prec, 'R': rec, 'F1': 2 * (prec * rec) / (prec + rec)}


def vmeasure(gold, pred, na_id=-1):
    homo = homogeneity_score(gold, pred)
    comp = completeness_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    return {'H': homo, 'C': comp, 'V': v_m}


def ari(gold, pred, na_id=-1):
    ari = adjusted_rand_score(gold, pred)
    return {'ARI': ari}


def accuracy(gold, pred, na_id=-1):
    return {'ACC': accuracy_score(gold, pred)}

def micro_f1(gold, pred, na_id=-1):
    # return {"micro_f1": f1_score(gold, pred, average='micro', labels=range(1, 42))}
    # prec_micro, recall_micro, f1_micro = score(gold, pred, na_id)
    labels = list(set(gold))
    if na_id in labels: labels.remove(na_id)

    # print(f1_score(gold, pred, average='micro', labels=labels))
    (p, r, f, _) = precision_recall_fscore_support(
        gold, pred, beta=1.0, average='micro', labels=labels)
    return {"micro_p": p, "micro_r": r, "micro_f1": f}


def macro_f1(gold, pred, na_id=-1):
    labels = list(set(gold))
    if na_id in labels:
        labels.remove(na_id)
    (p, r, f, _) = precision_recall_fscore_support(
        gold, pred, beta=1.0, average='macro', labels=labels)
    return {"macro_p": p, "macro_r": r, "macro_f1": f}


def score(key, prediction, verbose=False, na_id=-1):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == na_id and guess == na_id:
            pass
        elif gold == na_id and guess != na_id:
            guessed_by_relation[guess] += 1
        elif gold != na_id and guess == na_id:
            gold_by_relation[gold] += 1
        elif gold != na_id and guess != na_id:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro


def check_with_bcubed_lib(gold, pred):
    import bcubed
    ldict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(gold)])
    cdict = dict([('item{}'.format(i), set([k])) for i, k in enumerate(pred)])

    precision = bcubed.precision(cdict, ldict)
    recall = bcubed.recall(cdict, ldict)
    fscore = bcubed.fscore(precision, recall)

    print('P={} R={} F1={}'.format(precision, recall, fscore))


if __name__ == '__main__':
    gold = [0, 0, 0, 0, 0, 1, 1, 2, 1, 3, 4, 1, 1, 1]
    pred = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

    print(bcubed(gold, pred, na_id=-1), 'should be 0.69')

    check_with_bcubed_lib(gold, pred)
    homo = homogeneity_score(gold, pred)
    v_m = v_measure_score(gold, pred)
    ari = adjusted_rand_score(gold, pred)
    print(homo, v_m, ari)
