import os
import json


def find_mention_position_charbased(sentence, mention):
    start = sentence.find(mention)
    end = start + len(mention)
    return start, end


def find_mention_position(sentence, mention):
    char_idx = sentence.find(mention)
    start = sentence[:char_idx].count(' ')
    end = start + mention.count(' ') + 1
    return (char_idx, start, end)


def find_mentions_position(sentence, head_mention, tail_mention):
    (hp, ss, se) = find_mention_position(sentence, head_mention)
    (tp, os, oe) = find_mention_position(sentence, tail_mention)
    if head_mention in tail_mention and os <= ss and oe >= se:
        sub_sent = sentence[hp+1:]
        ss = sub_sent[:sub_sent.find(head_mention)].count(' ') + ss
        se = ss + head_mention.count(' ') + 1
    elif tail_mention in head_mention and ss <= os and se >= oe:
        sub_sent = sentence[tp+1:]
        os = sub_sent[:sub_sent.find(tail_mention)].count(' ') + os
        oe = os + tail_mention.count(' ') + 1
    assert(ss >= 0 and se >= 0 and os >= 0 and oe >= 0)
    return (ss, se), (os, oe)



def parse_line(line):
    # decode for python2
    # line = line.decode(errors='replace')
    (deppath, head_mention, tail_mention, enttypes, trigger, fname,
     sentence, postags, relation, head_pos, tail_pos,
     pos_templates, neg_templates) = line.split('\t')
    ss, se = head_pos.split('-')
    ss, se = int(ss), int(se)
    os, oe = tail_pos.split('-')
    os, oe = int(os), int(oe)
    pos_templates = [int(t) 
                    for t in pos_templates.strip().split(',')]
    neg_templates = [int(t) 
                    for t in neg_templates.strip().split(',')]

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe), pos_templates, neg_templates)


def parse_line_w_weight(line):
    # decode for python2
    # line = line.decode(errors='replace')
    (deppath, head_mention, tail_mention, enttypes, trigger, fname,
     sentence, postags, relation, head_pos, tail_pos,
     pos_templates, neg_templates) = line.split('\t')
    ss, se = head_pos.split('-')
    ss, se = int(ss), int(se)
    os, oe = tail_pos.split('-')
    os, oe = int(os), int(oe)
    pos_templates = [t.split(':')
                     for t in pos_templates.strip().split(',')]
    pos_templates = [float(t[1]) for t in pos_templates]
    neg_templates = []

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe), pos_templates, neg_templates)


def parse_line_w_weight_new(line):
    # decode for python2
    # line = line.decode(errors='replace')
    (deppath, head_mention, tail_mention, enttypes, trigger, fname,
     sentence, postags, relation, head_pos, tail_pos,
     pos_templates, neg_templates) = line.split('\t')
    ss, se = head_pos.split('-')
    ss, se = int(ss), int(se)
    os, oe = tail_pos.split('-')
    os, oe = int(os), int(oe)
    pos_templates = [float(t) for t in pos_templates.strip().split(',')]
    neg_templates = []

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe), pos_templates, neg_templates)


def parse_line_w_bags(line):
    # decode for python2
    # line = line.decode(errors='replace')
    (deppath, head_mention, tail_mention, enttypes, trigger, fname,
     sentence, postags, relation, head_pos, tail_pos,
     pos_templates, neg_templates) = line.split('\t')
    ss, se = head_pos.split('-')
    ss, se = int(ss), int(se)
    os, oe = tail_pos.split('-')
    os, oe = int(os), int(oe)
    pos_templates = [int(t) for t in pos_templates.strip().split(',')]
    neg_templates = [int(t) for t in neg_templates.strip().split(',')]

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe), pos_templates, neg_templates)

def parse_line_wo_position(line):
    deppath, head_mention, tail_mention, enttypes, trigger, fname, sentence, postags, relation = line.split('\t')
    hp = sentence.find(head_mention)
    ss = sentence[:hp].count(' ')
    se = ss + head_mention.count(' ') + 1
    tp = sentence.find(tail_mention)
    os = sentence[:tp].count(' ')
    oe = os + tail_mention.count(' ') + 1
    if head_mention in tail_mention and os <= ss and oe >= se:
        sub_sent = sentence[hp+1:]
        ss = sub_sent[:sub_sent.find(head_mention)].count(' ') + ss
        se = ss + head_mention.count(' ') + 1
    elif tail_mention in head_mention and ss <= os and se >= oe:
        sub_sent = sentence[tp+1:]
        os = sub_sent[:sub_sent.find(tail_mention)].count(' ') + os
        oe = os + tail_mention.count(' ') + 1
    assert(ss >= 0 and se >= 0 and os >= 0 and oe >= 0)
    # s_start, s_end = ss, se
    # o_start, o_end = os, oe
    # TODO fill templates ?????
    pos_templates = neg_templates = []

    return (deppath, head_mention, tail_mention, enttypes, trigger, 
            fname, sentence, postags, relation, (ss, se), (os, oe),
            pos_templates, [])


def parse_line_w_position2charpos(line):
    (deppath, head_mention, tail_mention, enttypes, trigger, fname, 
     sentence, postags, relation, head_pos, tail_pos) = line.split('\t')

    ss, se = find_mention_position_charbased(sentence, head_mention)
    os, oe = find_mention_position_charbased(sentence, tail_mention)
    pos_templates = neg_templates = []

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe),
            pos_templates, neg_templates)


def parse_line_w_bags_charpos(line):
    # decode for python2
    # line = line.decode(errors='replace')
    (deppath, head_mention, tail_mention, enttypes, trigger, fname,
     sentence, postags, relation, head_pos, tail_pos,
     pos_templates, neg_templates) = line.split('\t')
    ss, se = find_mention_position_charbased(sentence, head_mention)
    os, oe = find_mention_position_charbased(sentence, tail_mention)
    pos_templates = [int(t) for t in pos_templates.strip().split(',')]
    neg_templates = [int(t) for t in neg_templates.strip().split(',')]

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe), pos_templates, neg_templates)


def parse_line_w_position(line):
    (deppath, head_mention, tail_mention, enttypes, trigger, fname, 
     sentence, postags, relation, head_pos, tail_pos) = line.split('\t')
    ss, se = head_pos.split('-')
    ss, se = int(ss), int(se)
    os, oe = tail_pos.split('-')
    os, oe = int(os), int(oe)
    # TODO fill templates ?????
    pos_templates = neg_templates = []

    return (deppath, head_mention, tail_mention, enttypes, trigger,
            fname, sentence, postags, relation, (ss, se), (os, oe),
            pos_templates, [])


def read_tsv(path, parse_func='parse_line'):
    with open(path, 'rb') as f:
        data = []
        for (i, line) in enumerate(f):
            line = line.decode(errors='replace')
            (deppath, head_mention, tail_mention, enttypes, 
            trigger, fname, sentence, postags, 
            relation, head_pos, tail_pos, pos_templates, 
            neg_templates) = eval(parse_func)(line)
            head_etype, tail_etype = enttypes.split('-')
            item = {
                'deppath': deppath,
                'fname': fname,
                'sentence': sentence.strip(),
                'relation': relation.strip(),
                'postags': postags.strip(),
                'trigger': trigger.strip(),  # [8:].split('|'),
                'enttypes': enttypes,
                # head
                'head_mention': head_mention.strip(),
                'head_etype': head_etype,
                'head_pos': head_pos,
                # tail
                'tail_mention': tail_mention.strip(),
                'tail_etype': tail_etype,
                'tail_pos': tail_pos,
                'pos_template_ids': pos_templates,
                'neg_template_ids': neg_templates,
            }
            
            data.append(item)
            if len(data) > int(5e10):
                break
            if int(i+1) % 1e2:
                print(i, end='\r')
    print('Load <{} raw items'.format(len(data)))
    return data


def tokenize(sentence, head_pos, tail_pos, 
             mask_entity=False, 
             head_mask="#UNK#", tail_mask="#UNK#"):
    tokens = sentence.split()
    if head_pos[0] > tail_pos[0]:
        pos_min_s, pos_min_e = tail_pos
        pos_max_s, pos_max_e = head_pos
        rev = True
    else:
        pos_min_s, pos_min_e = head_pos
        pos_max_s, pos_max_e = tail_pos
        rev = False
    piece_0 = tokens[:pos_min_s]
    piece_1 = tokens[pos_min_e:pos_max_s]
    piece_2 = tokens[pos_max_e:]
    ent_0  = tokens[pos_min_s:pos_min_e]
    ent_1  = tokens[pos_max_s:pos_max_e]
    if mask_entity:
        if mask_entity == "add_marker":
            if rev:
                ent_0 = [tail_mask] + ent_0 + [tail_mask]
                ent_1 = [head_mask] + ent_1 + [head_mask]
                tail_pos = [tail_pos[0], tail_pos[0] + 2]
                head_pos = len(piece_0) + len(ent_0) + len(piece_1)
                head_pos = [head_pos, head_pos + 2]
            else:
                ent_0 = [head_mask] + ent_0 + [head_mask]
                ent_1 = [tail_mask] + ent_1 + [tail_mask]
                head_pos = [len(piece_0), len(piece_0) + 2]
                tail_pos = len(piece_0) + len(ent_0) + len(piece_1)
                tail_pos = [tail_pos, tail_pos+2]
        else:
            ent_0 = [head_mask]*len(ent_0)
            ent_1 = [tail_mask]*len(ent_1)

    tokens = piece_0 + ent_0 + piece_1 + ent_1 + piece_2

    return tokens, head_pos, tail_pos


def get_position_wrt_entities(max_position, head_pos, tail_pos, n_tokens):
    pos_wrt_head = [max(-max_position, i - head_pos[0])
                    for i in range(0, head_pos[0])] \
                    + [0] * (head_pos[1]-head_pos[0]) \
                    + [min(max_position, i - head_pos[1] + 1)
                        for i in range(head_pos[1], n_tokens)]
    pos_wrt_tail = [max(-max_position, i - tail_pos[0])
                    for i in range(0, tail_pos[0])] \
                    + [0]*(tail_pos[1]-tail_pos[0]) \
                    + [min(max_position, i - tail_pos[1] + 1)
                        for i in range(tail_pos[1], n_tokens)]
    return pos_wrt_head, pos_wrt_tail


def get_mask(max_position, head_pos, tail_pos, n_tokens, mask_entity=False):
    mask = []
    if head_pos[0] > tail_pos[0]:
        pos_min_s, pos_min_e = tail_pos
        pos_max_s, pos_max_e = head_pos
        rev = True
    else:
        pos_min_s, pos_min_e = head_pos
        pos_max_s, pos_max_e = tail_pos
    for i in range(n_tokens):
        if i < pos_min_e:
            if mask_entity and i >= pos_min_s:
                mask.append(0)
            else:
                mask.append(1)
        elif i < pos_max_s:
            mask.append(2)
        else:
            if mask_entity and i < pos_max_e:
                mask.append(0)
            else:
                mask.append(3)
    return mask


def read_raw_tsv(path, char_pos=False):
    from ure.loader.vocabulary import BRACKETS
    with open(path, 'rb') as f:
        data = []
        npasses = 0
        for (i, line) in enumerate(f):
            line = line.decode(errors='replace')
            columns = line.split('\t')
            n_cols = len(columns)
            if n_cols == 9:
                deppath, head_mention, tail_mention, enttypes, trigger, fname, sentence, postags, relation = line.split('\t')
                (s_start, s_end), (o_start, o_end) = find_mentions_position(
                    sentence, head_mention, tail_mention)
            elif n_cols == 11:
                deppath, head_mention, tail_mention, enttypes, trigger, fname, sentence, postags, relation, head_pos, tail_pos = line.split(
                    '\t')
                if char_pos:
                    (s_start, s_end) = find_mention_position_charbased(sentence, head_mention)
                    (o_start, o_end) = find_mention_position_charbased(sentence, tail_mention)
                else:
                    s_start, s_end = head_pos.split('-')
                    s_start, s_end = int(s_start), int(s_end)
                    o_start, o_end = tail_pos.split('-')
                    o_start, o_end = int(o_start), int(o_end)
            else:
                print("#cols={}".format(n_cols))
                npasses += 1
            head_etype, tail_etype = enttypes.split('-')
            sentence = sentence.strip()
            for k, v in BRACKETS.items():
                sentence = sentence.replace(k, v)
            item = {
                'deppath': deppath,
                'fname': fname,
                'sentence': sentence,
                'postags': postags.strip(),
                'head_mention': head_mention.strip(),
                'tail_mention': tail_mention.strip(),
                'enttypes': enttypes,
                'head_etype': head_etype,
                'tail_etype': tail_etype,
                'relation': relation.strip(),
                "head_pos": (s_start, s_end),
                "tail_pos": (o_start, o_end),
                'subj_offset': '{}-{}'.format(s_start, s_end),
                'obj_offset': '{}-{}'.format(o_start, o_end),
                'trigger': trigger.strip()  # [len('TRIGGER:'):].split('|')
            }
            data.append(item)
            if len(data) > int(5e10):
                break
                #pass
            if int(i+1) % 1e2:
                print(i, npasses, end='\r')
    print('Load <{} raw items'.format(len(data)))
    return data


def get_raw_predefined_relations(filepath, file_format='jsonl'):
    """ Get pre-defined relations from a jsonl file """
    if file_format == 'jsonl':
        relations = dict()
        with open(filepath, 'r') as rf:
                for _, l in enumerate(rf):
                    r = json.loads(l)
                    r['template'] = [{'weight': 1.0, 'template': r['template'], 'rid': r['relation']}]
                    relations[r['relation']] = r
    elif file_format == 'json':
        with open(filepath, 'r') as rf:
            relations = json.load(rf)
            for k, r in relations.items():
                r['template'] = [{'weight': 1.0, 'template': r['template'], 'rid': r['relation']}]
    print("#relations = {}".format(len(relations)))
    return relations


def get_from_json(filepath):
    """ Get pre-defined relations/all selected templates from a json file """
    with open(filepath, 'r') as rf:
        data = json.load(rf)
    return data


if __name__ == "__main__":
    data = read_tsv('data/nyt/dev.tsv')
    print(data[0])
