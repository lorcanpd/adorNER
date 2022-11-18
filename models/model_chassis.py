import json
import os
import pickle
import re
from collections import namedtuple
from random import uniform, betavariate, randint, choice

import numpy as np
import regex as re2
import tensorflow as tf
import tensorflow_hub as hub

from models.multi_scale_self_attention import MSSACore
from models.ncr_core import NCRCore
from utils.utils import phrase2vec, phrase2vecUSE


class ExplicitChassis():

    def __init__(self, config, ont):
        self.config = config
        self.ont = ont
        if not config.model_type == "use":
            self.elmo = hub.load(config.hub_dir).signatures['tokens']

            if config.model_type == 'mssa':
                self.cores = [MSSACore(config, ont)
                                  for _ in range(config.n_ensembles)]
            else:
                self.cores = [NCRCore(config, ont) for _ in
                                  range(config.n_ensembles)]

            inputs = tf.keras.Input(shape=(config.max_sequence_length,
                                           512))

        if config.model_type != "mssa":
            outputs = [core(inputs) for core in self.cores]
        else:
            outputs = [core(inputs, tf.convert_to_tensor(1, tf.float32))
                       for core in self.cores]


        if config.n_ensembles == 1:
            merged_outputs = outputs[0]
        else:
            merged_outputs = tf.keras.layers.Average()(outputs)

        self.ensembled_model = tf.keras.Model(inputs=inputs,
                                              outputs=merged_outputs)

    @classmethod
    def loadfromfile(cls, param_dir):
        ont = pickle.load(open(param_dir + '/ont.pickle', "rb"))

        class Config(object):
            def __init__(self, d):
                self.__dict__ = d

        config = Config(json.load(open(param_dir + '/config.json', 'r')))
        # Post norm models were trained before pre_norm added - this should fix
        # for now. Change for release?
        try:
            config.pre_norm
        except AttributeError:
            setattr(config, "pre_norm", False)
        try:
            config.scales
        except AttributeError:
            setattr(config, "scales", False)

        model = cls(config, ont)
        model.ensembled_model.load_weights(param_dir + '/model_weights.h5')
        return model

    @classmethod
    def safeloadfromjson(cls, param_dir, word_model_file):
        ont_dict = json.load(open(param_dir + '/onto.json', 'r'))
        ont = namedtuple('Struct', ont_dict.keys())(*ont_dict.values())

        class Config(object):
            def __init__(self, d):
                self.__dict__ = d

        config = Config(json.load(open(param_dir + '/config.json', 'r')))

        model = cls(config, ont, word_model_file)
        model.ensembled_model.load_weights(param_dir + '/model_weights.h5')
        return model

    def save_weights(self, param_dir):
        self.ensembled_model.save_weights(
            param_dir + '/model_weights.h5', save_format='h5')

    def get_match(self, query, count=1):
        batch_size = 512

        was_string = False
        if isinstance(query, str):
            was_string = True
            query = [query.encode()]
        elif isinstance(query, bytes):
            was_string = True
            query = [query]
        elif isinstance(query, list):
            query = [x.encode() for x in query]

        if not self.config.model_type == "use":
            seq, seq_len = phrase2vec(self.elmo, query, self.config)
        else:
            seq = phrase2vecUSE(self.use, query)

        result_probs = []
        for head in range(0, len(query), batch_size):
            query_subset = seq[head:head + batch_size]
            result_probs.append(tf.nn.softmax(self.ensembled_model(query_subset),
                                              axis=-1).numpy())
        res_query = np.concatenate(result_probs)

        results = []
        indecies_query = np.argpartition(res_query, -count, axis=-1)[:, -count:]
        for s in range(len(query)):
            tmp_indecies_query = indecies_query[
                s, np.argsort(-res_query[s, indecies_query[s]])
            ]
            tmp_res = []
            for i in tmp_indecies_query:
                if i >= self.ont.idx_thresh:
                    tmp_res.append(('None', res_query[s, i]))
                else:
                    tmp_res.append((self.ont.concepts[i], res_query[s, i]))
                if len(tmp_res) >= count:
                    break
            results.append(tmp_res)
        if was_string:
            return results[0]
        return results

    # TODO: Fix match entailment code - seems buggy!
    def annotate_text(self, text, threshold=0.8):
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

        def nonconsumesplitconcat(pat, content):
            outer = pat.split(content)
            inner = pat.findall(content) + ['']
            return [pair[0] + pair[1] for pair in zip(outer, inner)]

        pat = re2.compile(r"(?<!\w\.\w.)(?![^(]*\))(?<![A-Z]\.)(?!\s+[a-z\[,])"
                          r"(?!\s+[0-9]{4})(?<![\w]{1}\.[\w]{1})"
                          r"(?<=\.|\?|\!\:)\s+|\p{Cc}+|\p{Cf}+")
        sentences = nonconsumesplitconcat(pat, text)
        pat2 = re.compile('[\\\\/\r\n\t-]')
        text_replaced = [pat2.sub(' ', line) for line in sentences]
        pat3 = re.compile('[^\w\d ]')
        chunks_large = [chunk for chunks in text_replaced for chunk in
                        nonconsumesplitconcat(pat3, chunks)]
        candidates = []
        candidates_info = []
        total_chars = 0

        for c, chunk in enumerate(chunks_large):
            pat4 = re.compile(' ')
            tokens = nonconsumesplitconcat(pat4, chunk)
            chunk_chars = 0
            for i, w in enumerate(tokens):
                for ra in range(7):
                    if i + ra >= len(tokens) or len(tokens[i + ra]) == 0:
                        break
                    if ra > 0:
                        if phrase.startswith(' '):
                            continue
                        elif phrase.endswith(' '):
                            phrase += tokens[i + ra]
                        else:
                            phrase += " " + tokens[i + ra]
                    else:
                        phrase = tokens[i + ra]
                    cand_phrase = phrase
                    if len(cand_phrase) > 0 and cand_phrase != ' ':
                        candidates.append(cand_phrase)
                        location = total_chars + chunk_chars
                        if phrase.startswith(' '):
                            candidates_info.append(
                                (location + 1, location + len(phrase) - 1, c))
                        else:
                            candidates_info.append(
                                (location, location + len(phrase) - 1, c))
                chunk_chars += len(w)  # + 1
            total_chars += chunk_chars
        cand_len = len(candidates)

        def batch_match(candidates, batch_size):
            matches = []
            for _ in range(int(np.ceil(len(candidates)/batch_size))):
                try:
                    batch = candidates[0:batch_size]
                    del candidates[0:batch_size]
                except IndexError:
                    batch = candidates
                matches.append([x[0] for x in self.get_match(batch, 1)])
            return [item for sublist in matches for item in sublist]

        matches = batch_match(candidates, 256)
        filtered = {}
        for i in range(cand_len):
            if (matches[i][0] != self.ont.root_id and matches[i][
                0] != "None" and
                    matches[i][1] > threshold):

                if candidates_info[i][2] not in filtered:
                    filtered[candidates_info[i][2]] = []

                filtered[candidates_info[i][2]].append((
                    candidates_info[i][0],
                    candidates_info[i][1],
                    matches[i][0],
                    matches[i][1]))
        final = []
        for c in filtered:
            tmp_final = []
            for x in filtered[c]:
                bad = False
                for y in filtered[c]:
                    if (x[0] <= y[0] and x[1] >= y[1] and x[2] == y[2]
                            and (x is not y) and x[3] < y[3]):
                        bad = True
                        break
                if not bad:
                    tmp_final.append(x)
            cands = sorted(tmp_final, key=lambda x: x[0] - x[1])
            tmp_final = []
            for x in cands:
                conflict = False
                for y in tmp_final:
                    if x[1] > y[0] and x[0] < y[1]:
                        conflict = True
                        break
                if not conflict:
                    tmp_final.append(x)
            final += tmp_final
        return final
