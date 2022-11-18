import json
import re


def _get_tag_value(line):
    line = re.sub(r"\s?\{[^}]+\}", "", line)  # Removes text from inside curly brackets.
    line = line[:line.find('!')]
    col_index = line.find(':')

    if col_index == -1:
        return "", ""

    tag = line[:col_index]
    quotes = re.findall(r'\"(.+?)\"', line)

    if len(quotes) == 0:
        value = line[col_index + 1:].strip().replace("_", " ")
    else:
        value = quotes[0].strip().replace("_", " ")

    return tag, value


def _dfs(c, kids, mark):
    mark.add(c)

    for kid in kids[c]:
        if kid not in mark:
            _dfs(kid, kids, mark)


def _remove_duplicate_values(dictionary):
    return {k: list(set(v)) for k, v in dictionary.items()}


def _merge_dicts(*dicts, single_ints=False):
    d = {}
    for dictionary in dicts:
        for key in dict:
            if single_ints:
                d[key] = dictionary[key]
            elif isinstance(dictionary[key], list):
                try:
                    d[key] += dictionary[key]
                except KeyError:
                    d[key] = dictionary[key]
            else:
                try:
                    d[key].append(dictionary[key])
                except KeyError:
                    d[key] = [dictionary[key]]

    return d


class OntCore:

    def __init__(self, mark, names, def_text, parents, kids, real_id, start_idx,
                 concepts):
        self.names = {c: names[c] for c in mark}
        self.def_text = {c: def_text[c] for c in mark if c in def_text}
        self.parents = {c: parents[c] for c in mark}
        self.kids = {c: kids[c] for c in mark}
        self.real_id = real_id

        for c in self.parents:
            self.parents[c] = [p for p in parents[c] if p in mark]

        self.concepts = [c for c in sorted(self.names.keys())
                         if c not in concepts]
        self.concept2id = dict(zip(self.concepts,
                                   range(start_idx,
                                         start_idx + len(self.concepts))))

        self.name2conceptid = {}

        for c in self.concepts:
            for name in self.names[c]:
                normalized_name = name.strip().lower()
                self.name2conceptid[normalized_name] = self.concept2id[c]


class Ontology:

    def __init__(self, ont_param_json):

        self.names = {}
        self.concepts = []
        self.concept2id = {}
        self.name2conceptid = {}
        self.parents = {}
        self.kids = {}
        self.real_id = {}
        self.root_id = []

        with open(ont_param_json) as f:
            self.params = json.load(f)

        start_idx = 0

        for ont in self.params['ontologies']:

            obo_file = open(ont['obo_path'], encoding='utf-8', errors='ignore')
            data = self._load_obofile(obo_file, ont['root_id'], start_idx)

            if ont['main'] == 1:
                self.idx_thresh = len(data.concepts)

            self.root_id.append(ont['root_id'])
            self.concepts += data.concepts
            self.names = _merge_dicts(self.names, data.names)
            self.names = _merge_dicts(self.names, data.names)
            self.concept2id = _merge_dicts(self.concept2id, data.concept2id,
                                           single_ints=True)
            self.name2conceptid = _merge_dicts(self.name2conceptid,
                                               data.name2conceptid,
                                               single_ints=True)
            self.parents = _merge_dicts(self.parents, data.parents)
            self.kids = _merge_dicts(self.kids, data.kids)
            self.real_id = _merge_dicts(self.real_id, data.real_id,
                                        single_ints=True)

            start_idx = len(self.concepts)

        self.ancestor_weight = {}
        self.samples = []

        for c in self.concepts:
            self._update_ancestry_sparse(c)

        self.sparse_ancestors = []
        self.sparse_ancestors_values = []

        for cid in self.ancestor_weight:
            self.sparse_ancestors += [[cid, ancid]
                                      for ancid in self.ancestor_weight[cid]]
            self.sparse_ancestors_values += [self.ancestor_weight[cid][ancid]
                                             for ancid in
                                             self.ancestor_weight[cid]]

        self.names = _remove_duplicate_values(self.names)
        self.n_concepts = len(self.concepts)

    def _load_obofile(self, obofile, root, start_idx):
        names = {}
        def_text = {}
        kids = {}
        parents = {}
        real_id = {}

        while True:
            line = obofile.readline()

            if line == "":
                break
            tag, value = _get_tag_value(line)

            if tag == "":
                continue

            if tag == "id":
                concept_id = value
                parents[concept_id] = []
                kids[concept_id] = []
                names[concept_id] = []
                def_text[concept_id] = []
                real_id[concept_id] = concept_id

            if tag == "def":
                def_text[concept_id].append(value)

            if tag == "name" or tag == "synonym":
                if re.match(r"\\", value) or re.match('[^\x00-\x7F]+', value):
                    pass
                else:
                    names[concept_id].append(value)

            if tag == "alt_id":
                real_id[value] = concept_id

        obofile.seek(0)

        while True:
            line = obofile.readline()

            if line == "":
                break

            tag, value = _get_tag_value(line)

            if tag == "":
                continue

            if tag == "id":
                concept_id = value
                last_id_unique = True

            if tag == "is_a" and last_id_unique:
                parent_id = real_id[value]
                kids[parent_id].append(concept_id)
                parents[concept_id].append(parent_id)

        mark = set()
        _dfs(root, kids, mark)

        return OntCore(mark, names, def_text, parents, kids, real_id, start_idx,
                       self.concepts)

    def _update_ancestry_sparse(self, con):
        cid = self.concept2id[con]
        if cid in self.ancestor_weight:
            return self.ancestor_weight[cid].keys()
        self.ancestor_weight[cid] = {cid: 1.0}
        num_parents = len(self.parents[con])
        for p in self.parents[con]:
            tmp_ancestors = self._update_ancestry_sparse(p)
            pid = self.concept2id[p]
            for ancestor in tmp_ancestors:
                if ancestor not in self.ancestor_weight[cid]:
                    self.ancestor_weight[cid][ancestor] = 0.0
                self.ancestor_weight[cid][ancestor] += (
                        self.ancestor_weight[pid][ancestor] / num_parents)

        return self.ancestor_weight[cid].keys()
