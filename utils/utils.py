import json
import pickle
import re

import numpy as np
import tensorflow as tf


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def tokenize(phrase, max_seq_len):
    pattern = re.compile(b'[\W_]')
    tmp = pattern.sub(b' ', phrase).lower().strip().split()
    return [b"INT" if w.isdigit() else
            (b"FLOAT" if is_number(w) else w) for w in tmp][:max_seq_len]


def tokenize_str(phrase):
    pattern = re.compile('[\W_]')
    tmp = pattern.sub(' ', phrase).lower().strip().split()
    return ["INT" if w.isdigit() else
            ("FLOAT" if is_number(w) else w) for w in tmp]


def pad(tokens, max_len):
    return tokens + [b""] * (max_len - len(tokens))


def pad_tensor(tensor, emb_dim, max_seq_len):
    add_pad = max_seq_len - tensor.shape[0]
    return tf.concat((tensor, tf.zeros(shape=(add_pad, emb_dim))), axis=0)


def phrase2vec(elmo, phrase_list, config):
    tokens = [tokenize(phrase, config.max_sequence_length)
              for phrase in phrase_list]
    lens = tf.convert_to_tensor([len(p) for p in tokens], tf.int32)
    max_len = max(lens.numpy())
    tokens = tf.convert_to_tensor([pad(p, max_len) for p in tokens], tf.string)
    embeddings = elmo(tokens=tokens, sequence_len=lens)['word_emb']
    mask = tf.sequence_mask(lens, max_len, dtype=tf.float32)
    embeddings = tf.multiply(embeddings, tf.expand_dims(mask, -1))
    embeddings = tf.convert_to_tensor([pad_tensor(emb, emb.shape[-1],
                                                  config.max_sequence_length)
                                       for emb in embeddings],
                                      dtype=tf.float32)
    return embeddings, lens


# def phrase2vecUSE(use, phrase_list):
#     return use(phrase_list)


def exp_data_gen_fn(data_path, label_path):
    """
    A generator function to stream data and ground truth labels for model
    training.
    ...
    Attributes
    ----------
    data_path : str
        path to the txt file containing all the natural language names of the
        ontology terms
    label_path : str
        path to the txt file containing all the concept id labels corresponding
        to the natural language names
    """
    with open(data_path, 'r') as raw_data, \
            open(label_path, 'r') as labels:
        for phrase, label in zip(raw_data, labels):
            yield phrase.strip(), int(label)


def exp_input_fn(config, num_phrases, data_path, label_path):
    """
    Wrapper function that converts the training data generator into a
    tf.data.Dataset for streaming training data from disk.
    ...
    Attributes
    ----------
    config :
        configuration file taken from the --params .json file
    num_phrases :
        the total number of natural language phrases in the training data
    data_path : str
        path to the txt file containing all the natural language names of the
        ontology terms
    label_path : str
        path to the txt file containing all the concept id labels corresponding
        to the natural language names

    Returns
    -------
     : tf.data.Dataset
        a tensorflow dataset object that is used to iteratively stream training
        examples from disk
    """
    types = (tf.string, tf.int32)
    dataset = tf.data.Dataset.from_generator(
        generator=exp_data_gen_fn,
        output_types=types,
        args=(data_path, label_path)
    )
    if config.shuffle and config.repeat:
        dataset = (dataset.shuffle(buffer_size=num_phrases)
                   .batch(batch_size=config.batch_size)
                   .repeat(config.epochs))
    elif config.shuffle and not config.repeat:
        dataset = (dataset.shuffle(buffer_size=num_phrases)
                   .batch(batch_size=config.batch_size))
    elif config.repeat and not config.shuffle:
        dataset = (dataset.batch(batch_size=config.batch_size)
                   .repeat(config.epochs))
    else:
        dataset = dataset.batch(batch_size=config.batch_size)
    return dataset


def save_ont_and_args(ont, args, param_dir):
    """
    Saves the ontology used to train the model and the arguments supplied at
    initialisation. These are used when a model is reloaded after training.
    """
    pickle.dump(ont, open(param_dir + '/ont.pickle', "wb"))
    with open(param_dir + '/config.json', 'w') as fp:
        json.dump(vars(args), fp)


def sample_negatives_from_file(file_path, count, max_seq_len):
    """
    Function for extracting negative (non-ontology) natural language text
    samples from a text file. This is used to provide greater training samples
    for a model from outside the domain of the training ontology.
    ...
    Attributes
    ----------
    file_path : str
        path to the txt file
    count : int
        number of text samples to generate
    max_seq_len : int
        the maximum number of tokens in the extracted text sample sequences

    Returns
    -------
    negative_samples : [str]
        a list containing strings of negative text samples
    """
    max_text_size = 10 * 1000 * 1000
    with open(file_path, errors='replace') as f:
        text = f.read()[:max_text_size]
    tokens = tokenize_str(text)
    indecies = np.random.choice(len(tokens), count)
    lengths = np.random.randint(1, max_seq_len, count)
    negative_samples = [' '.join(tokens[indecies[i]:indecies[i] + lengths[i]])
                        for i in range(count)]
    return negative_samples


class TrainingCounter(object):
    """
    Object for keeping track of training iterations, learning rate changes, the
    loss, training patience, and reporting the loss and learning rate changes.
    """
    def __init__(self, num_examples, batch_size, report_every, model_type,
                 ensemble_num=1, patience=5, max_lr_change=5):
        self.best_loss = 999999999
        self.total_loss = 0
        self.reporting_loss = 0
        self.epoch_loss = 0
        self.best_epoch = 0
        self.lr_changes = 0
        self.total_iter = 0
        self.epoch_iter = 0
        self.epoch = 1
        self.epoch_size = int(np.ceil(num_examples/batch_size))
        self.report_every = report_every
        self.ensemble_num = ensemble_num
        self.patience = patience
        self.no_improvement = 0
        self.max_lr_change = max_lr_change
        self.lr_changes = 0
        self.change_lr = False
        self.finish = False
        self.model_type = model_type

    def report_loss(self, epoch_end=True, report_iters=None):
        if epoch_end is True:
            loss = self.epoch_loss / self.epoch_iter
            print(
                f"ensemble {self.ensemble_num} epoch {self.epoch} loss: {loss}"
            )
        elif report_iters:
            loss = self.reporting_loss / report_iters
            print(f"Iteration {self.total_iter-1}, Loss: {loss}")
            self.reporting_loss = 0

    def update_counts(self, loss):
        self.total_loss += loss
        self.reporting_loss += loss
        self.epoch_loss += loss
        self.total_iter += 1
        self.epoch_iter += 1

        if self.epoch_iter == self.epoch_size:
            epoch_loss = self.epoch_loss / self.epoch_iter
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.best_epoch = self.epoch
                self.no_improvement = 0
            else:
                self.no_improvement += 1

            if self.epoch % self.report_every == 0:
                self.report_loss()

            self.epoch_iter = 0
            self.epoch_loss = 0
            self.epoch += 1

            return True
        else:
            return False

    def _print_update(self):
        if self.finish:
            print(f"No improvement in last 5 epochs. "
                  f"Reverting to best epoch weights "
                  f"(epoch: {self.best_epoch}, "
                  f"loss: {self.best_loss}).")
        else:
            print(f"Loss plateau - loading best weights "
                  f"(epoch: {self.best_epoch}, loss: {self.best_loss})")

    def reset_lr_change(self):
        self.change_lr = False

    def patience_function(self, warm_up):
        if self.no_improvement < self.patience:
            pass
        elif self.no_improvement == self.patience:
            if self.model_type == "mssa" or self.model_type == "setgen":
                if self.total_iter > warm_up:
                    if self.lr_changes == self.max_lr_change:
                        self.finish = True
                    else:
                        self.change_lr = True
                        self.lr_changes += 1
                        self.no_improvement = 0
                    self._print_update()
                else:
                    self.change_lr = False
            else:
                self.finish = True
                self._print_update()


class TrainingArgsExp(object):
    """
    Class used to take the arguments from a json file to supply them for
    model training. It performs a number of checks and ensures that default
    values are supplied when absent from the json file.
    ...
    Attributes
    ----------
    path_to_json : str
        path to the .json file containing the arguments for parameterising the
        model
    """
    def __init__(self, path_to_json):
        with open(path_to_json, 'r') as file:
            d = json.load(file)

        for key in ['model_type', 'obo_params', 'hub_dir', 'output']:
            try:
                d[key]
            except KeyError:
                raise KeyError(
                    f"JSON argument file requires '{key}' parameter.")

        if d['model_type'] == "ncr":
            for key in ['num_filters', 'concept_dim']:
                try:
                    d[key]
                except KeyError:
                    d[key] = 1024
        elif d['model_type'] == "sae":
            for key in ['num_filters', 'concept_dim']:
                try:
                    d[key]
                except KeyError:
                    d[key] = 1024
            for key in ['mean', 'std']:
                try:
                    d[key]
                except KeyError:
                    d[key] = True
            try:
                d['compression_ratio']
            except KeyError:
                d['compression_ratio'] = 8
        elif d['model_type'] == "mssa":
            try:
                d['scales']
            except KeyError:
                d['scales'] = [[1]]
            try:
                d['pre_norm']
            except KeyError:
                d['pre_norm'] = False
            try:
                d['concept_dim']
            except KeyError:
                d['concept_dim'] = 1024
            try:
                d['num_heads']
            except KeyError:
                d['num_heads'] = 16
        # elif d['model_type'] == "mhsa":
        #     pass
        elif d['model_type'] == "use":
            pass
        else:
            raise KeyError("Please specify model type in JSON parameter file.\n"
                           "Using either 'ncr', 'sae', or 'mssa'.")

        try:
            d['lr']
        except KeyError:
            d['lr'] = 1 / 512
        try:
            d['batch_size']
        except KeyError:
            d['batch_size'] = 256
        try:
            d['max_sequence_length']
        except KeyError:
            d['max_sequence_length'] = 10
        try:
            d['n_ensembles']
        except KeyError:
            d['n_ensembles'] = 1
        try:
            d['epochs']
        except KeyError:
            d['epochs'] = 80
        try:
            d['num_negs']
        except KeyError:
            d['num_negs'] = 0
        try:
            d['neg_file']
        except KeyError:
            d['neg_file'] = ""
        try:
            d['phrase_val']
        except KeyError:
            pass
        try:
            d['validation_rate']
        except KeyError:
            d['validation_rate'] = 5
        try:
            d['eval_mimic']
        except KeyError:
            d['eval_mimic'] = False
        try:
            d['shuffle']
        except KeyError:
            d['shuffle'] = True
        if d['epochs'] > 1:
            d['repeat'] = True

        for key in ['ignore_extra_obo', 'flat', 'no_negs']:
            try:
                d[key]
            except KeyError:
                d[key] = True

        for key in ['no_l2norm', 'verbose']:
            try:
                d[key]
            except KeyError:
                d[key] = False

        self.__dict__ = d


class AnnotateArgsExp(object):

    def __init__(self, path_to_json):
        with open(path_to_json, 'r') as file:
            d = json.load(file)

        for key in ['hub_dir', 'model_dir', 'input_dir', 'output_dir']:
            try:
                d[key]
            except KeyError:
                raise KeyError(
                    f"JSON argument file requires '{key}' parameter.")

        try:
            d['threshold']
        except KeyError:
            d['threshold'] = 0.8

        try:
            d['max_rows']
        except KeyError:
            d['max_rows'] = -1

        self.__dict__ = d


class EvaluationArgsExp(object):

    def __init__(self, path_to_json):
        with open(path_to_json, 'r') as file:
            d = json.load(file)

        for key in ['true_label_dir', 'pred_label_dir', 'obo_params']:
            try:
                d[key]
            except KeyError:
                raise KeyError(
                    f"JSON argument file requires '{key}' parameter.")

        try:
            d['output_column']
        except KeyError:
            d['output_column'] = 0

        try:
            d['no_error']
        except KeyError:
            d['no_error'] = False

        self.__dict__ = d
