import argparse
import os
import re

import tensorflow as tf

from models import model_chassis
from utils.lr_shed import LinearWarmup
from utils.onto import Ontology
from utils.utils import save_ont_and_args, sample_negatives_from_file, \
    TrainingCounter, TrainingArgsExp, exp_input_fn, phrase2vec#, phrase2vecUSE


def main():

    parser = argparse.ArgumentParser(description='Hello!')
    parser.add_argument('--params',
                        help="json file containing training parameters")
    args = TrainingArgsExp(parser.parse_args().params)
    print('Loading the ontology...')
    ont = Ontology(args.obo_params)
    model = model_chassis.ExplicitChassis(args, ont)

    if (not args.no_negs) and args.neg_file != "":
        print("Sampling negatives")
        negative_samples = sample_negatives_from_file(args.neg_file,
                                                      args.num_negs,
                                                      args.max_sequence_length)
        ont.n_concepts += 1
    else:
        negative_samples = None

    try:
        os.mkdir(f'stream_data')
    except FileExistsError:
        pass

    len_labs = 0
    num_phrases = args.num_negs
    pat = re.compile(r"\/")
    data_path = f'stream_data/{pat.sub("_", args.output)}_phrase.txt'
    label_path = f'stream_data/{pat.sub("_", args.output)}_label.txt'

    with open(data_path, 'w') as raw_data, \
            open(label_path, 'w') as labels:
        for c in ont.concepts:
            for name in ont.names[c]:
                raw_data.write(f'{name}\n')
                labels.write(f'{str(ont.concept2id[c])}\n')
                num_phrases += 1

            len_labs += 1

        if negative_samples is not None:
            none_id = len_labs
            len_labs += 1

            for neg in negative_samples:
                raw_data.write(f'{neg}\n')
                labels.write(f'{none_id}\n')

    print(f"Number of unique concepts: {len(ont.concepts)}")
    print(f"Number of training samples: {num_phrases}")
    ds_gen = exp_input_fn(config=args, num_phrases=num_phrases,
                          data_path=data_path, label_path=label_path)

    if args.model_type != "mssa":
        wu_iter = 1
        optimizers = [tf.keras.optimizers.Adam(learning_rate=args.lr)
                      for _ in range(args.n_ensembles)]
    else:
        if not args.pre_norm:
            wu_iter = 20000
            lr_shed = LinearWarmup(warmup_steps=wu_iter,
                                   final_learning_rate=args.lr,
                                   init_learning_rate=0)
            optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_shed)
                          for _ in range(args.n_ensembles)]
        else:
            wu_iter = 1
            lr_shed = LinearWarmup(warmup_steps=wu_iter,
                                   final_learning_rate=args.lr,
                                   init_learning_rate=args.lr)

            optimizers = [tf.keras.optimizers.Adam(learning_rate=lr_shed)
                          for _ in range(args.n_ensembles)]

    param_dir = args.output

    if not os.path.exists(param_dir):
        os.makedirs(param_dir)

    report_len = 20
    print("training start")

    for ens_i, core in enumerate(model.cores):
        counter = TrainingCounter(num_examples=num_phrases,
                                  batch_size=args.batch_size, report_every=1,
                                  model_type=args.model_type,
                                  ensemble_num=ens_i, patience=5)
        for batch in ds_gen:
            if args.model_type != "use":
                input_seq, lens = phrase2vec(model.elmo,
                                             batch[0].numpy(),
                                             model.config)
            # else:
            #     input_seq = phrase2vecUSE(model.use,
            #                               batch[0].numpy())

            labels = batch[1]

            with tf.GradientTape() as tape:
                if args.model_type != "mssa":
                    logits = core(input_seq)
                else:
                    logits = core(input_seq, lens, training=True)

                loss = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels,
                        logits=logits
                    )
                )

            grads = tape.gradient(
                loss, core.trainable_weights
            )
            optimizers[ens_i].apply_gradients(
                zip(grads, core.trainable_weights)
            )

            counter.update_counts(loss.numpy())

            if counter.no_improvement == 0:
                save_ont_and_args(ont, args, param_dir)
                model.save_weights(param_dir)

            if args.verbose and counter.total_iter % report_len == 0:
                counter.report_loss(epoch_end=False, report_iters=report_len)

            counter.patience_function(warm_up=wu_iter)

            if counter.finish:
                model.ensembled_model.load_weights(
                    param_dir + '/model_weights.h5')
                break
            elif counter.change_lr:
                model.ensembled_model.load_weights(
                    param_dir + '/model_weights.h5')
                new_lr = optimizers[ens_i].learning_rate.learning_rate * 0.2
                optimizers[ens_i].learning_rate.assign_lr(new_lr)
                print(f"new learning rate: {new_lr}")
                counter.reset_lr_change()

    print("Training complete")
    print("Saving model")
    save_ont_and_args(ont, args, param_dir)
    model.save_weights(param_dir)
    print("Done")


if __name__ == "__main__":
    main()
