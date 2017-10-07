#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys

import time
import numpy as np

import gflags as flags

from rst_evaluation import const2rst, evaltrees, compare_disc_constituency
from tree import Tree

sys.setrecursionlimit(20000)
FLAGS = flags.FLAGS

flags.DEFINE_string("train", "data/joint/shuffle.train.txt",
                    "training data")
flags.DEFINE_string("dev", "data/joint/shuffle.dev.txt", "development data")
flags.DEFINE_string("save", None, "model saving path")
flags.DEFINE_integer("epoch", 30, "number of training epochs")
flags.DEFINE_integer("batch", 1, "Minibatch size")

flags.DEFINE_bool("extlabelfeatures", True,
                  "extend label features to include two children of s0")

# arguments for span-based parser
flags.DEFINE_integer("dynet_mem", 10000,
                     "Memory allocation for Dynet. (DEFAULT=10000)")
flags.DEFINE_float("dynet_l2", 0, "L2 regularization parmeter. (DEFAULT=0)")
flags.DEFINE_integer("dynet_seed", 123, "Seed for PNG (0: generate)")
flags.DEFINE_integer("word_dims", 50,
                     "Embedding dimensions for word forms. (DEFAULT=50)")
flags.DEFINE_integer("tag_dims", 20,
                     "Embedding dimensions of POS tags. (DEFAULT=2-)")
flags.DEFINE_integer("lstm_units", 200,
                     "Number of LSTM units in each layer/direction. "
                     "(DEFAULT=200)")
flags.DEFINE_integer("hidden_units", 200,
                     "Number of hidden units for each FC ReLU layer. "
                     "(DEFAULT=200)")
flags.DEFINE_float("droprate", 0.5, "Drouput probability. (DEFAULT=0.5)")
flags.DEFINE_float("unk_param", 0.8375,
                   "Parameter z for random UNKing. (DEFAULT=0.8375)")
flags.DEFINE_float("alpha", 1.0,
                   "Softmax distribution weighting for exploration. "
                   "(DEFAULT=1.0)")
flags.DEFINE_float("beta", 0.8,
                   "Probability of using oracle action in exploration. "
                   "(DEFAULT=0.8)")
flags.DEFINE_bool("eval_rstpunct", False,
                  "include punctuations in rst evaluation")

np.random.seed(123)


class Trainer(object):

    def __init__(self):
        from spanparser.features import FeatureMapper
        from spanparser.network import Network
        from spanparser.phrase_tree import PhraseTree

        self.training_trees = PhraseTree.load_treefile(FLAGS.train)
        self.dev_trees = PhraseTree.load_treefile(FLAGS.dev)

        self.fm = FeatureMapper(self.training_trees)
        self.word_count = self.fm.total_words()
        self.tag_count = self.fm.total_tags()

        self.batch = FLAGS.batch
        self.alpha = FLAGS.alpha
        self.beta = FLAGS.beta
        self.droprate = FLAGS.droprate
        self.unk_param = FLAGS.unk_param
        self.save = FLAGS.save
        self.extlabelfeatures = FLAGS.extlabelfeatures

        struct_spans = 4 if self.extlabelfeatures else 3
        label_spans = 4 if self.extlabelfeatures else 3

        self.network = Network(
            word_count=self.word_count,
            tag_count=self.tag_count,
            word_dims=FLAGS.word_dims,
            tag_dims=FLAGS.tag_dims,
            lstm_units=FLAGS.lstm_units,
            hidden_units=FLAGS.hidden_units,
            struct_out=2,
            label_out=self.fm.total_label_actions(),
            droprate=self.droprate,
            struct_spans=struct_spans,
            label_spans=label_spans
        )
        self.network.init_params()

        print('Hidden units: {},  per-LSTM units: {}'.format(
            FLAGS.hidden_units,
            FLAGS.lstm_units,
        ))
        print('Embeddings: word={}  tag={}'.format(
            (self.word_count, FLAGS.word_dims),
            (self.tag_count, FLAGS.tag_dims),
        ))
        print('Dropout rate: {}'.format(self.droprate))
        print('Parameters initialized in [-0.01, 0.01]')
        print('Random UNKing parameter z = {}'.format(self.unk_param))
        print('Exploration: alpha={} beta={}'.format(self.alpha, self.beta))

        self.training_data = self.fm.gold_data_from_trees(self.training_trees)
        self.num_batches = -(-len(self.training_data) // self.batch)
        print('Loaded {} training sentences ({} batches of size {})!'.format(
            len(self.training_data),
            self.num_batches,
            self.batch,
        ))
        self.parse_every = -(-self.num_batches // 4)

        print('Loaded {} validation trees!'.format(len(self.dev_trees)))


    def train(self):
        import dynet
        from spanparser.parser import Parser as SParser
        from spanparser.phrase_tree import FScore

        SParser.extlabelfeatures = self.extlabelfeatures

        start_time = time.time()

        best_acc = FScore()

        for epoch in xrange(1, FLAGS.epoch+1):
            print('........... epoch {} ...........'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()
            training_dis_acc = FScore()

            np.random.shuffle(self.training_data)

            for b in xrange(self.num_batches):
                batch = self.training_data[(b * self.batch): ((b + 1) * self.batch)]

                explore = [
                    SParser.exploration(
                        example,
                        self.fm,
                        self.network,
                        alpha=self.alpha,
                        beta=self.beta,
                    ) for example in batch
                ]
                for (example, acc) in explore:
                    training_acc += acc
                    training_dis_acc += compare_disc_constituency(example['predicted'], example['ref'])

                batch = [example for (example, _) in explore]

                dynet.renew_cg()
                self.network.prep_params()

                errors = []

                for example in batch:

                    # random UNKing
                    for (i, w) in enumerate(example['w']):
                        if w <= 2:
                            continue

                        freq = self.fm.word_freq_list[w]
                        drop_prob = self.unk_param / (self.unk_param + freq)
                        r = np.random.random()
                        if r < drop_prob:
                            example['w'][i] = 0

                    fwd, back = self.network.evaluate_recurrent(
                        example['w'],
                        example['t'],
                    )

                    for (left, right), correct in example['struct_data'].iteritems():
                        scores = self.network.evaluate_struct(fwd, back, left, right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs, correct))
                        errors.append(loss)
                    total_states += len(example['struct_data'])

                    for (left, right), correct in example['label_data'].items():
                        scores = self.network.evaluate_label(fwd, back, left, right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs, correct))
                        errors.append(loss)
                    total_states += len(example['label_data'])

                batch_error = dynet.esum(errors)
                total_cost += batch_error.scalar_value()
                batch_error.backward()
                self.network.trainer.update()

                mean_cost = total_cost / total_states

                print(
                    '\rBatch {}.{}  Mean Cost {:.4f} [Train: {} disc {}]'.format(
                        epoch,
                        b,
                        mean_cost,
                        training_acc,
                        training_dis_acc
                    ),
                    end='',
                    file=sys.stderr
                )
                sys.stdout.flush()

                if ((b + 1) % self.parse_every) == 0 or b == (self.num_batches - 1):
                    dev_info = self.evaluate(self.dev_trees)
                    suffix = ""
                    if dev_info["labels"] > best_acc:
                        suffix = "+"
                        best_acc = dev_info["labels"]
                    print("", file=sys.stderr)
                    print("Batch {}.{}  Mean Cost {:.4f} [Train: {} disc {}]".format(
                        epoch, b, mean_cost, training_acc, training_dis_acc
                    ), end="")
                    print('  [Dev: const {} disc_const {} seg {} span {} nucs {} labels {}{}]'.format(
                        dev_info["const"],
                        dev_info["disc_const"],
                        dev_info["seg"],
                        dev_info["span"],
                        dev_info["nucs"],
                        dev_info["labels"],
                        suffix),
                          end="")

                    if suffix == "+" and self.save:
                        self.network.save(self.save)
                        print('    [saved model: {}]'.format(self.save))
                    else:
                        print()

            current_time = time.time()
            runmins = (current_time - start_time)/60.
            print('  Elapsed time: {:.2f}m'.format(runmins))

    def evaluate(self, trees):
        from spanparser.parser import Parser as SParser
        from spanparser.phrase_tree import FScore

        const_acc = FScore()
        disc_const_acc = FScore()
        seg_acc, span_acc, nucs_acc, labels_acc = FScore(), FScore(), FScore(), FScore()
        for tree in trees:
            predicted = None
            predicted = SParser.parse(tree.sentence, self.fm, self.network)

            local_acc = predicted.compare(tree)
            const_acc += local_acc
            local_disc_acc = compare_disc_constituency(predicted, tree)
            disc_const_acc += local_disc_acc

            predicted_rst = const2rst(Tree.parse(str(predicted)),
                                      keep_punct=FLAGS.eval_rstpunct)
            ref_rst = const2rst(Tree.parse(str(tree)),
                                keep_punct=FLAGS.eval_rstpunct)
            local_accs = evaltrees(predicted_rst, ref_rst)
            seg_acc += local_accs['segs']
            span_acc += local_accs['spans']
            nucs_acc += local_accs['nucs']
            labels_acc += local_accs['labels']

        return {"const": const_acc,
                "disc_const": disc_const_acc,
                "seg": seg_acc,
                "span": span_acc,
                "nucs": nucs_acc,
                "labels": labels_acc}


if __name__ == "__main__":
    FLAGS(sys.argv)

    sys.argv.insert(1, str(FLAGS.dynet_mem))
    sys.argv.insert(1, "--dynet-mem")
    sys.argv.insert(1, str(FLAGS.dynet_l2))
    sys.argv.insert(1, "--dynet-l2")
    sys.argv.insert(1, str(FLAGS.dynet_seed))
    sys.argv.insert(1, "--dynet-seed")

    trainer = Trainer()
    trainer.train()
