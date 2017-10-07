#!/usr/bin/env python

from __future__ import print_function

import sys
from collections import defaultdict

import gflags as flags
FLAGS = flags.FLAGS

sys.setrecursionlimit(20000)

flags.DEFINE_string("train", "data/joint/shuffle.train.txt",
                    "training data")
flags.DEFINE_string("test", "data/joint/test.txt", "test data")
flags.DEFINE_string("model", None, "model to be evaled")
flags.DEFINE_bool("verbose", False, "output every parse")
flags.DEFINE_bool("extlabelfeatures", True,
                  "extend label features to include two children of s0")

flags.DEFINE_bool("eval_rstpunct", False, "include punctuations in rst evaluation")


def main():
    from spanparser.phrase_tree import PhraseTree, FScore
    from spanparser.parser import Parser as SParser
    from spanparser.features import FeatureMapper
    from spanparser.network import Network
    from rst_evaluation import const2rst, evaltrees, compare_disc_constituency
    from tree import Tree

    SParser.extlabelfeatures = FLAGS.extlabelfeatures

    training_trees = PhraseTree.load_treefile(FLAGS.train)
    fm = FeatureMapper(training_trees)
    network = Network.load(FLAGS.model)
    print("Loaded model from {}".format(FLAGS.model), file=sys.stderr)

    trees = PhraseTree.load_treefile(FLAGS.test)
    print("Evaluating on {}".format(FLAGS.test), file=sys.stderr)

    const_acc = FScore()
    dis_const_acc = FScore()
    seg_acc, span_acc, nucs_acc, labels_acc = FScore(), FScore(), FScore(), FScore()
    label_specific_acc = defaultdict(FScore)
    for i, tree in enumerate(trees):
        predicted = None
        predicted = SParser.parse(tree.sentence, fm, network)
        local_const_acc = predicted.compare(tree)
        local_dis_const_acc = compare_disc_constituency(predicted, tree)
        const_acc += local_const_acc
        dis_const_acc += local_dis_const_acc

        predicted_rst = const2rst(Tree.parse(str(predicted)),
                                  keep_punct=FLAGS.eval_rstpunct)
        ref_rst = const2rst(Tree.parse(str(tree)),
                            keep_punct=FLAGS.eval_rstpunct)
        accuracies = evaltrees(predicted_rst, ref_rst, label_specific=True)

        if FLAGS.verbose:
            print("######## Tree {} ########".format(i))
            print("# input")
            print(tree.sentence)
            print("# reference")
            print(ref_rst.pretty_str())
            print("# predicted")
            print(predicted_rst.pretty_str())

        seg_acc += accuracies['segs']
        span_acc += accuracies['spans']
        nucs_acc += accuracies['nucs']
        labels_acc += accuracies['labels']
        for label, acc in accuracies['label_specific'].iteritems():
            label_specific_acc[label] += acc

    print('Const {}, DIS:Const {}, Seg {}, Span {}, Nucs {}, Labels {}'.format(const_acc, dis_const_acc, seg_acc, span_acc, nucs_acc, labels_acc), file=sys.stderr)
    print('----------- label-specific accuracies -----------', file=sys.stderr)
    for label, acc in label_specific_acc.iteritems():
        print(label, acc, file=sys.stderr)


if __name__ == "__main__":
    # arguments for span-based parser
    flags.DEFINE_integer("dynet_mem", 10000, "Memory allocation for Dynet. (DEFAULT=10000)")
    flags.DEFINE_float("dynet_l2", 0, "L2 regularization parmeter. (DEFAULT=0)")
    flags.DEFINE_integer("dynet_seed", 123, "Seed for PNG (0: generate)")

    FLAGS(sys.argv)

    sys.argv.insert(1, str(FLAGS.dynet_mem))
    sys.argv.insert(1, "--dynet-mem")
    sys.argv.insert(1, str(FLAGS.dynet_l2))
    sys.argv.insert(1, "--dynet-l2")
    sys.argv.insert(1, str(FLAGS.dynet_seed))
    sys.argv.insert(1, "--dynet-seed")

    main()
