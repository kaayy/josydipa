#!/usr/bin/env python

"""
Tokenize the RST Trees
"""

import sys
logs = sys.stderr
import glob
import gflags as flags
import nltk.tokenize.treebank as treebank

from rsttree import RSTTree

flags.DEFINE_string("rst_path", "dataset/rst/train", "path to RST trees")

FLAGS = flags.FLAGS

def tokenize_tree():

    tokenizer = treebank.TreebankWordTokenizer()

    def tokenize_edu(edu_node):
        if edu_node.text:
            edu_node.text = tokenizer.tokenize(edu_node.text, convert_parentheses=True, return_str=True)

    for rstf in glob.glob(FLAGS.rst_path + "/*.dis"):
        if rstf.endswith("dis"):
            basename = rstf.rsplit("/", 1)[1].split(".")[0]
            if basename.startswith("wsj"):
                print >> logs, "Tokenizing", basename
                rstlines = " ".join([line.strip() for line in open(rstf).readlines()])
                rstt = RSTTree.parse(rstlines)

                rstt.postorder_visit(tokenize_edu)

                tgtfile = FLAGS.rst_path + "/" + basename + ".out.dis.tok"
                prettystr = rstt.pretty_str() + "\n"
                open(tgtfile, "w").write(prettystr)

if __name__ == "__main__":
    FLAGS(sys.argv)
    tokenize_tree()
