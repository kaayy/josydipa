#!/usr/bin/env python

"""
Aligns the RST trees with constituency trees. Outputs in bracketing format.
"""

import sys

import glob
import gflags as flags

from rsttree import RSTTree
from tree import Tree
from utils import collapse_rst_label, collapse_label_rst
from rst_evaluation import const2rst, evaltrees

logs = sys.stderr
FLAGS = flags.FLAGS

flags.DEFINE_string("rst_path", "data/rst/train", "RST tree path")
flags.DEFINE_string("const_path", "data/wsj", "Constituency tree path")


def undo_parentheses(text):
    return text.replace("-LRB-", "(")\
               .replace("-RRB-", ")")\
               .replace("-LSB-", "[")\
               .replace("-RSB-", "]")\
               .replace("-LCB-", "{")\
               .replace("-RCB-", "}")


def undo_symbol(text):
    return undo_parentheses(text).replace("\/", "/")\
        .replace("\*", "*")


def redo_parentheses(text):
    return text.replace("(", "-LRB-")\
               .replace(")", "-RRB-")\
               .replace("[", "-LSB-")\
               .replace("]", "-RSB-")\
               .replace("{", "-LCB-")\
               .replace("}", "-RCB-")


def load_trees():
    rstpath = "%s/" % FLAGS.rst_path
    constpath = "%s/" % FLAGS.const_path
    treepairs = []
    for rstf in glob.glob(rstpath + "*.dis.tok"):
        basename = rstf.rsplit("/", 1)[1].split(".")[0]
        try:
            if basename.startswith("wsj"):
                rstline = " ".join([line.strip() for line in open(rstf).readlines()])
                rstt = RSTTree.parse(rstline)
                constts = [Tree.parse(line.strip()) for line in open(constpath + basename + ".cleangold")]
                treepairs.append((basename, rstt, constts))
        except:
            print >> logs, "Failed in loading", basename  
    return treepairs


def fuzzy_startswith(reftext, token):
    if reftext.startswith(token):
        return True
    elif len(reftext) >= len(token) and all(not x.isalnum() for x in token) and all(not x.isalnum() for x in reftext[:len(token)]):
        # if both are punctuations
        return True
    else:
        return False


def align_tokens(rstt, tokens, start=0):
    if rstt.text is None:
        rstt.leftidx = start
        for c in rstt.children:
            start = align_tokens(c, tokens, start)
        rstt.rightidx = start
        return start
    else:
        rstt.leftidx = start
        educhars = "".join(x for x in undo_parentheses(rstt.text).split())  # ignore all whitespaces
        while len(educhars) > 0 and start < len(tokens):
            if fuzzy_startswith(educhars, tokens[start]):
                educhars = educhars[len(tokens[start]):]
                start += 1
            else:
                print >> logs, rstt.text
                print >> logs, educhars
                print >> logs, tokens[start:start+10]
                assert False
        if len(educhars) > 0:
            print >> logs, rstt.text
            print >> logs, educhars
            print >> logs, tokens[-10:]
        rstt.rightidx = start
        rstt.text = " ".join(tokens[rstt.leftidx:rstt.rightidx])
        return start


def mark_spans(tree, start, span2const):
    if tree.is_leaf():
        tree.leftidx = start
        tree.rightidx = start+1
        span2const[(tree.leftidx, tree.rightidx)] = tree
        return start+1
    else:
        tree.leftidx = start
        for c in tree.children:
            start = mark_spans(c, start, span2const)
        tree.rightidx = start
        # note that for unary rules, only the parent node is remembered
        span2const[(tree.leftidx, tree.rightidx)] = tree
        return start


def find_largest_right_subtree(span2const, left, right):
    """find the largest subtree that starts at left, ends before right"""
    rightmost = -1
    tree = None
    for ((l, r), t) in span2const.iteritems():
        if l == left and r > rightmost and r <= right:
            rightmost = r
            tree = t
    return tree, rightmost


def find_smallest_common_subtree(span2const, left, right):
    """find the lowest subtree that covers span (left, right)"""
    root = None
    for ((l, r), t) in span2const.iteritems():
        if l <= left and right <= r and (root == None or r-l < root.rightidx-root.leftidx):
            root = t
    return root


def prune_tree(root, left, right):
    """remove all nodes that are (totally) out of the span (left, right)"""
    newchildren = []
    for c in root.children:
        #print c.leftidx, c.rightidx, left, right
        if c.leftidx < left and c.rightidx > right:  # this children fully convers the span
            newchildren.append(prune_tree(c, left, right))
        elif c.leftidx < left and c.rightidx <= right and c.rightidx > left:
            newchildren.append(prune_tree(c, left, right))
        elif c.leftidx >= left and c.rightidx <= right:
            newchildren.append(c)
        elif c.leftidx >= left and c.leftidx < right and c.rightidx > right:
            newchildren.append(prune_tree(c, left, right))
    #print len(newchildren)
    # aggregate all children, remove all consecutive unary rules
    if len(newchildren) == 1 and len(newchildren[0].children) == 1:
        newchildren[0].val = root.val
        return newchildren[0]
    else:
        ret = Tree(val=root.val, children=newchildren)
        ret.leftidx = newchildren[0].leftidx
        ret.rightidx = newchildren[-1].rightidx
        return ret


def build_newtree(rstt, span2const):
    if rstt.text is None:
        # non-leaf nodes
        sats = []
        nucs = []
        label_appendix = ""
        for i, c in enumerate(rstt.children):
            if c.nodetype == "Nucleus":
                nucs.append(c)
                label_appendix = ":%d" % i
            elif c.nodetype == "Satellite":
                sats.append(c)
            else:
                assert False
        assert (len(nucs) > 1 and len(sats)==0) or (len(nucs)==1 and len(sats)>0)

        new_children = [build_newtree(c, span2const) for c in rstt.children]
        if len(sats) > 0:
            # nuc + sat
            label = "DIS:" + collapse_rst_label(sats[0].rel2par) + label_appendix
            newtree = Tree(val=label, children=new_children)
            return newtree
        else:
            # nucs only
            label = "DIS:" + collapse_rst_label(nucs[0].rel2par)
            newtree = Tree(val=label, children=new_children)
            return newtree
    else:
        # leaf nodes
        if (rstt.leftidx, rstt.rightidx) in span2const:
            return span2const[(rstt.leftidx, rstt.rightidx)]
        else:
            left = rstt.leftidx
            right = rstt.rightidx
            subtrees = []
            while left < right:
                tree, left = find_largest_right_subtree(span2const, left, right)
                subtrees.append(tree)
            assert len(subtrees) > 1

            #print >> logs, "for span", rstt.leftidx, rstt.rightidx
            root = find_smallest_common_subtree(span2const, rstt.leftidx, rstt.rightidx)
            if root is not None:
                #print >> logs, "root span", root.leftidx, root.rightidx
                newtree = prune_tree(root, rstt.leftidx, rstt.rightidx)
            else:
                # fallback
                print >> logs, "fallback", len(subtrees), "subtrees =>", rstt.text
                newtree = Tree(val="S", children=subtrees)

            return newtree


def align_trees(treepairs):
    for wsjname, rstt, constts in treepairs:
        print >> logs, "align", wsjname
        rst_sent = rstt.get_sentence()
        const_sents = [t.get_sentence() for t in constts]
        const_allsent = []
        for s in const_sents:
            const_allsent += s

        # 1. compare the characters without whitespaces & punctuations, i.e., ignore the tokenization
        rst_string = undo_parentheses(" ".join(rst_sent))
        const_string = undo_parentheses(" ".join(const_allsent))
        rst_string = "".join(x for x in rst_string if x.isalnum())
        const_string = "".join(x for x in const_string if x.isalnum())

        # const string should cover rst string
        if not const_string.startswith(rst_string):
            print >> logs, "string length rst", len(rst_string), "const", len(const_string)
            for i in xrange(min(len(rst_string), len(const_string))):
                if rst_string[i] != const_string[i]:
                    print >> logs, i
                    print >> logs, rst_string[i:]
                    print >> logs, const_string[i:]
                    break

        # 2. compare the tokens
        align_tokens(rstt, [undo_symbol(t) for t in const_allsent])

        # 3. mark global indexes for the boundaries of each constituent tree
        i = 0
        span2const = dict()
        for t in constts:
            i = mark_spans(t, i, span2const)

        # 4. build rst tree in constituent format
        newtree = build_newtree(rstt, span2const)

        # 5. sanity check: make sure the new tree contains
        # the same text as the old rst tree
        new_sent = newtree.get_sentence()
        ref_rst_sent = const_allsent[rstt.leftidx:rstt.rightidx]
        if new_sent != ref_rst_sent:
            for i in xrange(len(new_sent)):
                if new_sent[i] != ref_rst_sent[i]:
                    print >> logs, "diffs at", i
                    print >> logs, "new sent", " ".join(new_sent[i:i+10])
                    print >> logs, "rst", " ".join(ref_rst_sent[i:i+10])
                    break
            print >> logs, "new sent ends with", " ".join(new_sent[-5:])
            print >> logs, "rst sent ends with", " ".join(ref_rst_sent[-5:])
        assert new_sent == ref_rst_sent, "new tree differs from constituent trees"

        # 6. sanity check: convert from constituent tree to rst tree,
        # compare it with the orginal (aligned) rst tree
        new_rst = const2rst(newtree, keep_punct=True)
        collapse_label_rst(rstt)
        # rstt.compare_with(new_rst)
        accuracies = evaltrees(new_rst, rstt)
        print >> logs, "seg", accuracies['segs'].fscore(), \
            "span", accuracies['spans'].fscore(), \
            "nucs", accuracies['nucs'].fscore(), \
            "labels", accuracies['labels'].fscore()

        # output aligned tree
        print newtree


if __name__ == "__main__":
    FLAGS(sys.argv)
    treepairs = load_trees()
    align_trees(treepairs)
    print >> logs, len(treepairs), "aligned"
