#!/usr/bin/env python

from rsttree import RSTTree
from spanparser.fscore import FScore


def const2rst(constt, keep_punct=False):
    """convert a constituent tree to RST tree"""
    t = _const2rst(constt, "Root", None)
    if not keep_punct:
        remove_rst_punct(t)
    _mark_span(t, 1)
    return t


def _const2rst(constt, nodetype, rel2par):
    # assert constt.val.startswith("DIS:")
    if not constt.val.startswith("DIS:"):
        text = " ".join(constt.get_sentence())
        return RSTTree(nodetype, loc=0, rel2par=rel2par, text=text, children=[])

    root = RSTTree(nodetype, loc=0, rel2par=rel2par, text=None, children=[])
    relation = constt.val[4:]
    relationsplit = relation.rsplit(":")
    children_nodetype = None
    children_rel2par = None
    if len(relationsplit) == 1:
        # conjunctions
        children_nodetype = ["Nucleus" for _ in xrange(len(constt.children))]
        children_rel2par = [relation for _ in xrange(len(constt.children))]
    else:
        # normal nucleus-satellite relation
        nucidx = int(relationsplit[1])
        children_nodetype = []
        children_rel2par = []
        for i, c in enumerate(constt.children):
            if i == nucidx:
                children_nodetype.append("Nucleus")
                children_rel2par.append("span")
            else:
                children_nodetype.append("Satellite")
                children_rel2par.append(relationsplit[0])

    children = []
    for nt, r, c in zip(children_nodetype, children_rel2par, constt.children):
        if not c.val.startswith("DIS:"):
            text = " ".join(c.get_sentence())
            newc = RSTTree(nt, loc=0, rel2par=r, text=text, children=[])
            children.append(newc)
        else:
            newc = _const2rst(c, nt, r)
            children.append(newc)

    root.children = children
    return root


def remove_rst_punct(t):
    if t.text is None:
        for c in t.children:
            remove_rst_punct(c)
    else:
        toks = t.text.split()
        toks = [tok for tok in toks if any(w.isalnum() for w in tok)]
        t.text = " ".join(toks)


def _mark_span(t, idx):
    left = float("inf")
    right = -float("inf")
    if len(t.children) == 0:
        t.loc = idx
        return idx + 1
    else:
        for c in t.children:
            idx = _mark_span(c, idx)
            if type(c.loc) == tuple:
                left = min(left, c.loc[0])
                right = max(right, c.loc[1])
            else:
                left = min(left, c.loc)
                right = max(right, c.loc)
        t.loc = (left, right)
        return idx


def collect_token_edus(tree):

    def collect_edus(tree):
        ret = dict()
        if type(tree.loc) == tuple:
            for c in tree.children:
                ret.update(collect_edus(c))
        else:
            assert tree.text is not None, "wrong leaf %s" % str(tree)
            ret[tree.loc] = tree.text
        return ret

    def collect_spans(tree, leafspan, spans):
        if tree.nodetype != "Root":
            #assert type(tree.loc) == tuple
            # N.B.! we consider the spans of leaves also
            if type(tree.loc) == tuple:
                spans.append((tree.nodetype, tree.rel2par, leafspan[tree.loc[0]][0], leafspan[tree.loc[1]][1]))
            else:
                spans.append((tree.nodetype, tree.rel2par, leafspan[tree.loc][0], leafspan[tree.loc][1]))
        for c in tree.children:
            collect_spans(c, leafspan, spans)

    edus = collect_edus(tree)
    leafspan = dict()
    num_leaves = max(edus.keys())
    prev = 0
    segs = []
    for i in xrange(1, num_leaves+1):
        text = edus[i]
        edulen = len(text.split())
        leafspan[i] = (prev, prev + edulen)
        prev += edulen
        segs.append(prev)

    segs.pop()
    assert len(segs) == num_leaves - 1

    spans = []
    collect_spans(tree, leafspan, spans)

    return segs, spans


def precision_recall_f1(test, ref):
    return FScore(correct=len(test & ref), predcount=len(test), goldcount=len(ref))


def evaltrees(tree, reftree, label_specific=False):
    global_segs = set()
    global_spans = set()
    global_refsegs = set()
    global_refspans = set()
    global_nucs = set()
    global_refnucs = set()
    global_labels = set()
    global_reflabels = set()

    t = tree
    rt = reftree

    segs, spans = collect_token_edus(t)
    refsegs, refspans = collect_token_edus(rt)
    global_segs.update(segs)
    global_refsegs.update(refsegs)
    global_labels.update((nodetype, label, left, right) for nodetype, label, left, right in spans)
    global_reflabels.update((nodetype, label, left, right) for nodetype, label, left, right in refspans)
    global_nucs.update((nodetype, left, right) for nodetype, _, left, right in spans)
    global_refnucs.update((nodetype, left, right) for nodetype, _, left, right in refspans)
    global_spans.update((left, right) for _, _, left, right in spans)
    global_refspans.update((left, right) for _, _, left, right in refspans)

    accuracies = {}
    accuracies['segs'] = precision_recall_f1(global_segs, global_refsegs)
    accuracies['spans'] = precision_recall_f1(global_spans, global_refspans)
    accuracies['nucs'] = precision_recall_f1(global_nucs, global_refnucs)
    accuracies['labels'] = precision_recall_f1(global_labels, global_reflabels)

    if label_specific:
        labels = set([label for (_, label, _, _) in refspans])
        labels.update(label for (_, label, _, _) in spans)
        label_specific_accuracy = {}
        for label in labels:
            label_refspans = set([(nodetype, left, right) for (nodetype, l, left, right) in refspans if l == label])
            label_spans = set([(nodetype, left, right) for (nodetype, l, left, right) in spans if l == label])
            label_specific_accuracy[label] = precision_recall_f1(label_spans, label_refspans)
        accuracies['label_specific'] = label_specific_accuracy

    return accuracies


def compare_disc_constituency(predicted, reference):
    predicted_brackets = predicted.brackets(True)
    reference_brackets = reference.brackets(True)

    predicted_brackets = set(x for x in predicted_brackets if x[0].startswith("DIS:"))
    reference_brackets = set(x for x in reference_brackets if x[0].startswith("DIS:"))

    return FScore(correct=len(predicted_brackets & reference_brackets),
                  predcount=len(predicted_brackets), goldcount=len(reference_brackets))
