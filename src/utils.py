#!/usr/bin/env python

import re


def collapse_rst_label(edge):
    relation_lc = edge.lower()
    relation = edge
    if re.search(r'^attribution', relation_lc):
        relation = "AT"  # "ATTRIBUTION"
    elif re.search(r'^(background|circumstance)', relation_lc):
        relation = "BA"  # "BACKGROUND"
    elif re.search(r'^(cause|result|consequence)', relation_lc):
        relation = "CA"  # "CAUSE"
    elif re.search(r'^(comparison|preference|analogy|proportion)',
                   relation_lc):
        relation = "COMP"  # "COMPARISON"
    elif re.search(r'^(condition|hypothetical|contingency|otherwise)',
                   relation_lc):
        relation = "COND"  # "CONDITION"
    elif re.search(r'^(contrast|concession|antithesis)', relation_lc):
        relation = "CONT"  # "CONTRAST"
    elif re.search(r'^(elaboration.*|example|definition)', relation_lc):
        relation = "EL"  # "ELABORATION"
    elif re.search(r'^(purpose|enablement)', relation_lc):
        relation = "EN"  # "ENABLEMENT"
    elif re.search(r'^(problem\-solution|question\-answer|statement\-response|topic\-comment|comment\-topic|rhetorical\-question)', relation_lc):
        relation = "TO"  # "TOPICCOMMENT"
    elif re.search(r'^(evaluation|interpretation|conclusion|comment)',
                   relation_lc):
        # note that this check for "comment" needs to come after the one
        # above that looks for "comment-topic"
        relation = "EV"  # "EVALUATION"
    elif re.search(r'^(evidence|explanation.*|reason)', relation_lc):
        relation = "EX"  # "EXPLANATION"
    elif re.search(r'^(list|disjunction)', relation_lc):
        relation = "JO"  # "JOINT"
    elif re.search(r'^(manner|means)', relation_lc):
        relation = "MA"  # "MANNERMEANS"
    elif re.search(r'^(summary|restatement)', relation_lc):
        relation = "SU"  # "SUMMARY"
    elif re.search(r'^(temporal\-.*|sequence|inverted\-sequence)',
                   relation_lc):
        relation = "TE"  # "TEMPORAL"
    elif re.search(r'^(topic-.*)', relation_lc):
        relation = "TC"  # "TOPICCHANGE"
    elif re.search(r'^(span|textualorganization)$', relation_lc):
        pass
    elif re.search(r'^(same\-unit)', relation_lc):
        relation = "SameUnit"
    else:
        raise ValueError('unknown relation type in label: {}'.format(edge))
    return relation


def collapse_label_rst(rstt):
    if rstt.rel2par is not None:
        rstt.rel2par = collapse_rst_label(rstt.rel2par)
    for c in rstt.children:
        collapse_label_rst(c)
