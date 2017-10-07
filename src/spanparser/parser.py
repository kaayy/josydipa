"""
Shift-Combine-Label parser.
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import dynet

from phrase_tree import PhraseTree, FScore


class Parser(object):

    extlabelfeatures = None

    def __init__(self, n):
        """
        Initial state for parsing an n-word sentence.
        """
        self.n = n
        self.i = 0
        self.stack = []
        # information needed by extended label features
        # split point of s0
        self.splitpoint = None

    def can_shift(self):
        return (self.i < self.n)

    def can_combine(self):
        return (len(self.stack) > 1)

    def shift(self):
        j = self.i  # (index of shifted word)
        treelet = PhraseTree(leaf=j)
        self.stack.append((j, j, False, [treelet]))
        self.i += 1

    def combine(self):
        (s0left, right, disc0, treelist0) = self.stack.pop()
        (left, _, disc1, treelist1) = self.stack.pop()
        self.splitpoint = s0left
        self.stack.append((left, right, disc0 or disc1, treelist1 + treelist0))

    def label(self, nonterminals=[]):

        for nt in nonterminals:
            (left, right, disc, trees) = self.stack.pop()
            disc = disc or nt.startswith("DIS:")
            tree = PhraseTree(symbol=nt, children=trees)
            self.stack.append((left, right, disc, [tree]))

    def take_action(self, action):
        if action == 'sh':
            self.shift()
        elif action == 'comb':
            self.combine()
        elif action == 'none':
            return
        elif action == "DIS:none":
            left, right, _, trees = self.stack[-1]
            self.stack[-1] = (left, right, True, trees)
        elif action.startswith('label-'):
            self.label(action[6:].split('-'))
        else:
            raise RuntimeError('Invalid Action: {}'.format(action))

    def finished(self):
        return (
            (self.i == self.n) and
            (len(self.stack) == 1) and
            (len(self.stack[0][3]) == 1)
        )

    def tree(self):
        if not self.finished():
            raise RuntimeError('Not finished.')
        return self.stack[0][3][0]

    def s_features(self):
        """
        Features for predicting structural action (shift, combine):
            (pre-s1-span, s1-span, s0-span, post-s0-span)
        Note features use 1-based indexing:
            ... a span of (1, 1) means the first word of sentence
            ... (x, x-1) means no span
        """
        lefts = []
        rights = []

        # pre-s1-span
        lefts.append(1)
        if len(self.stack) < 2:
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            rights.append(s1_left - 1)

        # s1-span
        if len(self.stack) < 2:
            lefts.append(1)
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            lefts.append(s1_left)
            s1_right = self.stack[-2][1] + 1
            rights.append(s1_right)

        # s0-span
        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def l_features(self):
        """
        Features for predicting label action:
            (pre-s0-span, s0-span, post-s0-span)
        """
        lefts = []
        rights = []

        # pre-s0-span
        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            rights.append(s0_left - 1)

        # s0-span
        if Parser.extlabelfeatures:
            if len(self.stack) < 1:
                lefts.append(1)
                lefts.append(1)
                rights.append(0)
                rights.append(0)
            else:
                s0_left = self.stack[-1][0] + 1
                s0_right = self.stack[-1][1] + 1
                if self.splitpoint is None:
                    # from shift
                    lefts.append(s0_left)
                    rights.append(s0_right)
                    lefts.append(1)
                    rights.append(0)
                else:
                    # from combine
                    lefts.append(s0_left)
                    rights.append(self.splitpoint)
                    lefts.append(self.splitpoint+1)
                    rights.append(s0_right)
        else:
            if len(self.stack) < 1:
                lefts.append(1)
                rights.append(0)
            else:
                s0_left = self.stack[-1][0] + 1
                lefts.append(s0_left)
                s0_right = self.stack[-1][1] + 1
                rights.append(s0_right)

        # post-s0-span
        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def s_oracle(self, tree):
        """
        Returns correct structural action in current (arbitrary) state, 
            given gold tree.
            Deterministic (prefer combine).
        """
        if not self.can_shift():
            return 'comb'
        elif not self.can_combine():
            return 'sh'
        else:
            (left0, right0, _, _) = self.stack[-1]
            a, _ = tree.enclosing(left0, right0)
            if a == left0:
                return 'sh'
            else:
                return 'comb'

    def l_oracle(self, tree):
        (left0, right0, disc, _) = self.stack[-1]
        labels = tree.span_labels(left0, right0)[::-1]
        if len(labels) == 0:
            return 'none' if not disc else "DIS:none"
        else:
            return 'label-' + '-'.join(labels)

    @staticmethod
    def gold_actions(tree):
        n = len(tree.sentence)
        state = Parser(n)
        result = []

        for step in range(2 * n - 1):

            if state.can_combine():
                (left0, right0, _, _) = state.stack[-1]
                (left1, _, _, _) = state.stack[-2]
                a, b = tree.enclosing(left0, right0)
                if left1 >= a:
                    result.append('comb')
                    state.combine()
                else:
                    result.append('sh')
                    state.shift()
            else:
                result.append('sh')
                state.shift()

            (left0, right0, disc, _) = state.stack[-1]
            labels = tree.span_labels(left0, right0)[::-1]
            if len(labels) == 0:
                if disc:
                    left, right, _, trees = state.stack[-1]
                    state.stack[-1] = (left, right, True, trees)
                    result.append("DIS:none")
                else:
                    result.append('none')
            else:
                result.append('label-' + '-'.join(labels))
                state.label(labels)

        return result

    @staticmethod
    def training_data(tree):
        """
        Using oracle (for gold sequence), omitting mandatory S-actions
        """
        s_features = []
        l_features = []

        n = len(tree.sentence)
        state = Parser(n)

        for step in range(2 * n - 1):

            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                action = state.s_oracle(tree)
                features = state.s_features()
                s_features.append((features, action))
            state.take_action(action)

            action = state.l_oracle(tree)
            features = state.l_features()
            l_features.append((features, action))
            state.take_action(action)

        return (s_features, l_features)

    @staticmethod
    def exploration(data, fm, network, alpha=1.0, beta=0):
        """
        Only data from this parse, including mandatory S-actions.
            Follow softmax distribution for structural data.
        """

        dynet.renew_cg()
        network.prep_params()

        struct_data = {}
        label_data = {}

        tree = data['tree']
        sentence = tree.sentence

        n = len(sentence)
        state = Parser(n)

        w = data['w']
        t = data['t']
        fwd, back = network.evaluate_recurrent(w, t, test=True)

        for step in xrange(2 * n - 1):

            features = state.s_features()
            if not state.can_combine():
                action = 'sh'
                correct_action = 'sh'
            elif not state.can_shift():
                action = 'comb'
                correct_action = 'comb'
            else:
                correct_action = state.s_oracle(tree)

                r = np.random.random()
                if r < beta:
                    action = correct_action
                else:
                    left, right = features
                    scores = network.evaluate_struct(
                        fwd,
                        back,
                        left,
                        right,
                        test=True,
                    ).npvalue()

                    # sample from distribution
                    exp = np.exp(scores * alpha)
                    softmax = exp / (exp.sum())
                    r = np.random.random()

                    if r <= softmax[0]:
                        action = 'sh'
                    else:
                        action = 'comb'

            struct_data[features] = fm.s_action_index(correct_action)
            state.take_action(action)

            features = state.l_features()
            correct_action = state.l_oracle(tree)
            label_data[features] = fm.l_action_index(correct_action)

            r = np.random.random()
            if r < beta:
                action = correct_action
            else:
                left, right = features
                scores = network.evaluate_label(
                    fwd,
                    back,
                    left,
                    right,
                    test=True,
                ).npvalue()
                sorted_actions = sorted([(s, a) for (a, s) in enumerate(scores)], reverse=True)
                if step < (2 * n - 2):
                    i = 0
                    action_index = sorted_actions[i][1]
                    action = fm.l_action(action_index)
                    while not valid_laction(action, state):
                        i += 1
                        action_index = sorted_actions[i][1]
                        action = fm.l_action(action_index)
                else:
                    i = 0
                    action_index = sorted_actions[i][1]
                    action = fm.l_action(action_index)
                    while (not valid_laction(action, state)) or action_index == 0 or action_index == 1:
                        i += 1
                        action_index = sorted_actions[i][1]
                        action = fm.l_action(action_index)
                action = fm.l_action(action_index)
            state.take_action(action)

        predicted = state.stack[0][3][0]
        predicted.propagate_sentence(sentence)
        accuracy = predicted.compare(tree)

        example = {
            'w': w,
            't': t,
            'struct_data': struct_data,
            'label_data': label_data,
            'predicted': predicted,
            'ref': tree
        }

        return example, accuracy

    @staticmethod
    def parse(sentence, fm, network):

        dynet.renew_cg()
        network.prep_params()

        n = len(sentence)
        state = Parser(n)

        w, t = fm.sentence_sequences(sentence)

        fwd, back = network.evaluate_recurrent(w, t, test=True)

        for step in xrange(2 * n - 1):

            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                left, right = state.s_features()
                scores = network.evaluate_struct(
                    fwd,
                    back,
                    left,
                    right,
                    test=True,
                ).npvalue()
                action_index = np.argmax(scores)
                action = fm.s_action(action_index)
            state.take_action(action)

            left, right = state.l_features()
            scores = network.evaluate_label(
                fwd,
                back,
                left,
                right,
                test=True,
            ).npvalue()
            sorted_actions = sorted([(s, a) for (a, s) in enumerate(scores)], reverse=True)
            if step < (2 * n - 2):
                i = 0
                action_index = sorted_actions[i][1]
                action = fm.l_action(action_index)
                while not valid_laction(action, state):
                    i += 1
                    action_index = sorted_actions[i][1]
                    action = fm.l_action(action_index)
            else:
                i = 0
                action_index = sorted_actions[i][1]
                action = fm.l_action(action_index)
                while (not valid_laction(action, state)) or action_index == 0 or action_index == 1:
                    i += 1
                    action_index = sorted_actions[i][1]
                    action = fm.l_action(action_index)
            action = fm.l_action(action_index)
            state.take_action(action)

        if not state.finished():
            raise RuntimeError('Bad ending state!')

        tree = state.stack[0][3][0]
        tree.propagate_sentence(sentence)
        return tree

    @staticmethod
    def evaluate_corpus(trees, fm, network):
        accuracy = FScore()
        for tree in trees:
            predicted = Parser.parse(tree.sentence, fm, network)
            local_accuracy = predicted.compare(tree)
            accuracy += local_accuracy
        return accuracy

    @staticmethod
    def write_predicted(fname, test_trees, fm, network):
        """
        Input trees being used only to carry sentences.
        """
        f = open(fname, 'w')
        for tree in test_trees:
            predicted = Parser.parse(tree.sentence, fm, network)
            topped = PhraseTree(
                symbol='TOP',
                children=[predicted],
                sentence=predicted.sentence,
            )
            f.write(str(topped))
            f.write('\n')
        f.close()


def valid_laction(action, state):
    """ two heuristics for discourse labels
        1. if the children are marked discourse level, the parent should also
           be in discourse level
        2. if the label is discourse label and the tree is not binary, the
           label must not contain the index of the nucleus, since every child
           is nucleus
        3. if the label is discourse label, then the labels of the children
           must be discourse label
    """
    (_, _, disc, subtrees) = state.stack[-1]
    num_subtrees = len(subtrees)
    if disc and action == "DIS:none":
        return True
    if disc and not action[6:].startswith("DIS:"):
        return False
    if action[6:].startswith("DIS:") and num_subtrees > 2 and len(action[6:].split(":")) != 2:
        return False
    return True
