#!/usr/bin/env python

import sys

class Tree(object):
    """ Data structure for constituency tree."""

    __slots__ = "children", "val", "leftidx", "rightidx"

    def __init__(self, val, children):
        self.val = val
        self.children = children
        self.leftidx = None
        self.rightidx = None

    def __str__(self):
        if self.is_leaf():
            return "%s" % self.val
        else:
            return "(%s %s)" % (self.val, " ".join(str(x) for x in self.children))

    def is_leaf(self):
        return len(self.children) == 0

    def get_sentence(self, sent=None):
        if sent is None:
            sent = []
        if self.is_leaf():
            sent.append(self.val)
        else:
            for c in self.children:
                c.get_sentence(sent)
        return sent

    def labels(self):
        if not self.is_leaf():
            ret = [self.val]
            for c in self.children:
                ret += c.labels()
            return ret
        else:
            return []

    @staticmethod
    def parse(string):
        """Loads a tree from the input string.

        Args:
            string: tree string in parentheses form.
        Returns:
            A tree represented as nested tuples.
        """
        string += " "
        _, t = Tree._parse(string, 0)
        return t

    @staticmethod
    def _parse(line, index):
        assert line[index] == "(", "Invalid tree string %s at %d" % (line, index)
        index += 1
        children = []
        val = None
        while line[index] != ")":
            if line[index] == "(":
                index, t = Tree._parse(line, index)
                children.append(t)

            else:
                # leaf
                rpos = min(line.find(" ", index), line.find(")", index))

                label = line[index:rpos]  # label or word
                if val == None:
                    val = label
                else:
                    children.append(Tree(val=label, children=[]))
                index = rpos

            if line[index] == " ":
                index += 1

        assert line[index] == ")", "Invalid tree string %s at %d" % (line, index)

        t = Tree(val=val, children=children)
        return index+1, t


def main():
    for line in sys.stdin:
        print line.strip()
        t = Tree.parse(line.strip())
        print t
        print


if __name__ == "__main__":
    main()
