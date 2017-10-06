#!/usr/bin/env python

"""
RST tree structure
"""

import sys
from collections import defaultdict


class RSTTree(object):

    __slots__ = "nodetype", "loc", "rel2par", "children", "text", "postorder_id", "leftidx", "rightidx"

    # possible tree node types
    NodeTypes = ("Root", "Nucleus", "Satellite")

    def __init__(self, nodetype, loc, rel2par, text, children):
        # node type can be: Root, Nucleus, Satellite
        self.nodetype = nodetype
        # loc is either a leaf number or a span
        self.loc = loc
        self.rel2par = rel2par  # None for Root
        self.text = text
        self.children = children
        self.leftidx = None
        self.rightidx = None

    def pretty_str(self, indent=0):
        content = ""
        ind = "  " * indent
        if type(self.loc) == tuple:
            # span => multiple lines
            content = "(span %d %d)" % (self.loc[0], self.loc[1])
            if self.rel2par is not None:
                content += " (rel2par %s)" % self.rel2par
            firstline = "%s( %s %s" % (ind, self.nodetype, content)
            lastline = "%s)" % ind
            return "\n".join([firstline] + [c.pretty_str(indent+1) for c in self.children] + [lastline])
        else:
            # leaf => one line
            content = "(leaf %d) (rel2par %s) (text _!%s_!)" % (self.loc, self.rel2par, self.text)
            return "%s(%s %s )" % (ind, self.nodetype, content)

    def __str__(self):
        content = ""
        if type(self.loc) == tuple:
            content = "(span %d %d)" % (self.loc[0], self.loc[1])
            if self.rel2par is not None:
                content += " (rel2par %s)" % self.rel2par
            firstline = "( %s %s" % (self.nodetype, content)
            lastline = ")"
            return " ".join([firstline] + [str(c) for c in self.children] + [lastline])
        else:
            content = "(leaf %d) (rel2par %s) (text _!%s_!)" % (self.loc, self.rel2par, self.text)
            return "(%s %s )" % (self.nodetype, content)

    def get_sentence(self):
        if self.text is not None:
            return self.text.split()
        elif len(self.children) > 0:
            ret = []
            for c in self.children:
                ret += c.get_sentence()
            return ret
        else:
            return None

    def postorder_visit(self, func):
        for c in self.children:
            c.postorder_visit(func)
        func(self)

    def mark_postorder_id(self, idx=0):
        for c in self.children:
            idx = c.mark_postorder_id(idx)
        self.postorder_id = idx
        return idx+1

    @staticmethod
    def parse(line):
        line += " "
        _, t = RSTTree._parse(line, 0)
        return t

    @staticmethod
    def _parse(line, index):
        # returns a RSTTree if there is an RST tree starting at index, otherwise returns a tuple
        assert line[index] == "(", "Invalid tree string %s at %d" (line, index)
        index += 1
        info = []
        while line[index] == " " and index < len(line)-1:
            index += 1
        while line[index] != ")":
            if line[index] == "(":
                index, t = RSTTree._parse(line, index)
                info.append(t)
            else:
                rpos = None
                if line[index:index+2] == "_!":
                    rpos = line.find("_!", index+2)
                    info.append(line[index+2:rpos])
                    index = rpos + 2
                else:
                    rpos = min(line.find(" ", index), line.find(")", index))
                    info.append(line[index:rpos])
                    index = rpos
            while line[index] == " " and index < len(line)-1:
                index += 1

        assert line[index] == ")", "Invalid tree string %s at %d" % (line, index)
        index += 1
        if info[0] in RSTTree.NodeTypes:
            nodetype = info[0]
            props = defaultdict(lambda : None)
            children = []
            for i in info[1:]:
                if type(i) == tuple:
                    props[i[0]] = tuple(i[1:]) if len(i) > 2 else i[1]
                else:
                    children.append(i)
            return index, RSTTree(nodetype, props["loc"], props["rel2par"], props["text"], children)
        elif info[0] == "span":
            return index, ("loc", int(info[1]), int(info[2]))
        elif info[0] == "leaf":
            return index, ("loc", int(info[1]))
        elif info[0] == "rel2par":
            return index, ("rel2par", info[1])
        elif info[0] == "text":
            text = info[1].replace("<P>","")
            return index, ("text", text)
        else:
            assert False, "Wrong infomation extracted %s" % str(info)

    def binarize(self):
        """Following the tradition to binarize the non-binary branches into
        cascades of right-branching binary branches
        """
        if len(self.children) > 2:
            for c in self.children:
                assert c.nodetype == "Nucleus"
            left_node = self.children[0]
            new_node = RSTTree(nodetype="Nucleus",
                               loc=((self.children[1].loc[0]
                                     if type(self.children[1].loc) is tuple
                                     else self.children[1].loc),
                                    (self.children[-1].loc[1]
                                     if type(self.children[-1].loc) is tuple
                                     else self.children[-1].loc)),
                               rel2par=left_node.rel2par,
                               text=None,
                               children=self.children[1:])
            self.children = [left_node, new_node]
        for c in self.children:
            c.binarize()

    def compare_with(self, other):
        if not (self.nodetype == other.nodetype and self.loc == other.loc and \
                self.rel2par == other.rel2par and self.text == other.text and \
                len(self.children) == len(other.children)):
            print "difference\n>>>>\n%s\n<<<<\n%s" % (self.pretty_str(), other.pretty_str())
        else:
            for c1, c2 in zip(self.children, other.children):
                c1.compare_with(c2)


if __name__ == "__main__":
    lines = []
    for line in sys.stdin:
        lines.append(line.strip())
        if line == ")\n":
            rsttree = RSTTree.parse(" ".join(lines))
            print rsttree.pretty_str()
            rsttree.binarize()
            print rsttree.pretty_str()
            lines = []
