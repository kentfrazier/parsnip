import ast
from collections import Sequence
from functools import partial
import json
from xml.etree.ElementTree import TreeBuilder


def no_underscores(node, attr):
    return not attr.startswith('_')

class Walker(ast.NodeVisitor):
    def __init__(self, attr_filter=no_underscores):
        self.root = None
        self.attr_filter = attr_filter
        self.handlers = [partial(setattr, self, 'root')]

    def _process(self, value, handler):
        stack = self.handlers
        if isinstance(value, ast.AST):
            stack.append(handler)
            self.visit(value)
            stack.pop()
        elif isinstance(value, Sequence) and not isinstance(value, basestring):
            seq = []
            handler(seq)
            for item in value:
                self._process(item, seq.append)
        else:
            handler(value)

    def generic_visit(self, node):
        current = {
            '__type__': type(node).__name__,
        }

        for attr in dir(node):
            if self.attr_filter(node, attr):
                self._process(getattr(node, attr),
                              partial(current.__setitem__, attr))

        self.handlers[-1](current)

    def parse_source(self, src):
        tree = ast.parse(src)
        self.visit(tree)

    def to_json(self):
        return json.dumps(self.root)

    def to_xml(self):
        builder = TreeBuilder()

        def process_node(node):
            node_type = None
            simple_attrs = {}
            complex_attrs = {}

            for key, value in node.iteritems():
                if key == '__type__':
                    node_type = value
                elif isinstance(value, (dict, list)):
                    complex_attrs[key] = value
                else:
                    simple_attrs[key] = value

            builder.start(
                node_type,
                dict((k, unicode(v)) for k, v in simple_attrs.iteritems()),
            )
            for subkey, subvalue in complex_attrs.iteritems():
                builder.start(subkey, {})
                if isinstance(subvalue, list):
                    for item in subvalue:
                        process_node(item)
                else:
                    process_node(subvalue)
                builder.end(subkey)
            builder.end(node_type)

        process_node(self.root)

        return builder.close()

