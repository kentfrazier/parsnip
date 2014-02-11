import ast
from collections import Sequence
from functools import partial
import json
from xml.etree.ElementTree import TreeBuilder


class Walker(ast.NodeVisitor):
    def __init__(self, node):
        if isinstance(node, Sequence) and not isinstance(node, basestring):
            self.root = []
            self.handlers = [self.root.append]
            for n in node:
                self.visit(n)
        else:
            self.root = None
            self.handlers = [partial(setattr, self, 'root')]
            self.visit(node)

    def _process(self, value, handler):
        stack = self.handlers
        if isinstance(value, list):
            seq = []
            handler(seq)
            for item in value:
                self._process(item, seq.append)
        else:
            assert isinstance(value, ast.AST), "Unexpected type: {0}".format(type(value))
            stack.append(handler)
            self.visit(value)
            stack.pop()

    def generic_visit(self, node):
        current = {
            '__ast__': node,
            '__type__': type(node).__name__,
        }

        lineno = getattr(node, 'lineno', None)
        if lineno is not None:
            current['lineno'] = lineno
        col_offset = getattr(node, 'col_offset', None)
        if col_offset is not None:
            current['col_offset'] = col_offset

        # TODO: maybe try to see if this could be detected as duplicate code
        # and refactored into a for loop using the tool.
        # Finished product should be something like:
        #     for gensym0 in ('lineno', 'col_offset'):
        #         gensym1 = getattr(node, gensym0, None)
        #         if gensym1 is not None:
        #             current[gensym0] = gensym1

        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            current['docstring'] = ast.get_docstring(node)

        for field, value in ast.iter_fields(node):
            if value is None or isinstance(value, (str, unicode, int, float)):
                current[field] = value
            else:
                assert isinstance(value, (ast.AST, list)), "Unexpected type for field '{0}': {1}".format(field, type(value))
                self._process(value, partial(current.__setitem__, field))

        self.handlers[-1](current)

    @classmethod
    def from_source_string(self, src, filename='<string>'):
        return Walker(ast.parse(src))

    @classmethod
    def from_source_path(self, path):
        with open(path) as f:
            return Walker.from_source_string(f.read(), path)

    def to_json(self):
        def clean(item):
            if isinstance(item, list):
                return [clean(el) for el in item]
            elif isinstance(item, dict):
                return dict(
                    (k, clean(v)) for k, v in item.iteritems() if k != '__ast__'
                )
            else:
                return item

        return json.dumps(clean(self.root))

    def to_xml(self):
        builder = TreeBuilder()

        def process_node(node):
            node_type = None
            simple_attrs = {}
            complex_attrs = {}

            for key, value in node.iteritems():
                if key == '__ast__':
                    continue
                elif key == '__type__':
                    node_type = value
                elif key == '__docstring__':
                    simple_attrs['docstring'] = repr(value)
                elif value is None or isinstance(value, (dict, list)):
                    complex_attrs[key] = value
                else:
                    simple_attrs[key] = repr(value)

            builder.start(node_type, simple_attrs)
            for subkey, subvalue in complex_attrs.iteritems():
                builder.start(subkey, {})
                if isinstance(subvalue, list):
                    for item in subvalue:
                        process_node(item)
                elif isinstance(subvalue, dict):
                    process_node(subvalue)
                else:
                    pass  # None, so should be an empty element
                builder.end(subkey)
            builder.end(node_type)

        if isinstance(self.root, list):
            builder.start('nodelist', {})
            for n in self.root:
                process_node(n)
            builder.end('nodelist')
        else:
            process_node(self.root)

        return builder.close()


