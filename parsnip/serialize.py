import ast
from collections import namedtuple
from functools import partial
import json
try:
    from cStringIO import StringIO
    StringIO
except ImportError:
    from StringIO import StringIO
import tokenize
from xml.etree.ElementTree import (
    ElementTree,
    TreeBuilder,
)


types = namedtuple('types', 'AST TOKENS SYMTABLE DEFAULT ALL')(
    AST=1,
    TOKENS=2,
    SYMTABLE=4,
    DEFAULT=1 | 2 | 4,  # no BYTECODES by default, since it requires compiling
    ALL=1 | 2 | 4,
)

# TODO: add handling for parse tree and disassembled byte code


class ModuleSerializer(ast.NodeVisitor):

    @classmethod
    def serialize(cls, module):
        serializer = cls(module)
        serializer.run()
        return serializer

    def __init__(self, module, flags=types.DEFAULT):
        self.module = module
        self.flags = flags

    def run(self):
        if self.flags & types.AST:
            self.run_ast()
        if self.flags & types.TOKENS:
            self.run_tokens()
        if self.flags & types.SYMTABLE:
            self.run_symtable()

    def run_ast(self):
        self.visit(self.module.ast)

    def run_tokens(self):
        for offset, token in enumerate(self.module.tokens):
            self.process_token(self.normalize_token(token), offset)

    def run_symtable(self):
        pass  # TODO

    def normalize_token(self, token):
        return token + (None,) * (5 - len(token))

    def get_root(self):
        raise NotImplementedError()

    def process_token(self, token, offset):
        raise NotImplementedError()

    def start_ast(self, ast_node):
        raise NotImplementedError()

    def end_ast(self, ast_node):
        raise NotImplementedError()

    def start_list_field(self, field, value):
        raise NotImplementedError()

    def end_list_field(self, field, value):
        raise NotImplementedError()

    def process_string_field(self, field, value):
        raise NotImplementedError()

    def process_number_field(self, field, value):
        raise NotImplementedError()

    def process_null_field(self, field):
        raise NotImplementedError()

    def process_ast_field(self, field, value):
        self.visit(value)

    def process_field(self, field, value):
        if value is None:
            self.process_null_field(field)
        elif isinstance(value, list):
            self.start_list_field(field, value)
            for item in value:
                self.visit(item)
            self.end_list_field(field, value)
        elif isinstance(value, basestring):
            self.process_string_field(field, value)
        elif isinstance(value, (int, long, float)):
            self.process_number_field(field, value)
        else:
            assert isinstance(value, ast.AST)
            self.process_ast_field(field, value)

    def generic_visit(self, node):
        self.start_ast(node)
        for field, value in ast.iter_fields(node):
            self.process_field(field, value)
        self.end_ast(node)


class DictModuleSerializer(ModuleSerializer):

    def __init__(self, module, flags=types.DEFAULT):
        super(DictModuleSerializer, self).__init__(module, flags)
        self.tokens = []
        self.stack = []
        self.root = {}
        self.handlers = []

    def run_ast(self):
        self.handlers.append(partial(self.root.__setitem__, 'ast'))
        super(DictModuleSerializer, self).run_ast()
        self.handlers.pop()

    def run_tokens(self):
        self.root['tokens'] = self.tokens
        super(DictModuleSerializer, self).run_tokens()

    def get_root(self):
        return self.root

    def process_token(self, token, offset):
        ttype, tval, tstart, tend, tline = token
        t_dict = {
            'type_name': tokenize.tok_name[ttype],
            'type': ttype,
            'value': tval,
            'offset': offset,
        }
        if tstart is not None:
            t_dict.update({
                'start_line': tstart[0],
                'start_col': tstart[1],
            })
        if tend is not None:
            t_dict.update({
                'end_line': tend[0],
                'end_col': tend[1],
            })
        if tline is not None:
            t_dict['line'] = tline
        self.tokens.append(t_dict)

    def process_field(self, field, value):
        self.handlers.append(partial(self.stack[-1].__setitem__, field))
        super(DictModuleSerializer, self).process_field(field, value)
        self.handlers.pop()

    def handle(self, value):
        self.handlers[-1](value)

    def start_ast(self, node):
        node_dict = {
            '__type__': type(node).__name__,
        }
        line = getattr(node, 'lineno', None)
        if line is not None:
            node_dict['__line__'] = line
        col = getattr(node, 'col_offset', None)
        if col is not None:
            node_dict['__col__'] = col
        if self.flags & types.TOKENS:
            n = self.module.ast_map.get(node)
            if n is not None and n.token_start < n.token_end:
                node_dict['__tokens__'] = [n.token_start, n.token_end]
        self.handle(node_dict)
        self.stack.append(node_dict)

    def end_ast(self, ast_node):
        self.stack.pop()

    def start_list_field(self, field, value):
        items = []
        self.handle(items)
        self.handlers.append(items.append)

    def end_list_field(self, field, value):
        self.handlers.pop()

    def process_string_field(self, field, value):
        self.handle(value)

    def process_number_field(self, field, value):
        self.handle(value)

    def process_null_field(self, field):
        self.handle(None)

    def json(self):
        return json.dumps(self.root)

    def write_json(self, path):
        json_s = self.json()
        if hasattr(path, 'write'):
            path.write(json_s)
        else:
            with open(path, 'w') as f:
                f.write(json_s)


ElementBuilder = namedtuple('ElementBuilder', 'tag attrs children')


class EtreeModuleSerializer(ModuleSerializer):

    def __init__(self, module, flags=types.DEFAULT):
        super(EtreeModuleSerializer, self).__init__(module, flags)
        self.stack = [ElementBuilder('module', {}, [])]
        self.builder = TreeBuilder()
        self.etree = None

    def run_ast(self):
        ast_el = ElementBuilder('ast', {}, [])
        self.add_child(ast_el)
        self.stack.append(ast_el)
        super(EtreeModuleSerializer, self).run_ast()
        self.stack.pop()

    def run_tokens(self):
        tokens_el = ElementBuilder('tokens', {}, [])
        self.add_child(tokens_el)
        self.stack.append(tokens_el)
        super(EtreeModuleSerializer, self).run_tokens()
        self.stack.pop()

    def get_root(self):
        if self.etree is None:
            assert len(self.stack) == 1
            self._build(self.stack[0])
            self.etree = ElementTree(self.builder.close())
        return self.etree

    def process_token(self, token, offset):
        ttype, tval, tstart, tend, tline = token
        attrs = {
            'type': repr(ttype),
            'value': repr(tval).lstrip('u')[1:-1],
            'offset': repr(offset),
        }
        if tstart is not None:
            attrs.update({
                'start_line': repr(tstart[0]),
                'start_col': repr(tstart[1]),
            })
        if tend is not None:
            attrs.update({
                'end_line': repr(tend[0]),
                'end_col': repr(tend[1]),
            })
        if tline is not None:
            attrs['line'] = repr(tline).lstrip('u')[1:-1]
        el = ElementBuilder(tokenize.tok_name[ttype], attrs, [])
        self.add_child(el)

    def add_attr(self, attr, value):
        self.stack[-1].attrs[attr] = value

    def add_child(self, child):
        self.stack[-1].children.append(child)

    def start_ast(self, node):
        el = ElementBuilder(type(node).__name__, {}, [])
        line = getattr(node, 'lineno', None)
        if line is not None:
            el.attrs['line'] = repr(line)
        col = getattr(node, 'col_offset', None)
        if col is not None:
            el.attrs['col'] = repr(col)
        if self.flags & types.TOKENS:
            n = self.module.ast_map.get(node)
            if n is not None:
                if n.token_start < n.token_end:
                    el.attrs['token_start'] = repr(n.token_start)
                    el.attrs['token_end'] = repr(n.token_end)
        self.add_child(el)
        self.stack.append(el)

    def end_ast(self, node):
        self.stack.pop()

    def start_list_field(self, field, value):
        el = ElementBuilder(field, {}, [])
        self.add_child(el)
        self.stack.append(el)

    def end_list_field(self, field, value):
        self.stack.pop()

    def process_ast_field(self, field, value):
        el = ElementBuilder(field, {}, [])
        self.add_child(el)
        self.stack.append(el)
        super(EtreeModuleSerializer, self).process_ast_field(field, value)
        self.stack.pop()

    def process_string_field(self, field, value):
        self.add_attr(field, value)

    def process_number_field(self, field, value):
        self.add_attr(field, repr(value))

    def process_null_field(self, field):
        self.add_child(ElementBuilder(field, {}, []))

    def _build(self, el):
        self.builder.start(el.tag, el.attrs)
        for child in el.children:
            self._build(child)
        self.builder.end(el.tag)

    def write_xml(self, path):
        etree = self.get_root()
        if hasattr(path, 'write'):
            etree.write(path, 'utf-8')
        else:
            with open(path, 'w') as f:
                etree.write(f, 'utf-8')

    def xml_string(self):
        f = StringIO()
        self.write_xml(f)
        f.seek(0)
        return f.read()

    def generic_visit(self, node):
        super(EtreeModuleSerializer, self).generic_visit(node)
