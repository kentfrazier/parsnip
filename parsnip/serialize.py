import ast
from collections import namedtuple
from functools import partial
import json
try:
    from cStringIO import StringIO
    StringIO
except ImportError:
    from StringIO import StringIO
from xml.etree.ElementTree import (
    ElementTree,
    TreeBuilder,
)


types = namedtuple('types', 'AST CST TOKENS BYTECODES DEFAULT ALL')(
    AST=1,
    CST=2,
    TOKENS=4,
    BYTECODES=8,
    DEFAULT=1 | 2 | 4,  # no BYTECODES by default, since it requires compiling
    ALL=1 | 2 | 4 | 8,
)


class ModuleSerializer(ast.NodeVisitor):

    @classmethod
    def serialize(cls, module):
        serializer = cls(module)
        serializer.run()
        return serializer

    def __init__(self, module):
        self.module = module

    def run(self):
        self.visit(self.module.ast)

    def get_root(self):
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

    def __init__(self, module):
        super(DictModuleSerializer, self).__init__(module)
        self.stack = []
        self.root = None
        self.handlers = [self.set_root]

    def set_root(self, root):
        self.root = root

    def get_root(self):
        return self.root

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

    def __init__(self, module):
        super(EtreeModuleSerializer, self).__init__(module)
        self.stack = [ElementBuilder('root', {}, [])]
        self.builder = TreeBuilder()
        self.etree = None

    def get_root(self):
        if self.etree is None:
            assert len(self.stack) == 1
            assert len(self.stack[0].children) == 1
            self._build(self.stack[0].children[0])
            self.etree = ElementTree(self.builder.close())
        return self.etree

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
