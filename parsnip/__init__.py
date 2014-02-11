import __future__
import ast
from collections import (
    defaultdict,
    Sequence,
)
from functools import partial
import json
try:
    from cStringIO import StringIO
    StringIO  # calm down, pyflakes
except ImportError:
    from StringIO import StringIO
import symtable
import tokenize
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


class Module(object):

    def __init__(self, source, path='<string>'):
        self.source = source
        self.path = path
        self.ast = ast.parse(source, path)
        self.symtable = symtable.symtable(source, path, 'exec')
        self.tokens = tokenize_string(source)

    @classmethod
    def from_path(self, path):
        with open(path) as f:
            return Module(f.read(), path)


class Node(object):

    def __init__(self, level, ast_node, parent, compiler_flags=0):
        self.compiler_flags = compiler_flags
        self.level = level
        self.line = getattr(ast_node, 'lineno', None)
        self.col = getattr(ast_node, 'col_offset', None)

        self.ast_node = ast_node
        self.parent = parent

        self.token_start = float('Inf')
        self.token_end = float('-Inf')

    def __repr__(self):
        if self.parent is None:
            return '{ast_type}[{level}]'.format(
                ast_type=type(self.ast_node).__name__,
                level=self.level,
            )
        else:
            return '{ast_type}[{level}] -> {parent!r}'.format(
                ast_type=type(self.ast_node).__name__,
                level=self.level,
                parent=self.parent,
            )

    def associate_token(self, offset):
        self.token_start = min(offset, self.token_start)
        self.token_end = max(offset + 1, self.token_end)

        if self.parent is not None:
            self.parent.associate_token(offset)

    def has_pos(self):
        return self.line is not None and self.col is not None


class ChainWalker(ast.NodeVisitor):

    def __init__(self, ast_node):
        self.compiler_flags = ast.PyCF_ONLY_AST
        self.levels = {}
        self.level = 0
        self.nodes = []
        self.visit(ast_node)

    def visit_ImportFrom(self, ast_node):
        # If there are __future__ imports in effect, they affect parsing,
        # so we have to make sure to pass them along later.
        if ast_node.module == '__future__':
            for alias in ast_node.names:
                feature = getattr(__future__, alias.name)
                self.compiler_flags |= feature.compiler_flag
        self.generic_visit(ast_node)

    def generic_visit(self, ast_node):
        node = Node(level=self.level,
                    ast_node=ast_node,
                    parent=self.levels.get(self.level - 1, None),
                    compiler_flags=self.compiler_flags)
        self.levels[self.level] = node
        self.nodes.append(node)
        self.level += 1
        super(ChainWalker, self).generic_visit(ast_node)
        self.level -= 1


def ast_equal(node1, node2):
    # don't bail if they disagree on string type
    if isinstance(node1, basestring) and isinstance(node2, basestring):
        return node1 == node2
    elif type(node1) != type(node2):
        return False
    elif isinstance(node1, ast.AST):
        return all(
            ast_equal(value, getattr(node2, field)) for field, value
            in ast.iter_fields(node1)
            if field != 'ctx'  # context won't be accurate with surrounding code
        )
    elif isinstance(node1, list):
        return all(ast_equal(val1, val2) for val1, val2 in zip(node1, node2))
    else:
        return node1 == node2


def tokenize_string(source):
    f = StringIO()
    f.write(source)
    f.seek(0)
    tokens = tuple(tokenize.generate_tokens(f.readline))
    f.close()
    return tokens


def _decontextify_token(token):
    return token[:2]


TRY_PREFIX = map(_decontextify_token, tokenize_string('try: pass\n'))
LPAREN = (tokenize.OP, '(')
RPAREN = (tokenize.OP, ')')
LSQUARE = (tokenize.OP, '[')
RSQUARE = (tokenize.OP, ']')

BRACKET_MAP = {
    '}': '{',
    ']': '[',
    ')': '(',
}
OPEN_BRACKETS = set(BRACKET_MAP.itervalues())
CLOSE_BRACKETS = set(BRACKET_MAP)

class TokenAssociator(object):

    def __init__(self, nodes, tokens):
        self.nodes = nodes
        self.tokens = tokens
        self.decontextified_tokens = map(_decontextify_token, tokens)
        self.ordered_nodes = sorted(
            (node for node in nodes if node.has_pos()),
            key=lambda node: (-node.level, node.line, node.col)
        )
        self.token_map = self.build_token_map(tokens)

        self.associate_initial_tokens()
        self.find_token_extents()

    def build_token_map(self, tokens):
        token_map = defaultdict(dict)
        for i, token in enumerate(tokens):
            t_type, _, t_start, t_end, _ = token
            start_line, start_col = t_start
            end_line, end_col = t_end

            # very ugly, but the AST parser does weird things with lineno and
            # col_offset for multiline strings
            if t_type == tokenize.STRING and start_line != end_line:
                line = end_line
                col = -1
            else:
                line = start_line
                col = start_col

            token_map[line][col] = i
        return dict(token_map)

    def associate_initial_tokens(self):
        for node in self.ordered_nodes:
            offset = self.token_map[node.line][node.col]
            node.associate_token(offset)

    def find_token_extents(self):
        num_tokens = len(self.tokens)
        for i, node in enumerate(self.ordered_nodes):
            success = False
            while node.token_end < num_tokens:
                #if i == 1682:
                #    import ipdb; ipdb.set_trace()
                if self.token_match(node):
                    success = True
                    break
                else:
                    node.associate_token(node.token_end)
            if success:
                print i, 'SUCCESS!'
                #print '  ', node
                #print '  ', tokenize.untokenize(map(_decontextify_token, tokens[node.token_start:node.token_end]))
            else:
                print i, 'FAILURE!'
                print '  ', node
                print ast.dump(node.ast_node)
                print '  ', tokenize.untokenize(map(_decontextify_token, self.tokens[node.token_start:node.token_end]))

    def fix_brackets(self, tokens):
        """
        Fix bracket mismatches in token stream.
        """
        bracket_stack = []
        to_add = []
        for token in tokens:
            if token[0] == tokenize.OP:
                op = token[1]
                if op in OPEN_BRACKETS:
                    bracket_stack.append(op)
                elif op in CLOSE_BRACKETS:
                    opener = BRACKET_MAP[op]
                    if bracket_stack and bracket_stack[-1] == opener:
                        bracket_stack.pop()
                    else:
                        to_add.append((tokenize.OP, opener))
        to_add.reverse()
        return to_add + tokens

    def match_AST(self, node, tokens):
        source = tokenize.untokenize(tokens)

        try:
            # TODO: try 'eval' mode here. It might work better.
            new_ast = compile(
                source,
                '<string>',
                'exec',
                node.compiler_flags,
                1,
            )
        except SyntaxError:
            return None

        new_ast = new_ast.body[0]  # unwrap from Module
        if isinstance(new_ast, ast.Expr) and not isinstance(node.ast_node, ast.Expr):
            new_ast = new_ast.value  # unwrap from Expr if needed

        return new_ast

    def match_ExceptHandler(self, node, tokens):
        new_ast = self.match_AST(node, TRY_PREFIX + tokens)
        if new_ast is not None:
            new_ast = new_ast.handlers[0]
        return new_ast

    def match_If(self, node, tokens):
        if tokens[0][1] != 'if':
            if tokens[0][1] == 'elif':
                tokens[0] = (tokenize.NAME, 'if')
            else:
                prev_token = self.tokens[node.token_start - 1]
                assert prev_token[1] == 'elif', \
                        'unexpected token: {0}'.format(prev_token)
                node.associate_token(node.token_start - 1)
                tokens = [_decontextify_token(prev_token)] + tokens
        return self.match_AST(node, tokens)

    def match_ListComp(self, node, tokens):
        prev_offset = node.token_start - 1
        while self.decontextified_tokens[prev_offset][0] in {tokenize.NL,
                                                             tokenize.COMMENT}:
            prev_offset -= 1
        prev_token = self.decontextified_tokens[prev_offset]
        assert prev_token == LSQUARE, 'unexpected token: {0}'.format(prev_token)
        node.associate_token(prev_offset)

        end = node.token_end
        new_ast = None
        while new_ast is None and end >= 0:
            end = self.decontextified_tokens.index(RSQUARE, end)
            node.associate_token(end)
            tokens = self.decontextified_tokens[node.token_start:node.token_end]
            new_ast = self.match_AST(node, tokens)
        return new_ast

    def match_expr(self, node, tokens):
        return self.match_AST(node,
                              [LPAREN] + self.fix_brackets(tokens) + [RPAREN])

    def token_match(self, node):
        tokens = self.decontextified_tokens[node.token_start:node.token_end]
        for ast_type in type(node.ast_node).__mro__:
            method = getattr(self, 'match_' + ast_type.__name__, None)
            if method is not None:
                break

        new_ast = method(node, tokens)

        return ast_equal(node.ast_node, new_ast)

