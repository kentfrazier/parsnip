import __future__
import ast
from collections import defaultdict
try:
    from cStringIO import StringIO
    StringIO  # calm down, pyflakes
except ImportError:
    from StringIO import StringIO
import symtable
import tokenize


class Module(object):

    def __init__(self, source, path='<string>'):
        self.source = source
        self.path = path
        self.ast = ast.parse(source, path)
        self.symtable = symtable.symtable(source, path, 'exec')
        self.tokens = tokenize_string(source)
        cw = ChainWalker(self.ast)
        self.nodes = cw.nodes
        TokenAssociator(self.nodes, self.tokens)

    @classmethod
    def parse_path(cls, path):
        with open(path) as f:
            return cls.parse(f.read(), path)

    @classmethod
    def parse(cls, source, path='<string>'):
        return cls(source, path)


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
                if self.token_match(node):
                    success = True
                    break
                else:
                    node.associate_token(node.token_end)
            if not success:
                print i, 'FAILURE!'
                print '  ', node
                print ast.dump(node.ast_node)
                print '  ', tokenize.untokenize(map(_decontextify_token, self.tokens[node.token_start:node.token_end]))

    def find_adjacent_token(self, start, step, target_token):
        offset = start + step
        token = self.decontextified_tokens[offset]
        while token in {tokenize.NL, tokenize.COMMENT}:
            offset += step
            token = self.decontextified_tokens[offset]

        if token == target_token:
            return offset

        return None

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
        prev_offset = self.find_adjacent_token(node.token_start, -1, LSQUARE)
        assert prev_offset is not None, "Couldn't find ["
        node.associate_token(prev_offset)

        end = node.token_end
        new_ast = None
        while new_ast is None and end >= 0:
            end = self.decontextified_tokens.index(RSQUARE, end)
            node.associate_token(end)
            tokens = self.decontextified_tokens[node.token_start:node.token_end]
            new_ast = self.match_AST(node, tokens)
        return new_ast

    def match_Tuple(self, node, tokens):
        prev_offset = self.find_adjacent_token(node.token_start, -1, LPAREN)
        if prev_offset is not None:
            node.associate_token(prev_offset)
            next_offset = self.find_adjacent_token(node.token_end - 1, 1, RPAREN)
            assert next_offset is not None, "Couldn't find )"
            node.associate_token(next_offset)
            tokens = self.decontextified_tokens[node.token_start:node.token_end]
        return self.match_AST(node, tokens)

    def match_expr(self, node, tokens):
        return self.match_AST(node,
                              #[LPAREN] + self.fix_brackets(tokens) + [RPAREN])
                              [LPAREN] + tokens + [RPAREN])

    def token_match(self, node):
        tokens = self.decontextified_tokens[node.token_start:node.token_end]
        for ast_type in type(node.ast_node).__mro__:
            method = getattr(self, 'match_' + ast_type.__name__, None)
            if method is not None:
                break

        new_ast = method(node, tokens)

        return ast_equal(node.ast_node, new_ast)

