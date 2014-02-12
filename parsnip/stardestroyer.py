"""
Find `from [module] import *` statements and replace them with specific
imports.
"""

import __builtin__
import ast
from collections import defaultdict
import imp
import sys

from . import Module


MAGIC_NAMES = {'__file__', '__package__', '__path__', '__name__', '__loader__'}


# This doesn't really work as well as I had hoped. Modules may be present only
# as object code, so parsing them as an AST may or may not be possible. It
# might still be possible to get the job done by inspecting the object code,
# but that complicates things.
def get_module(name, pythonpath):
    f, path, desc = imp.find_module(name, pythonpath)
    src = f.read()
    f.close()
    return Module.parse(src, path)


def destroy_stars(path, pythonpath=sys.path):
    mod = Module.parse_path(path)
    namespace_nodes = [node for node in mod.nodes if node.is_namespace]
    #ast_map = {node.ast_node: node for node in mod.nodes}
    modules = {}
    star_imports = defaultdict(list)
    symbols = []

    import_star_statements = [
        node for node in mod.nodes
        if isinstance(node.ast_node, ast.ImportFrom) and
        node.ast_node.names[0].name == '*'
    ]

    for statement in reversed(import_star_statements):
        name = statement.ast_node.module
        print name
        imported_module = modules.get(name)
        if imported_module is None:
            imported_module = get_module(name, pythonpath)
            modules[name] = imported_module
        namespace_node = statement.ancestor(lambda node: node.is_namespace)
        star_imports[namespace_node].append(imported_module)

    for node in namespace_nodes:
        if any(namespace.has_import_star() for namespace in node.namespaces):
            for symbol in node.namespaces[0].get_symbols():
                name = symbol.get_name()
                if name in MAGIC_NAMES or hasattr(__builtin__, name):
                    continue

                if (symbol.is_local() or
                    symbol.is_parameter() or
                    symbol.is_free() or
                    symbol.is_imported() or
                    symbol.is_declared_global()):
                    continue

                found = False
                for namespace in node.namespaces[1:]:
                    try:
                        parent_symbol = namespace.lookup(name)
                        if parent_symbol.is_local():
                            found = True
                            break
                    except KeyError:
                        pass

                if found:
                    continue

                resolved_symbol = None
                for source in star_imports[node]:
                    try:
                        resolved_symbol = (source, source.symtable.lookup(name))
                        break
                    except KeyError:
                        pass
                assert resolved_symbol is not None, \
                        'Unable to resolve symbol {0} for {1}'.format(symbol,
                                                                      node)
                symbols.append((name, symbol, resolved_symbol))

    return symbols
    # TODO: don't just return symbols, manipulate statements
