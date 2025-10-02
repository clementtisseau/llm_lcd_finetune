from abc import ABC, abstractmethod

class Expression(ABC):
    @abstractmethod
    def __str__(self):
        pass

class Number(Expression):
    def __init__(self, n):
        self.n = n
    def __str__(self):
        return str(self.n)

class BinaryExpression(Expression):
    def __init__(self, left, op, right):
        self.left = left        
        self.op = op
        self.right = right
    def __str__(self):
        return f"{self.left} {self.op} {self.right}"

class ParenthesizedExpression(Expression):
    def __init__(self, inner):
        self.inner = inner
    def __str__(self):
        return f"({self.inner})"
    

# ---- AST utilities ----
def ast_depth(node):
    if isinstance(node, Number):
        return 1
    if isinstance(node, ParenthesizedExpression):
        # Parens don't change structural depth for our purposes
        return ast_depth(node.inner)
    if isinstance(node, BinaryExpression):
        return 1 + max(ast_depth(node.left), ast_depth(node.right))
    raise TypeError(type(node))

def count_operands(node):
    if isinstance(node, Number):
        return 1
    if isinstance(node, ParenthesizedExpression):
        return count_operands(node.inner)
    if isinstance(node, BinaryExpression):
        return count_operands(node.left) + count_operands(node.right)
    raise TypeError(type(node))

def count_operators(node):
    if isinstance(node, Number):
        return 0
    if isinstance(node, ParenthesizedExpression):
        return count_operators(node.inner)
    if isinstance(node, BinaryExpression):
        return 1 + count_operators(node.left) + count_operators(node.right)
    raise TypeError(type(node))

def eval_expr(node):
    if isinstance(node, Number):
        return float(node.n)
    if isinstance(node, ParenthesizedExpression):
        return eval_expr(node.inner)
    if isinstance(node, BinaryExpression):
        a = eval_expr(node.left)
        b = eval_expr(node.right)
        if node.op == "+":
            return a + b
        elif node.op == "-":
            return a - b
        elif node.op == "*":
            return a * b
        elif node.op == "/":
            return a / b
        else:
            raise ValueError(f"Unknown op {node.op}")
    raise TypeError(type(node))

def eval_expr_safe(node):
    """Return (value, ok). Skips divide-by-zero trees."""
    try:
        return eval_expr(node), True
    except ZeroDivisionError:   # Below we choose Number in [1,100) so it is useless here
        return None, False


# ---- Generate random expression ----
from random import random, randint, choices

OPS = ["+", "-", "*", "/"]
OP_WEIGHTS = [0.48, 0.27, 0.20, 0.05]  # +, -, *, /  (tweak to taste)

def weighted_op():
    return choices(OPS, weights=OP_WEIGHTS, k=1)[0]

def random_expression(prob, depth=0):
    if random() > prob:
        return Number(randint(1, 100))

    # Small chance to add parentheses — decreasing with depth
    want_parens = random() < min(0.75, 0.75 / (1 + depth))
    if want_parens:
        inner = random_expression(prob / 1.20, depth + 1)
        # Avoid pointless or nested parens
        if isinstance(inner, Number) or isinstance(inner, ParenthesizedExpression):
            return inner
        return ParenthesizedExpression(inner)

    # Otherwise, build a binary op (with biased operator choice)
    left  = random_expression(prob / 1.35, depth + 1)
    op    = weighted_op()
    right = random_expression(prob / 1.35, depth + 1)

    return BinaryExpression(left, op, right)

# Convert infix notation in RPN
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
def infix_minimal(node):
    def rec(n):
        if isinstance(n, Number):
            return str(n.n), 3  # leaf has highest "prec"
        if isinstance(n, ParenthesizedExpression):
            # flatten; we'll add only necessary parens ourselves
            return rec(n.inner)
        if isinstance(n, BinaryExpression):
            op = n.op
            prec = PRECEDENCE[op]
            ls, lp = rec(n.left)
            rs, rp = rec(n.right)

            if isinstance(n.left, BinaryExpression) and lp < prec:
                ls = f"({ls})"
            if isinstance(n.right, BinaryExpression) and (rp < prec or (rp == prec and op in ("-", "/"))):
                rs = f"({rs})"

            return f"{ls} {op} {rs}", prec
        raise TypeError(type(n))
    s, _ = rec(node)
    return s

def rpn_tokens(node):
    if isinstance(node, Number):
        yield str(node.n)
    elif isinstance(node, ParenthesizedExpression):
        yield from rpn_tokens(node.inner)
    elif isinstance(node, BinaryExpression):
        yield from rpn_tokens(node.left)
        yield from rpn_tokens(node.right)
        yield node.op
    else:
        raise TypeError(type(node))

def to_rpn(node, sep=" "):
    return sep.join(rpn_tokens(node))

# ---- Create dataset ----
import json

def create_dataset(n_samples, path=f"data/dataset.jsonl", require_value=True):
    """Write n_samples JSONL rows with metrics.
       If require_value=True, skip expressions that raise ZeroDivisionError.
    """
    with open(path, "w") as file:
        i = 0
        while i < n_samples:
            expr = random_expression(1)  # build an AST
            value, ok = eval_expr_safe(expr)
            if (require_value and not ok) or (isinstance(expr, Number)):
                continue  # try another sample

            infix = infix_minimal(expr)
            rpn = to_rpn(expr)
            rec = {
                "id": f"{i+1:04d}",
                "infix": infix,                         # your __str__ output
                "rpn": rpn,                             # RPN from AST
                "depth": ast_depth(expr),               # tree depth
                "num_operands": count_operands(expr),   # leaf Number count
                "num_operators": count_operators(expr),
                "value": value,                         # semantic value (float) or None
            }
            file.write(json.dumps(rec) + "\n")
            i += 1

if __name__ == "__main__":
    create_dataset(100)