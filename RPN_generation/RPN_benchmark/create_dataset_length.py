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
    except ZeroDivisionError:
        return None, False


# ---- Generate random expression (original, probability-driven) ----
from random import random, randint, choices, randrange, shuffle

OPS = ["+", "-", "*", "/"]
OP_WEIGHTS = [0.48, 0.27, 0.20, 0.05]  # +, -, *, /

def weighted_op():
    return choices(OPS, weights=OP_WEIGHTS, k=1)[0]

def random_expression(prob, depth=0):
    if random() > prob and depth > 0:
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


# ---- Length-controlled generator ----
def _maybe_wrap_parens(expr, depth):
    """Optionally wrap in parentheses (doesn't change length)."""
    if isinstance(expr, (Number, ParenthesizedExpression)):
        return expr
    if random() < min(0.75, 0.75 / (1 + depth)):
        return ParenthesizedExpression(expr)
    return expr

def _build_exact_length(length, depth=0):
    """
    Recursively build an expression with exactly `length` tokens (operands + operators).
    length must be odd and >= 1. length == 1 produces a Number.
    """
    if length == 1:
        return Number(randint(1, 100))

    # length >= 3 and odd: split into left_len + right_len + 1(op) = length
    # left_len, right_len must both be odd and >= 1.
    # Choose a random odd left_len in [1, length-2], then derive right_len.
    max_pairs = (length - 2) // 2  # number of odd choices from 1 to length-2 inclusive
    # pick k in [0, max_pairs], then left_len = 2*k + 1
    k = randrange(max_pairs + 1)
    left_len = 2 * k + 1
    right_len = length - 1 - left_len  # guaranteed odd and >= 1

    left = _build_exact_length(left_len, depth + 1)
    right = _build_exact_length(right_len, depth + 1)
    node = BinaryExpression(left, weighted_op(), right)
    return _maybe_wrap_parens(node, depth)

def _pick_length_from_range(min_len, max_len):
    """Pick a random odd length within [min_len, max_len]. Ensures odd and >= 1."""
    if min_len is None or max_len is None:
        raise ValueError("min_len and max_len must both be provided.")
    if min_len > max_len:
        raise ValueError("min_len cannot be greater than max_len.")
    # Ensure there is at least one odd number in range
    if max_len < 1:
        raise ValueError("max_len must be >= 1.")
    # Clamp min to at least 1
    min_len = max(1, min_len)
    # Make sure we will not produce the trivial leaf (length==1) since dataset skips it.
    # We still allow 1 if caller explicitly asks; create_dataset will skip if needed.
    # Build list of odd candidates
    start = min_len if (min_len % 2 == 1) else (min_len + 1)
    if start > max_len:
        raise ValueError("No odd lengths available in the given range.")
    candidates = list(range(start, max_len + 1, 2))
    return choices(candidates, k=1)[0]

def random_expression_with_length(min_len=None, max_len=None, exact_len=None):
    """
    Build an expression with controlled token-length (operands + operators).
    - exact_len: build exactly this odd length (>=1).
    - otherwise: pick an odd length randomly within [min_len, max_len].
    """
    if exact_len is not None:
        if exact_len < 1:
            raise ValueError("exact_len must be >= 1.")
        if exact_len % 2 == 0:
            raise ValueError("exact_len must be odd (operands + operators is always odd).")
        return _build_exact_length(exact_len, depth=0)

    # Range mode
    length = _pick_length_from_range(min_len, max_len)
    return _build_exact_length(length, depth=0)


# ---- Convert infix notation in RPN ----
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

def create_dataset(
    n_samples,
    path="data/dataset.jsonl",
    require_value=True,
    min_len=None,
    max_len=None,
    exact_len=None,
    *,
    start_id=1,
    assign_ids=True,
    shuffle_at_end=False,
    return_records=False,
):
    """Write n_samples JSONL rows with metrics.
       If require_value=True, skip expressions that raise ZeroDivisionError.
       You can constrain expression token-length via min/max or exact_len.

       Extra controls:
       - assign_ids: if True, assign sequential IDs starting at start_id.
       - shuffle_at_end: if True, shuffle the records before assigning IDs/writing.
       - return_records: if True, return the list of records (and still write if path is not None).
    """
    # quick validation for length params (so errors surface early)
    if exact_len is not None:
        if exact_len < 1 or exact_len % 2 == 0:
            raise ValueError("exact_len must be an odd integer >= 1.")
    elif (min_len is not None) or (max_len is not None):
        if (min_len is None) or (max_len is None):
            raise ValueError("Provide both min_len and max_len, or neither.")
        if min_len > max_len:
            raise ValueError("min_len cannot be greater than max_len.")
        _ = _pick_length_from_range(min_len, max_len)

    records = []
    produced = 0
    while produced < n_samples:
        # Choose generator
        if (exact_len is not None) or (min_len is not None and max_len is not None):
            expr = random_expression_with_length(min_len=min_len, max_len=max_len, exact_len=exact_len)
            if isinstance(expr, Number):
                continue
        else:
            expr = random_expression(1)
            if isinstance(expr, Number):
                continue

        value, ok = eval_expr_safe(expr)
        if require_value and not ok:
            continue

        infix = infix_minimal(expr)
        rpn = to_rpn(expr)
        rec = {
            # 'id' filled later if assign_ids==True
            "infix": infix,
            "rpn": rpn,
            "depth": ast_depth(expr),
            "num_operands": count_operands(expr),
            "num_operators": count_operators(expr),
            "value": value,
        }
        records.append(rec)
        produced += 1

    # optional shuffle of this batch
    if shuffle_at_end:
        shuffle(records)

    # optional ID assignment
    if assign_ids:
        for j, rec in enumerate(records, start=start_id):
            rec["id"] = f"{j:04d}"

    # optional write
    if path is not None:
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    return records if return_records else None


if __name__ == "__main__":
    # length -> number of examples (sum to 1000)
    # target = {
    #     3: 260, 5: 220, 7: 170, 9: 130,
    #     11: 90, 13: 55, 15: 40, 17: 20, 19: 15,
    # }
    target = {
        3: 270, 5: 226, 7: 174, 9: 132,
        11: 92, 13: 55, 15: 40, 17: 20, 19: 15,
    }

    out = "train/train_dataset1024.jsonl"

    # 1) Generate each bucket WITHOUT IDs and WITHOUT writing
    all_recs = []
    for L, k in target.items():
        batch = create_dataset(
            k,
            path=None,                # don't write yet
            exact_len=L,
            assign_ids=False,         # don't assign ids yet
            shuffle_at_end=False,     # we'll do a global shuffle
            return_records=True,      # get the list back
        )
        all_recs.extend(batch)

    # sanity check (optional)
    assert len(all_recs) == sum(target.values())

    # 2) Global shuffle so lengths are interleaved
    shuffle(all_recs)

    # 3) Assign one continuous sequence of IDs (0001..N)
    for i, rec in enumerate(all_recs, start=1):
        rec["id"] = f"{i:04d}"

    # 4) Write once
    import os, json
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for rec in all_recs:
            f.write(json.dumps(rec) + "\n")
