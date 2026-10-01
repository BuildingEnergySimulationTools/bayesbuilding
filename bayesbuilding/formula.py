"""Forward models declared as formulas instead of Python functions.

A :class:`FormulaModel` is a ``model_function`` for :class:`~bayesbuilding.wrapper.
PymcWrapper` built from plain-text expressions, so that a new model variant is
a line of JSON rather than a new function in :mod:`bayesbuilding.models`::

    FormulaModel(
        mu="g[occ]*dt - fs[occ]*rad + C*(beta*DTint_dt + (1-beta)*DText_dt)",
        sigma="s0[occ]",
        lower=0.0,
    )

Grammar
-------
- Operators: ``+ - * / **`` and unary ``+``/``-``.
- Numeric constants.
- Symbols: every name is either a prior (a key of the wrapper's
  ``priors_dict``) or an input (a feature column). Inputs are whatever names
  are not priors. Their order -- the column order of ``x`` -- is: continuous
  drivers first, then categorical inputs (used as an index or as a
  ``switch`` condition), each group in order of first appearance across
  ``mu``, ``sigma`` and ``extras``. So ``x[:, 0]`` is the main driver, e.g.
  ``dt`` in ``g[occ]*dt - fs[occ]*rad`` -> ``(dt, rad, occ)``. Pass
  ``inputs`` explicitly to force another order.
- Indexing: ``g[occ]`` or ``g[occ, 1]`` indexes a vector/matrix prior by a
  categorical input (cast to int) and/or integer constants.
- Functions: ``max(a, b)``, ``min(a, b)``, ``switch(cond, a, b)``,
  ``sigmoid(a)``, ``sqrt(a)``, ``exp(a)``, ``log(a)``, ``abs(a)``.

Anything else (attribute access, comparisons, arbitrary calls, lambdas...) is
rejected at construction time: expressions are parsed with :mod:`ast` and
walked against a whitelist, never passed to ``eval``.
"""

import ast
import warnings

import pymc as pm
import pytensor.tensor as pt

FUNCTIONS = {
    "max": (pm.math.maximum, 2),
    "min": (pm.math.minimum, 2),
    "switch": (pm.math.switch, 3),
    "sigmoid": (pm.math.sigmoid, 1),
    "sqrt": (pt.sqrt, 1),
    "exp": (pt.exp, 1),
    "log": (pt.log, 1),
    "abs": (pt.abs, 1),
}

_BINARY_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPERATORS = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: a,
}


class FormulaError(ValueError):
    """Raised when a formula uses a construct outside the allowed grammar."""


def _parse(expr: str) -> ast.expr:
    try:
        tree = ast.parse(expr.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"Invalid formula {expr!r}: {e.msg}") from e
    _check(tree, expr)
    return tree


def _check(node: ast.AST, expr: str):
    """Walk ``node`` and raise FormulaError on anything outside the grammar."""
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY_OPERATORS:
            raise FormulaError(f"Operator {type(node.op).__name__} not allowed in {expr!r}")
        _check(node.left, expr)
        _check(node.right, expr)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPERATORS:
            raise FormulaError(f"Operator {type(node.op).__name__} not allowed in {expr!r}")
        _check(node.operand, expr)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FormulaError(f"Only numeric constants allowed in {expr!r}")
    elif isinstance(node, ast.Name):
        if node.id in FUNCTIONS:
            raise FormulaError(f"Function {node.id!r} used as a symbol in {expr!r}")
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise FormulaError(f"Only a symbol can be indexed in {expr!r}")
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        for index in indices:
            is_int = isinstance(index, ast.Constant) and type(index.value) is int
            if not (isinstance(index, ast.Name) or is_int):
                raise FormulaError(
                    f"Index must be an input name or an integer in {expr!r}"
                )
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = getattr(node.func, "id", ast.unparse(node.func))
            raise FormulaError(
                f"Function {name!r} not allowed in {expr!r}. "
                f"Allowed: {sorted(FUNCTIONS)}"
            )
        if node.keywords:
            raise FormulaError(f"Keyword arguments not allowed in {expr!r}")
        arity = FUNCTIONS[node.func.id][1]
        if len(node.args) != arity:
            raise FormulaError(
                f"{node.func.id}() takes {arity} argument(s), "
                f"got {len(node.args)} in {expr!r}"
            )
        for arg in node.args:
            _check(arg, expr)
    else:
        raise FormulaError(f"{type(node).__name__} not allowed in {expr!r}")


def _symbols(node: ast.AST) -> list[str]:
    """Names in order of first appearance (functions excluded)."""
    # ast.walk is breadth-first: sort by source position for textual order.
    found = sorted(
        (
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Name) and child.id not in FUNCTIONS
        ),
        key=lambda child: (child.lineno, child.col_offset),
    )
    return list(dict.fromkeys(child.id for child in found))


def _index_symbols(node: ast.AST) -> set[str]:
    """Names used inside a subscript: categorical inputs, cast to int."""
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript):
            indices = (
                child.slice.elts if isinstance(child.slice, ast.Tuple) else [child.slice]
            )
            names |= {i.id for i in indices if isinstance(i, ast.Name)}
    return names


def _switch_condition_symbols(node: ast.AST) -> set[str]:
    """Names used directly as a ``switch`` condition (a flag input)."""
    return {
        child.args[0].id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and child.func.id == "switch"
        and isinstance(child.args[0], ast.Name)
    }


def _evaluate(node: ast.AST, env: dict, categorical: dict):
    if isinstance(node, ast.BinOp):
        return _BINARY_OPERATORS[type(node.op)](
            _evaluate(node.left, env, categorical),
            _evaluate(node.right, env, categorical),
        )
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand, env, categorical))
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.Subscript):
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        resolved = tuple(
            categorical[i.id] if isinstance(i, ast.Name) else i.value for i in indices
        )
        target = env[node.value.id]
        return target[resolved[0]] if len(resolved) == 1 else target[resolved]
    if isinstance(node, ast.Call):
        func = FUNCTIONS[node.func.id][0]
        return func(*(_evaluate(arg, env, categorical) for arg in node.args))
    raise FormulaError(f"{type(node).__name__} not allowed")  # unreachable


class FormulaModel:
    """A PymcWrapper ``model_function`` defined by text formulas.

    :param mu: expression of the likelihood mean.
    :param sigma: expression of the likelihood scale.
    :param lower: lower truncation bound returned as the ``"lower"`` extra
        (for a ``TruncatedNormal`` likelihood); ``None`` returns no ``"lower"``.
    :param extras: further named expressions exposed as ``pm.Deterministic``
        for diagnostics.
    :param inputs: ordered input names, i.e. the expected column order of
        ``x``. Usually left out and derived with :meth:`bind` from the prior
        names; when given, every other symbol is a prior.
    """

    def __init__(
        self,
        mu: str,
        sigma: str,
        lower: float | None = 0.0,
        extras: dict[str, str] | None = None,
        inputs: list[str] | tuple[str, ...] | None = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.extras = dict(extras or {})
        reserved = {"mu", "sigma", "lower"} & set(self.extras)
        if reserved:
            raise FormulaError(f"Reserved extras names: {sorted(reserved)}")
        self._trees = {"mu": _parse(mu), "sigma": _parse(sigma)}
        self._trees.update({name: _parse(e) for name, e in self.extras.items()})

        self.symbols: list[str] = []
        for tree in self._trees.values():
            for name in _symbols(tree):
                if name not in self.symbols:
                    self.symbols.append(name)
        self.categorical_inputs = set().union(
            *(_index_symbols(t) for t in self._trees.values())
        )
        self._flag_inputs = self.categorical_inputs | set().union(
            *(_switch_condition_symbols(t) for t in self._trees.values())
        )
        self.inputs = None if inputs is None else tuple(inputs)
        if self.inputs is not None:
            self._check_inputs(self.inputs)

    def _check_inputs(self, inputs):
        unknown = [name for name in inputs if name not in self.symbols]
        if unknown:
            raise FormulaError(f"Inputs {unknown} do not appear in the formula")
        not_inputs = self.categorical_inputs - set(inputs)
        if not_inputs:
            raise FormulaError(
                f"{sorted(not_inputs)} are used as indices, so they must be inputs"
            )

    def input_names(self, prior_names) -> tuple[str, ...]:
        """Symbols that are not priors: continuous drivers first, then
        categorical/flag inputs, each in order of first appearance."""
        names = [name for name in self.symbols if name not in set(prior_names)]
        return tuple(
            [n for n in names if n not in self._flag_inputs]
            + [n for n in names if n in self._flag_inputs]
        )

    def bind(self, prior_names) -> "FormulaModel":
        """Fix ``inputs`` from the prior names (keeping an explicit ``inputs``
        order if one was given); warns if a prior is not used by the formula.
        Returns ``self`` for chaining."""
        prior_names = set(prior_names)
        unused = sorted(prior_names - set(self.symbols))
        if unused:
            # Not an error: the prior is still sampled, it just doesn't inform
            # anything (legacy model functions silently ignored such priors).
            warnings.warn(f"Priors {unused} are not used in the formula", stacklevel=2)
        clash = sorted(prior_names & set(self.extras))
        if clash:
            raise FormulaError(f"Extras {clash} have the same name as a prior")
        inputs = self.input_names(prior_names)
        if self.inputs is not None:
            if set(self.inputs) != set(inputs):
                raise FormulaError(
                    f"Explicit inputs {list(self.inputs)} do not match the "
                    f"non-prior symbols {list(inputs)}"
                )
            inputs = self.inputs
        self._check_inputs(inputs)
        self.inputs = inputs
        return self

    def __call__(self, x, variables: dict):
        inputs = (
            self.inputs if self.inputs is not None else self.input_names(variables)
        )
        missing = [
            name for name in self.symbols if name not in inputs and name not in variables
        ]
        if missing:
            raise FormulaError(f"Symbols {missing} are neither priors nor inputs")
        env = dict(variables)
        env.update({name: x[:, i] for i, name in enumerate(inputs)})
        categorical = {
            name: x[:, inputs.index(name)].astype("int64")
            for name in self.categorical_inputs
        }
        values = {
            name: _evaluate(tree, env, categorical) for name, tree in self._trees.items()
        }
        mu = values.pop("mu")
        out = {"sigma": values.pop("sigma")}
        if self.lower is not None:
            out["lower"] = pt.constant(float(self.lower))
        out.update(values)
        return mu, out

    def to_dict(self) -> dict:
        spec = {"mu": self.mu, "sigma": self.sigma, "lower": self.lower}
        if self.extras:
            spec["extras"] = dict(self.extras)
        if self.inputs is not None:
            spec["inputs"] = list(self.inputs)
        return spec

    @classmethod
    def from_dict(cls, spec: dict) -> "FormulaModel":
        return cls(**spec)

    def __repr__(self):
        return f"FormulaModel(mu={self.mu!r}, sigma={self.sigma!r}, lower={self.lower!r})"
