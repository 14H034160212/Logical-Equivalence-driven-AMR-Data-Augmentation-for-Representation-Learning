"""Logic rules for AMR/UMR-based logical equivalence data augmentation.

Re-export the registry helpers so callers can do:

    from extensions.logic_rules import get_rule, all_rules, rule_names
    rule = get_rule("contraposition")
    results = rule.apply(graph)

Subclasses self-register on import via the `@register` decorator in `base.py`.
"""

from .base import (  # noqa: F401
    LogicRule,
    RuleMatch,
    RuleResult,
    all_rules,
    get_rule,
    rule_names,
)

# AMR-level rules (9 total)
from . import contraposition       # noqa: F401
from . import commutative          # noqa: F401
from . import implication          # noqa: F401
from . import double_negation      # noqa: F401
from . import de_morgan            # noqa: F401
from . import inverse_relation     # noqa: F401
from . import symmetric_asymmetric # noqa: F401
from . import predicate_implication # noqa: F401
from . import transitivity         # noqa: F401  (stub)

# UMR-level rules (AMR-layer approximations; full UMR conversion is future work)
from . import aspect_equivalence             # noqa: F401
from . import modal_strength_inversion       # noqa: F401
from . import doc_temporal_transitivity      # noqa: F401
from . import tense_transformation           # noqa: F401
