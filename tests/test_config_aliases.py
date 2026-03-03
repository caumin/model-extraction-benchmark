import warnings

from mebench.utils.config_aliases import (
    resolve_iterations,
    resolve_nominal_query_budget,
    resolve_query_budget,
)


def test_resolve_iterations_prefers_canonical_key() -> None:
    cfg = {"iterations": 7, "rounds": 9}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = resolve_iterations(cfg, default=10, context="unit")
    assert value == 7
    assert len(w) == 0


def test_resolve_iterations_supports_legacy_rounds_with_warning() -> None:
    cfg = {"rounds": 9}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = resolve_iterations(cfg, default=10, context="unit")
    assert value == 9
    assert any("deprecated" in str(msg.message).lower() for msg in w)


def test_resolve_iterations_supports_num_rounds_when_enabled() -> None:
    cfg = {"num_rounds": 11}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = resolve_iterations(cfg, default=10, context="unit", allow_num_rounds=True)
    assert value == 11
    assert any("num_rounds" in str(msg.message) for msg in w)


def test_resolve_query_budget_prefers_query_budget() -> None:
    cfg = {"query_budget": 123, "querybudget": 456}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = resolve_query_budget(cfg, default=2000, context="unit")
    assert value == 123
    assert len(w) == 0


def test_resolve_query_budget_legacy_aliases_warn() -> None:
    for cfg in ({"querybudget": 321}, {"budget": 654}):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = resolve_query_budget(cfg, default=2000, context="unit")
        assert value in {321, 654}
        assert any("deprecated" in str(msg.message).lower() for msg in w)


def test_resolve_nominal_query_budget_alias_warns() -> None:
    cfg = {"nominal_querybudget": 88}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        value = resolve_nominal_query_budget(cfg, default=0, context="unit")
    assert value == 88
    assert any("deprecated" in str(msg.message).lower() for msg in w)


def test_resolve_iterations_returns_default_when_missing() -> None:
    assert resolve_iterations({}, default=10, context="unit") == 10


def test_resolve_query_budget_returns_default_when_missing() -> None:
    assert resolve_query_budget({}, default=2000, context="unit") == 2000


def test_resolve_nominal_query_budget_returns_default_when_missing() -> None:
    assert resolve_nominal_query_budget({}, default=0, context="unit") == 0
