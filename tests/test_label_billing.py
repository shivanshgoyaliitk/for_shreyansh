from types import SimpleNamespace
import importlib
import label_billing


def fresh():
    return importlib.reload(label_billing)


def test_sdk_cost_not_flash_estimate():
    b = fresh()
    b.record_chat_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=2000, output_tokens=100,
        input_tokens_cost_nanodollars=6_000_000, output_tokens_cost_nanodollars=1_500_000)))
    assert b.spent_usd() == .0075
    assert not b.HALT_REQUESTS.is_set()
    assert 'not account quota' in b.cost_status()


def test_missing_usage_stops_and_marks_partial_total():
    b = fresh()
    b.record_chat_usage(SimpleNamespace(usage=SimpleNamespace(total_cost_nanodollars=9_000_000)))
    b.record_chat_usage(SimpleNamespace())
    assert b.HALT_REQUESTS.is_set()
    assert b.spent_usd() == .009
    assert 'total UNKNOWN' in b.cost_status()
    assert b.USAGE['attempts'] == 2


def test_quota_error_detection():
    b = fresh()
    assert b.quota_error(RuntimeError('RESOURCE_EXHAUSTED: Daily quota exceeded'))
    assert not b.quota_error(RuntimeError('invalid JSON response'))
