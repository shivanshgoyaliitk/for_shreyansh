"""SDK-reported per-attempt costs. Never substitute another model's prices."""
import threading

_usage_lock = threading.Lock()
USAGE = {'in': 0, 'out': 0, 'calls': 0, 'errors': 0, 'nanodollars': 0,
         'attempts': 0, 'unmetered': 0}
HALT_REQUESTS = threading.Event()


def record_chat_usage(chat):
    # Capture usage even if response parsing or the provider call failed.
    try:
        u = chat.usage
        n_in = getattr(u, 'input_tokens', 0) or 0
        n_out = getattr(u, 'output_tokens', 0) or 0
        nd = getattr(u, 'total_cost_nanodollars', None)
        if nd is None:
            a = getattr(u, 'input_tokens_cost_nanodollars', None)
            b = getattr(u, 'output_tokens_cost_nanodollars', None)
            nd = a + b if a is not None and b is not None else None
        # Zero/default usage is not evidence that a remote attempt was free.
        known = nd is not None and nd > 0
    except Exception:
        n_in = n_out = 0
        nd, known = None, False
    with _usage_lock:
        USAGE['attempts'] += 1
        USAGE['in'] += n_in
        USAGE['out'] += n_out
        USAGE['nanodollars'] += nd if known else 0
        USAGE['unmetered'] += int(not known)
    if not known:
        HALT_REQUESTS.set()


def spent_usd():
    return USAGE['nanodollars'] / 1e9


def cost_status():
    amount = f"${spent_usd():.4f} SDK-reported batch cost"
    if USAGE['unmetered']:
        amount += f"; total UNKNOWN ({USAGE['unmetered']} unmetered attempts)"
    return amount + '; not account quota remaining'


def quota_error(exc):
    message = str(exc).lower()
    return any(word in message for word in
               ('quota', 'resource_exhausted', 'resource exhausted', 'insufficient credit',
                'budget exceeded', 'daily limit', 'monthly limit'))
