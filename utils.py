import math

def is_int(x):  # is_int('310246_1501') return True
    x = str(x)
    try:
        _ = int(x)
        return True
    except ValueError:
        return False

def is_float(x):
    x = str(x)
    try:
        _ = float(x)
        return True
    except ValueError:
        return False

def is_invalid(x):
    return math.isnan(x) or math.isinf(x)

def is_pure_alphabet_space(x):
    return all(c.isalpha() or c.isspace() for c in str(x))

def _(a, b):
    if b in a:
        return a[b]
    if b[:-1] in a:
        return a[b[:-1]]
    if (b+'\r') in a:
        return a[b+'\r']
    return None

def __(s):
    return s.rstrip('\r') if s.endswith('\r') else s