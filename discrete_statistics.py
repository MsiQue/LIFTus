# uniqueness
# isEqualLength
# hasLetter
# hasSpace
# hasDigit
# hasSpecialCharacter

def uniqueness(col, limit = 10):
    # return a vector length (limit + 3)
    # one-hot [0] [1] ... [9] [diff_value < sqrt(len)] [diff_value >= sqrt(len)] [all diff]
    L = len(col)
    counts = col.value_counts().to_dict()
    diff_value = len(counts)
    res = [0 for i in range(limit)]
    if diff_value < limit:
        res[diff_value] = 1
    res.append(1 if (diff_value >= limit and diff_value * diff_value < L) else 0)
    res.append(1 if (diff_value * diff_value >= L and diff_value != L) else 0)
    res.append(1 if diff_value == L else 0)
    return res

def isEqualLength(col):
    if len(col) <= 1:
        res = 1
    else:
        res = 1
        str_lengths = col.astype(str).apply(len)
        for x in str_lengths:
            if x != str_lengths.iloc[0]:
                res = 0
                break
    # print(res)
    return [res]

def has_something(col):

    def has_single(L, f):
        for x in L:
            if f(x) is not None:
                return 1
        return 0

    import re
    string_series = col.astype(str)
    regex_template = [r'[A-Z]', r'[a-z]', r'[0-9]']
    regex_f = [lambda x, t=t: re.search(t, x) for t in regex_template]
    letter_template = [chr(x) for x in [10, 13] + list(range(32, 48)) + list(range(58, 65)) + list(range(91, 97)) + list(range(123, 127))]
    letter_f = [lambda x, t=t: re.search(re.escape(t), x) for t in letter_template]
    return [has_single(string_series, f) for f in regex_f + letter_f]