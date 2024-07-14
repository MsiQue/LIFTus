def numberRatio(col):
    # int + float
    if len(col) == 0:
        return [0.0]
    from utils import is_int, is_float
    int_count = col.apply(lambda x: (is_int(x) or is_float(x))).sum()
    return [int_count / len(col)]

def floatRatio(col):
    if len(col) == 0:
        return [0.0]
    from utils import is_float
    int_count = col.apply(lambda x: is_float(x)).sum()
    return [int_count / len(col)]

def intRatio(col):
    if len(col) == 0:
        return [0.0]
    from utils import is_int
    int_count = col.apply(lambda x: is_int(x)).sum()
    return [int_count / len(col)]

def stringRatio(col):
    if len(col) == 0:
        return [0.0]
    from utils import is_pure_alphabet_space
    int_count = col.apply(lambda x: is_pure_alphabet_space(x)).sum()
    return [int_count / len(col)]

# def averageLength(col):
#     if len(col) == 0:
#         return [0.0]
#     str_lengths = col.astype(str).apply(len)
#     return [str_lengths.mean()]
#
# def varianceLength(col):
#     if len(col) <= 1:
#         return [0.0]
#     str_lengths = col.astype(str).apply(len)
#     return [str_lengths.var()]

def average_variance_Length(col):
    if len(col) == 0:
        return [0.0, 0.0]
    str_lengths = col.astype(str).apply(len)
    if len(col) == 1:
        return [str_lengths.mean(), 0.0]
    return [str_lengths.mean(), str_lengths.var()]

def spaceRatio(col):
    if len(col) == 0:
        return [0.0]
    col_str = col.astype(str)
    return [col_str.str.contains(' ').sum() / len(col_str)]

def statistics_by_letter(col):
    import re
    string_series = col.astype(str)
    total_length = string_series.str.len().sum()
    regex_template = [r'[A-Z]', r'[a-z]', r'[0-9]']
    regex_result = [string_series.str.count(t).sum() / total_length for t in regex_template]
    letter_template = [chr(x) for x in [10, 13] + list(range(32, 48)) + list(range(58, 65)) + list(range(91, 97)) + list(range(123, 127))]
    letter_result = [string_series.str.count(re.escape(t)).sum() / total_length for t in letter_template]
    return regex_result + letter_result

# def NLW_Coverage(col):
#     from tokenizer import getWordsCount
#     from nltk.corpus import words
#     en_words = set(words.words())
#     string_counts, _ = getWordsCount(col)
#     if len(string_counts) == 0:
#         res = 0.0
#     else:
#         res = sum([x[0].lower() in en_words for x in string_counts]) / len(string_counts)
#     return [res]