import wordninja

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False

def split_string_number(L):
    List_string = []
    List_number = []
    for x in L:
        if isfloat(x):
            List_number.append(float(x))
        else:
            List_string.append(x)
    return List_string, List_number

def filter_default(s):
    if isfloat(s):
        return [], [float(s)]
    return split_string_number(wordninja.split(s))

def filter(s, method = 'default'):
    if method == 'default':
        return filter_default(s)

if __name__ == '__main__':
    print(filter('1.23'))
    print(filter('192.168.0.1'))
    print(filter('word2vector'))
    print(filter('wordninja'))
    print(filter('234r345tasbep354'))