import re


def get_coefficients():
    pattern_number = r"[+]?\d*\.?\d+|[+]?\d+"
    pattern_letter = r"[a-zA-Z]"
    matrix_of_const = []
    matrix_of_coefficients = []
    for line in lines:
        separate_words = [expression for expression in re.compile('[\s]').split(line) if expression != '']
        get_match = lambda pattern, string: re.findall(pattern, string)
        is_negative = lambda index: True if (index - 1 >= 0 and separate_words[index - 1] == '-') or '-' in \
                                            separate_words[
                                                index] else False
        match_number = lambda i: int(get_match(pattern_number, i)[0])
        get_coef_simple = lambda i: match_number(i) if len(get_match(pattern_number, i)) else 1
        get_number_with_sign = lambda i, index: (-1) * get_coef_simple(i) if is_negative(index) else get_coef_simple(i)
        values = {get_match(pattern_letter, i)[0]: get_number_with_sign(i, index)
                  for index, i in enumerate(separate_words) if len(get_match(pattern_letter, i))}
        matrix_line = [values.get(variable) if values.get(variable) else 0 for variable in ['x', 'y', 'z']]
        matrix_of_coefficients.append(matrix_line)
        matrix_of_const.append(int(separate_words[-1]))
    return matrix_of_coefficients, matrix_of_const


file1 = open('eq.txt', 'r')
lines = file1.readlines()
