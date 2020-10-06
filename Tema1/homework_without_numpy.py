import re

def get_coeficients():
    pattern_number = r"[+]?\d*\.?\d+|[+]?\d+"
    pattern_letter = r"[a-zA-Z]"
    matrix_const = []
    matrix_coefficients = []
    for line in lines:
        separate_words = [expression for expression in re.compile('[\s]').split(line) if expression != '']
        get_match = lambda pattern, string: re.findall(pattern, string)
        is_negative = lambda index: True if (index - 1 >= 0 and separate_words[index - 1] == '-') or '-' in separate_words[
            index] else False
        match_number = lambda i: int(get_match(pattern_number, i)[0])
        get_coef_simple = lambda i: match_number(i) if len(get_match(pattern_number, i)) else 1
        get_number_with_sign = lambda i, index: (-1) * get_coef_simple(i) if is_negative(index) else get_coef_simple(i)
        values = {get_match(pattern_letter, i)[0]: get_number_with_sign(i, index)
                  for index, i in enumerate(separate_words) if len(get_match(pattern_letter, i))}
        matrix_line = [values.get(variable) if values.get(variable) else 0 for variable in ['x', 'y', 'z']]
        matrix_coefficients.append(matrix_line)
        matrix_const.append(int(separate_words[-1]))
    return matrix_coefficients, matrix_const


def get_matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def get_det(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum([((-1) ** c) * matrix[0][c] * get_det(get_matrix_minor(matrix, 0, c)) for c in range(len(matrix))])


def get_transpose_matrix(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix))]


def get_matrix_of_minors(matrix):
    return [[get_det(get_matrix_minor(matrix, i, j)) for i in range(len(matrix))] for j in
            range(len(matrix))]


def get_matrix_of_cofactors(matrix):
    N = len(matrix)
    sign_matrix = [[-1 if (i + j) % 2 else 1 for i in range(N)] for j in range(N)]
    return [[matrix[i][j] * sign_matrix[i][j] for j in range(N)] for i in range(N)]


def get_adjugate_matrix(matrix):
    matrix_of_minors = get_matrix_of_minors(matrix)
    adjugate_matrix = get_matrix_of_cofactors(matrix_of_minors)
    return adjugate_matrix


def get_inverse(matrix):
    if get_det(matrix) == 0:
        print("Matricea nu este inversabila")
        return
    N = len(matrix)
    determinant = get_det(matrix)
    transpose = get_transpose_matrix(matrix)
    adjugate_matrix = get_adjugate_matrix(transpose)
    return [[adjugate_matrix[i][j] / determinant for i in range(N)] for j in range(N)]


def get_equation_result(matrix_coef, matrix_consts):
    invers = get_inverse(matrix_coef)
    return scalar_product(invers, matrix_consts)


def scalar_product(matrix, vector):
    return [sum([vector[j] * matrix[i][j] for j in range(len(vector))]) for i in range(len(matrix))]


if __name__ == '__main__':
    file1 = open('equation.txt', 'r')
    lines = file1.readlines()
    number_of_lines = len(lines)
    A, B = get_coeficients()
    print(get_equation_result(A, B))
