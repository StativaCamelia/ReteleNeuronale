import numpy as np


def get_index(d, c):
    if c in d.keys():
        return d[c]
    else:
        return -1


def get_coefiecients():
    sign = 1
    number_fg = 0
    equal_fg = 0
    sign_fg = 0
    letter = ''
    coef = ''
    curr_idx = 0

    N = number_of_lines

    letter_idx = {}

    mat = [[0 for i in range(N)] for i in range(N)]
    results = [0 for i in range(N)]

    for i in range(N):
        l = lines[i]
        coef = ''
        equal_fg = 0
        sign_fg = 0
        sign = 1
        ls = len(l)
        k = 0  # position in line str

        for c in l:
            if c == '-':
                sign_fg = 1
                sign = -1
            elif c == '+':
                sign_fg = 1
                sign = 1
            elif c.isalpha():
                if number_fg == 0:
                    coef = 1
                letter = c
            elif c.isdigit():
                number_fg = 1
                coef += c
                if k == ls - 2 or k == ls - 1:
                    if coef == '':
                        coef = 1
                    coef = sign * int(coef)
                    if equal_fg == 0:
                        j = get_index(letter_idx, letter)
                        if j == -1:
                            j = curr_idx
                            letter_idx[letter] = j
                            curr_idx += 1
                    else:
                        j = N
                    if equal_fg:
                        results[i] = coef
            elif (c == ' ' and sign_fg != 1):
                if coef == '':
                    coef = 1
                coef = sign * int(coef)
                if equal_fg == 0:
                    j = get_index(letter_idx, letter)
                    if j == -1:
                        j = curr_idx
                        letter_idx[letter] = j
                        curr_idx += 1
                else:
                    j = N
                mat[i][j] = coef
                coef = ''
                number_fg = 0
            elif c == ' ' and sign_fg == 1:
                sign_fg = 0
            elif c == '=':
                equal_fg = 1
                sign_fg = 1
                sign = 1
            k += 1
    return mat, results, ['x', 'y', 'z']


def get_matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def calc_det_without_numpy(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = 0
    for c in range(len(matrix)):
        determinant += ((-1) ** c) * matrix[0][c] * calc_det_without_numpy(get_matrix_minor(matrix, 0, c))
    return determinant


def get_transpose_matrix(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix))]


def get_matrix_of_minors(matrix):
    return [[calc_det_without_numpy(get_matrix_minor(matrix, i, j)) for i in range(len(matrix))] for j in
            range(len(matrix))]


def get_matrix_of_cofactors(matrix):
    N = len(matrix)
    sign_matrix = [[-1 if (i+j) % 2 else 1 for i in range(N)] for j in range(N)]
    return [[matrix[i][j] * sign_matrix[i][j] for j in range(N)] for i in range(N)]


def get_adjugate_matrix(matrix):
    matrix_of_minors = get_matrix_of_minors(matrix)
    adjugate_matrix = get_matrix_of_cofactors(matrix_of_minors)
    return adjugate_matrix


def calc_inverse_without_numpy(matrix):
    if calc_det_without_numpy(matrix) == 0:
        print("Matricea nu este inversabila")
        return
    N = len(matrix)
    determinant = calc_det_without_numpy(matrix)
    adjugate_matrix = get_adjugate_matrix(matrix)
    return [[adjugate_matrix[i][j]/determinant for i in range(N)] for j in range(N)]


def get_equation_result(matrix, results_matrix):
    invers = get_transpose_matrix(calc_inverse_without_numpy(matrix))
    return scalar_product(invers, results_matrix)


def scalar_product(matrix, vector):
    result = []
    for i in range(len(matrix)):
        total = 0
        for j in range(len(vector)):
            total += vector[j] * matrix[i][j]
        result.append(total)
    return result


if __name__ == '__main__':
    file1 = open('equation.txt', 'r')
    lines = file1.readlines()
    number_of_lines = len(lines)
    A, B, X = get_coefiecients()
    print(get_equation_result(A, B))
