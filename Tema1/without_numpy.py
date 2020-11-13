import parse_equation


def scalar_product(matrix, vector):
    return [sum([vector[j] * matrix[i][j] for j in range(len(vector))]) for i in range(len(matrix))]


def get_matrix_minor(matrix, i, j):
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def get_determinant(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] if len(matrix) == 2 else sum([((-1) ** elem) * matrix[0][elem] * get_determinant(get_matrix_minor(matrix, 0, elem)) for elem in range(len(matrix))])


def get_transpose_matrix(matrix):
    return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix))]


def get_matrix_of_minors(matrix):
    return [[get_determinant(get_matrix_minor(matrix, i, j)) for i in range(len(matrix))] for j in
            range(len(matrix))]


def get_matrix_of_cofactors(matrix):
    plus_minus_matrix = [[-1 if (i + j) % 2 else 1 for i in range(len(matrix))] for j in range(len(matrix))]
    return [[matrix[i][j] * plus_minus_matrix[i][j] for j in range(len(matrix))] for i in range(len(matrix))]


def get_adj_matrix(matrix):
    return get_matrix_of_cofactors(get_matrix_of_minors(matrix))


def get_inverse(matrix):
    if get_determinant(matrix) == 0:
        return
    determinant = get_determinant(matrix)
    adj_matrix = get_adj_matrix(get_transpose_matrix(matrix))
    return [[adj_matrix[i][j] / determinant for i in range(len(matrix))] for j in range(len(matrix))]


def get_equation_result(matrix_of_coefficients, matrix_of_constants):
    return scalar_product(get_inverse(matrix_of_coefficients), matrix_of_constants)


if __name__ == '__main__':
    A, B = parse_equation.get_coefficients()
    print(get_equation_result(A, B))
