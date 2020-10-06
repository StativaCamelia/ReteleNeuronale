import numpy as np


def prim(n):
    if n == 0 or n == 1:
        return False
    for i in range(2, int(n / 2)):
        if n % i == 0:
            return False
    return True


def sort_words(file_name):
    file = open(file_name, "r")
    file_content = file.read()
    file_array = file_content.split(" ")
    file_sorted = sorted(file_array, key=str.lower)
    print(' '.join(file_sorted))


def scalar_product(matrix, vector):
    result = []
    for i in range(len(matrix)):
        total = 0
        for j in range(len(vector)):
            total += vector[j] * matrix[i][j]
        result.append(total)
    return result


def scalar_numpy(matrix, vector):
    return np.dot(matrix, vector)


def with_numpy_1(matrix, vector):
    matrix_numpy = np.array(matrix)
    vector_numpy = np.array(vector)
    print(matrix_numpy[:2, -2:])
    print(vector_numpy[-2:])


def with_numpy_2(length):
    vector1 = np.random.rand(length)
    vector2 = np.random.rand(length)
    print("Vectori:")
    print(vector1, vector2)
    print("Suma cea mai mare:")
    print(vector1, vector1.sum()) if (vector1.sum() > vector2.sum()) else print(vector2, vector2.sum())
    print("Adunare:")
    print(vector1 + vector2)
    print("Inmultire vectoriala")
    print(vector1 * vector2)
    print("Inmultire scalara")
    print(vector1.dot(vector2))
    print("Radical vector 1")
    print(np.sqrt(vector1))
    print("Radical vector 2")
    print(np.sqrt(vector2))


def with_numpy_3(dim_x, dim_y):
    matrix = np.random.rand(dim_x, dim_y)
    print("Matrix")
    print(matrix)
    print("Transpusa:")
    print(np.transpose(matrix))
    print("Inversa + Determinant:")
    try:
        inverse = np.linalg.inv(matrix)
        print("Determinant:")
        print(np.linalg.det(matrix))
    except np.linalg.LinAlgError:
        print("Matrice neinversabila")
        pass
    return matrix


def with_numpy_4():
    vector = np.random.rand(3)
    print("Produs scalar matrice, vector: ")
    print(matrix.dot(vector))


if __name__ == '__main__':
    print("Prima problema:")
    print(prim(3))
    print(prim(1))
    print(prim(9))
    print("A doua problema:")
    sort_words("fileToRead.txt")
    print("A treia problema:")
    print(scalar_product([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], [2, -5, 7, 10]))
    print(scalar_numpy([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], [2, -5, 7, 10]))
    print("Probleme cu Numpy:")
    with_numpy_1([[1, 2, 3, 4], [11, 12, 13, 14], [21, 22, 23, 24]], [2, -5, 7, 10])
    with_numpy_2(5)
    matrix = with_numpy_3(5, 3)
    with_numpy_4()
