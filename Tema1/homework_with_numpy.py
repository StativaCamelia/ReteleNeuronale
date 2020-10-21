import numpy as np

from parse_equation import get_coefficients

if __name__ == '__main__':
    A, B = get_coefficients()
    print(np.linalg.solve(np.array(A), np.array(B)))
