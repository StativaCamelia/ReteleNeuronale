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

    mat = [[0 for _ in range(N)] for _ in range(N)]
    results = [0 for _ in range(N)]

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


def get_index(d, c):
    if c in d.keys():
        return d[c]
    else:
        return -1