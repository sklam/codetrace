def ifelse1(a):

    if a < 1:
        a = a + 1

    else:
        a = a - 1

    return a


def ifelse2(a):

    if 2 < a < 5:
        a = a + 4

    elif a >= 5:
        a = a + 3

    else:
        a = a + 2

    if a < 10:
        a = a + 5

    else:
        a = a + 6

    return a


def loop1(a):
    c = 0
    for i in range(a):
        c += i
    return c


def loop2(a):
    c = 0
    for i in range(a * 2):
        print('i', i)
        if c < a * 2:
            print('c', c)
            for i in range(a):
                print('i2', i)
                c += i
                print('c += i', c, i)
        else:
            print('break')
            break
    return c


def loop3(a):
    c = 0
    for i in range(a):
        d = 0
        # print('i', i, d)
        for j in range(a):
            # print('j', j, d)
            d += j
        c += d
    return c


def loop4(a):
    c = 0
    for i in range(a):
        for j in range(a):
            for k in range(a):
                c += (i + 1) * (j + 2) * (k + 3)
            c += k
        c += j
    c += 1
    return c


def loop5(a):
    c = 0
    for i in range(a):
        for j in range(a):
            if c < a * 10:
                for i in range(a):
                    c += i + j
                c += 1
            else:
                for i in range(a):
                    c += 2 * i + j
                c += 1
                break
                c += 1
        else:
            c += c
    return c
