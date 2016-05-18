from __future__ import print_function, absolute_import


class Poison(object):
    pass


Poison = Poison()


def foriter(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, Poison


def foriter_value(foriter_struct):
    return foriter_struct[1]


def foriter_valid(foriter_struct):
    return foriter_struct[0]
