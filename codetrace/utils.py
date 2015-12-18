"""
Utilities
"""
import string
import random
import itertools


def unique_name_generator(min_len=1):
    """
    A generator that yields unique string suitable for use as identifier
    (first letter is ascii letters and the remaining are ascii letter+digits).
    Use `min_len` to specify the minimum length of the string.
    The string length will increase when all characters combinations of the
    current length is exhausted.
    """
    assert min_len > 0
    sources = [list(string.ascii_letters)]
    random.shuffle(sources[0])

    seed = string.ascii_letters + string.digits
    for _ in range(1, min_len):
        chars = list(seed)
        random.shuffle(chars)
        sources.append(chars)

    while True:
        for out in itertools.product(*sources):
            yield ''.join(out)

        chars = list(seed)
        random.shuffle(chars)
        sources.append(chars)
