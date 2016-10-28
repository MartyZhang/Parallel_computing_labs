import math

def up(rank, size):
    return rank - rank / math.sqrt(size)

def down(rank, size):
    return rank + rank / math.sqrt(size)

def left(rank, size):
    return rank - 1

def right(rank, size):
    return rank + 1