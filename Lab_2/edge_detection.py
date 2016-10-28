import math

# Get the X index of the process
def processX(rank, size):
    return int(rank % math.sqrt(size)) 

# Get the Y index of the process
def processY(rank, size):
    return int(rank / math.sqrt(size))

def IsNotEdgeProcess(rank, size):
    return processX(rank, size) < math.sqrt(size) and processY(rank, size) < math.sqrt(size)

def IsTopLeftCorner(rank, size):
    return processX(rank, size) == 0 and processY(rank, size) == 0

def IsTopRightCorner(rank, size):
    return processX(rank, size) == math.sqrt(size) - 1 and processY(rank, size) == 0
    
def IsBottomLeftCorner(rank, size):
    x = processX(rank, size)
    y = processY(rank, size)
    return processX(rank, size) == 0 and processY(rank, size) == math.sqrt(size) - 1
    
def IsBottomRightCorner(rank, size):
    return processX(rank, size) == math.sqrt(size) - 1 and processY(rank, size) == math.sqrt(size) - 1

def IsLeftEdge(rank, size):
    y = processY(rank, size)
    return processX(rank, size) == 0 and y > 0 and y < math.sqrt(size) - 1

def IsRightEdge(rank, size):
    y = processY(rank, size)
    return processX(rank, size) == math.sqrt(size) - 1 and y > 0 and y < math.sqrt(size) - 1

def IsTopEdge(rank, size):
    y = processY(rank, size)
    return processY(rank, size) == 0 and x > 0 and x < math.sqrt(size) - 1

def IsBottomEdge(rank, size):
    y = processY(rank, size)
    return processY(rank, size) == math.sqrt(size) - 1 and x > 0 and x < math.sqrt(size) - 1

def IsCenterProcess(rank, size):
    width = math.sqrt(size)
    return processX(rank, size) == int(width / 2) and processY(rank, size) == int(width / 2)