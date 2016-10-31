#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

def NotAnEdge(rank):
    return True if (rank % 4 > 0 and rank % 4 < 3) and (rank / 4 > 0 and rank / 4 < 3) else False

def IsEdge(rank):
    return True if (rank % 4 == 0 and (rank / 4 > 0 and rank / 4 < 3)) or (rank % 4 == 3 and (rank / 4 > 0 and rank / 4 < 3)) or (rank / 4 == 0 and (rank % 4 > 0 and rank % 4 < 3)) or (rank / 4 == 3 and (rank % 4 > 0 and rank % 4 < 3)) else False

def Center(rank):
    return True if rank == 10 else False

def IsLeftEdge(rank):
    return True if (rank % 4 == 0 and (rank / 4 > 0 and rank / 4 < 3)) else False

def IsRightEdge(rank):
    return True if (rank % 4 == 3 and (rank / 4 > 0 and rank / 4 < 3)) else False

def IsTopEdge(rank):
    return True if (rank / 4 == 0 and (rank % 4 > 0 and rank % 4 < 3)) else False

def IsBottomEdge(rank):
    return True if (rank / 4 == 3 and (rank % 4 > 0 and rank % 4 < 3)) else False

def TopLeftCorner(rank):
    return True if rank == 0 else False

def TopRightCorner(rank):
    return True if rank == 3 else False

def BottomLeftCorner(rank):
    return True if rank == 12 else False

def BottomRightCorner(rank):
    return True if rank == 15 else False

def CalculateNewIntensity(top, left, right, bottom, lastTimeStep, lastLastTimeStep, p, n):
    return (p*(top + left + right + bottom - 4 * lastTimeStep) + 2 * lastTimeStep - (1-n)*lastLastTimeStep)/(1 + n)

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

T = int(sys.argv[1])
intensities = numpy.zeros(3, dtype=numpy.float32)

top = numpy.zeros(1, dtype=numpy.float32)
left = numpy.zeros(1, dtype=numpy.float32)
right = numpy.zeros(1, dtype=numpy.float32)
bottom = numpy.zeros(1, dtype=numpy.float32)

# Constants
n = 0.0002
p = 0.5
G = 0.75
arraySize = 3

def up(rank):
    return 4 * (int(rank / 4) - 1) + ((rank % 4))

def down(rank):
    return 4 * (int(rank / 4) + 1) + ((rank % 4))

def leftIndex(rank):
    return rank - 1

def rightIndex(rank):
    return rank + 1

current = numpy.zeros(1, dtype=numpy.float32)
previous = numpy.zeros(1, dtype=numpy.float32)
previousPrevious = numpy.zeros(1, dtype=numpy.float32)

if Center(rank):
    current[0] = 1.0
    previous[0] = 1.0
    
# Decrement to 0
for i in range(0, T):
    comm.Barrier()
    
    if NotAnEdge(rank):
        if rank == 5:
            comm.Send(previous[0], dest=rightIndex(rank))
            comm.Send(previous[0], dest=down(rank))

        if rank == 6:
            comm.Send(previous[0], dest=leftIndex(rank))
            comm.Send(previous[0], dest=down(rank))

        if rank == 9:
            comm.Send(previous[0], dest=up(rank))
            comm.Send(previous[0], dest=rightIndex(rank))

        if rank == 10:
            comm.Send(previous[0], dest=up(rank))
            comm.Send(previous[0], dest=leftIndex(rank))

        comm.Recv(top, source=up(rank))
        comm.Recv(left, source=leftIndex(rank))
        comm.Recv(right, source=rightIndex(rank))
        comm.Recv(bottom, source=down(rank))

        current[0] = CalculateNewIntensity(top[0], left[0], right[0], bottom[0], previous[0], previousPrevious[0], p, n)

        if rank == 5:
            comm.Send(current[0], dest=up(rank))
            comm.Send(current[0], dest=leftIndex(rank))

        if rank == 6:
            comm.Send(current[0], dest=up(rank))
            comm.Send(current[0], dest=rightIndex(rank))

        if rank == 9:
            comm.Send(current[0], dest=leftIndex(rank))
            comm.Send(current[0], dest=down(rank))

        if rank == 10:
            comm.Send(current[0], dest=rightIndex(rank))
            comm.Send(current[0], dest=down(rank))

    # Calculate comm.Send for edges
    elif IsLeftEdge(rank):
        comm.Send(current[0], dest=rightIndex(rank))

        comm.Recv(right, source=rightIndex(rank))
        current[0] = G * right[0]

        comm.Send(current[0], dest=up(rank))
        comm.Send(current[0], dest=down(rank))

    elif IsRightEdge(rank):
        comm.Send(current[0], dest=leftIndex(rank))

        comm.Recv(left, source=leftIndex(rank))
        current[0] = G * left[0]

        comm.Send(current[0], dest=up(rank))
        comm.Send(current[0], dest=down(rank))

    elif IsTopEdge(rank):
        comm.Send(current[0], dest=down(rank))

        comm.Recv(bottom, source=down(rank))
        current[0] = G * bottom[0]

        comm.Send(current[0], dest=leftIndex(rank))
        comm.Send(current[0], dest=rightIndex(rank))

    elif IsBottomEdge(rank):
        comm.Send(current[0], dest=up(rank))

        comm.Recv(top, source=up(rank))
        current[0] = G * top[0]

        comm.Send(current[0], dest=leftIndex(rank))
        comm.Send(current[0], dest=rightIndex(rank))

    # Must be a corner
    elif TopLeftCorner(rank):
        comm.Recv(right, source=rightIndex(rank))
        current[0] = G * right[0]
        
    elif TopRightCorner(rank):
        comm.Recv(left, source=leftIndex(rank))
        current[0] = G * left[0]

    elif BottomLeftCorner(rank):
        comm.Recv(top, source=up(rank))
        current[0] = G * top[0]

    elif BottomRightCorner(rank):
        comm.Recv(top, source=up(rank))
        current[0] = G * top[0]

    if NotAnEdge(rank):
        previousPrevious[0] = previous[0]
        previous[0] = current[0]

    if Center(rank):
        print "Rank N/2 has Intensities: %f\n" % current[0]

    comm.Barrier()