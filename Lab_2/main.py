#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

def Send(rank, intensity):
    comm.Send(intensity, dest=rank)

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

position = (rank / 4, rank % 4)

T = int(sys.argv[1])
intensities = numpy.zeros(T, dtype=numpy.int)

top = numpy.zeros(1, dtype=numpy.int)
left = numpy.zeros(1, dtype=numpy.int)
right = numpy.zeros(1, dtype=numpy.int)
bottom = numpy.zeros(1, dtype=numpy.int)

# Constants
n = 0.5
p = 1
G = 10

def up(rank):
    return 4 * (int(rank / 4) - 1) + ((rank % 4))

def down(rank):
    return 4 * (int(rank / 4) + 1) + ((rank % 4))

def leftIndex(rank):
    return 4 * (int(rank / 4)) + ((rank % 4) - 1)

def rightIndex(rank):
    return 4 * (int(rank / 4)) + ((rank % 4) + 1)

if Center(rank):
    intensities[0] = 1


# Decrement to 0
for i in range(T - 1, 0, -1):
    comm.Barrier()

    if NotAnEdge(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(leftIndex(rank), intensities[T - i - 1])
        Send(rightIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(top, source=up(rank))
        comm.Recv(left, source=leftIndex(rank))

        comm.Recv(right, source=rightIndex(rank))
        comm.Recv(bottom, source=down(rank))
        #print "Rank: %d Left: %d Top: %d Right: %d Bottom: %d" % (rank, leftIndex(rank), up(rank), rightIndex(rank), down(rank))
        
        lastIntensity = intensities[T - i - 1] if (T - i - 1 >= 0) else 0
        lastLastIntensity = intensities[T - i - 2] if (T - i - 2 >= 0) else 0 

        intensities[T - i] = CalculateNewIntensity(top[0], left[0], right[0], bottom[0], lastIntensity, lastLastIntensity, p, n)

    # Calculate Send for edges
    elif IsLeftEdge(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(rightIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(right, source=rightIndex(rank))
        intensities[T - i] = G * right[0]

    elif IsRightEdge(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(leftIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(left, source=leftIndex(rank))
        intensities[T - i] = G * left[0]

    elif IsTopEdge(rank):
        Send(leftIndex(rank), intensities[T - i - 1])
        Send(rightIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(bottom, source=down(rank))
        intensities[T - i] = G * bottom[0]

    elif IsBottomEdge(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(leftIndex(rank), intensities[T - i - 1])
        Send(rightIndex(rank), intensities[T - i - 1])

        comm.Recv(top, source=up(rank))
        intensities[T - i] = G * top[0]

    # Must be a corner
    elif TopLeftCorner(rank):
        Send(rightIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(right, source=rightIndex(rank))
        intensities[T - i] = G * right[0]
        
    elif TopRightCorner(rank):
        Send(leftIndex(rank), intensities[T - i - 1])
        Send(down(rank), intensities[T - i - 1])

        comm.Recv(left, source=leftIndex(rank))
        intensities[T - i] = G * left[0]

    elif BottomLeftCorner(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(rightIndex(rank), intensities[T - i - 1])

        comm.Recv(top, source=up(rank))
        intensities[T - i] = G * right[0]

    elif BottomRightCorner(rank):
        Send(up(rank), intensities[T - i - 1])
        Send(leftIndex(rank), intensities[T - i - 1])

        comm.Recv(top, source=up(rank))
        intensities[T - i] = G * left[0]

    if Center(rank):
        print "Rank N/2 has Intensities: %d\n" % intensities[T - i]

    comm.Barrier()

#print "Rank %d has Intensities: %s" % (rank, str(intensities)) 