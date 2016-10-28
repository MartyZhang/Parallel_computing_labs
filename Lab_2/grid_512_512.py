#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

import direction_calculation
import edge_detection

# Constants
n = 0.5
p = 1
G = 10
rowLength = 512

T = int(sys.argv[1])
arrayWidth = int(rowLength / size)

# Create a 3D Array with  
intensities = numpy.zeros((arrayWidth, arrayWidth, T), dtype=numpy.int)

# rowLength must be divisible by the number of processes
assert rowLength % size == 0

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

def Send(rank, intensity):
    comm.Send(intensity, dest=rank)

def calculateInArray(intensities, positionX, positionY, time, p, n):
    top = intensities[positionX][positionY - 1][time]
    left = intensities[positionX - 1][positionY][time]
    right = intensities[positionX + 1][positionY][time]
    bottom = intensities[positionX][positionY + 1][time]

    last = intensities[positionX][positionY][time - 1]
    lastLast = intensities[positionX][positionY][time - 2]

    CalculateNewIntensity(top, left, right, bottom, last, lastLast, p, n)

def CalculateNewIntensity(top, left, right, bottom, lastTimeStep, lastLastTimeStep, p, n):
    return (p*(top + left + right + bottom - rowLength * lastTimeStep) + 2 * lastTimeStep - (1-n)*lastLastTimeStep)/(1 + n)

if IsCenterProcess(rank, size):
    intensities[int(arrayWidth / 2)][int(arrayWidth / 2)][0] = 1

# Decrement to 0
for i in range(T - 1, 0, -1):
    comm.Barrier()

    if IsNotEdgeProcess(rank, size):
        
        
    elif IsTopLeftCorner(rank, size):

    elif IsTopRightCorner(rank, size):

    elif IsBottomLeftCorner(rank, size):

    elif IsBottomRightCorner(rank, size):

    elif IsLeftEdge(rank, size):

    elif IsRightEdge(rank, size):

    elif IsBottomEdge(rank, size):

    elif IsTopEdge(rank, size):
    
    comm.Barrier()