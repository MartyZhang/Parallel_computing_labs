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


# Create a 3D Array with  

# rowLength must be divisible by the number of processes

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#divide into 512 by 512/p arrays
Time = int(sys.argv[1])
arrayWidth = int(rowLength/size)
assert rowLength % size == 0
intensities = numpy.zeros((rowLength, arrayWidth, 3), dtype=numpy.float32)
#we're trying to find the 257th slot in the array since [256][256] is technically 257,257
#subtract by 1 to get correct index
offset = 257%arrayWidth - 1

def Send(rank, intensity):
    comm.Send(intensity, dest=rank)

specialRank = int(257/arrayWidth)
if(rank==specialRank):
    #initialize the first drum hit
    intensities[256][offset][0] = 1.0
    intensities[256][offset][1] = 1.0
#buffers to send and receive
leftArraySend = numpy.zeros((512), dtype=numpy.float32)
rightArraySend = numpy.zeros((512), dtype=numpy.float32)

leftArrayReceive = numpy.zeros((512), dtype=numpy.float32)
rightArrayReceive = numpy.zeros((512), dtype=numpy.float32)

def CalculateNewIntensity(top, left, right, bottom, lastTimeStep, lastLastTimeStep, p, n):
    return (p*(top + left + right + bottom - 4 * lastTimeStep) + 2 * lastTimeStep - (1-n)*lastLastTimeStep)/(1 + n)

#tags
right = 0
left = 1

for i in range(0, T):
    #first process special case
    if(rank==0):
        #fill out buffer
        for i in range(0, 512):
            rightArray[i] = intensities[i][arrayWidth-1][1]
        #send buffer
        Send(1, rightArray, tag = right)
        #receive
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
    elif(rank==size-1):
        #fill out buffer
        for i in range(0, 512):
            leftArray[i] = intensities[i][0][1]
        #send buffer
        Send(rank-1, leftArray, tag = left)
        #receive
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)
    else:
         for i in range(0, 512):
            leftArray[i] = intensities[i][0][1]
            rightArray[i] = intensities[i][arrayWidth-1][1]
        Send(rank-1, leftArray, tag = left)
        Send(rank+1, rightArray, tag = right)
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)

    comm.Barrier()
    for i in range(0, 512):
        for j in range (0, arrayWidth):
            if(rank==0):
                if(i==0||j==0||i==511):
                    #do nothing
                elif (j==(arrayWidth-1)):
                    top = intensities[i-1][j][1]
                    right = rightArrayReceive[i]
                    left = intensities[i][j-1][1]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
                else: 
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
            elif(rank==size-1):
                 if(i==0||j==arrayWidth-1||i==512):
                    #do nothing
                elif (j==(0)):
                    #get from buffer
                    top = intensities[i-1][j][1]
                    right = intensities[i][j+1][1]
                    left = leftArrayReceive[i]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
                else: 
                    #normal processing
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
            else:
                if(i==0||i==511):
                    #do nothing for top and bottom rows
                elif (j==(0)):
                    #get from buffer
                    top = intensities[i-1][j][1]
                    right = intensities[i][j+1][1]
                    left = leftArrayReceive[i]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
                elif (j==(arrayWidth-1)):
                    top = intensities[i-1][j][1]
                    right = rightArrayReceive[i]
                    left = intensities[i][j-1][1]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)
                else: 
                    #normal processing
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2)

                if(rank==specialRank):
                    print "Rank N/2 has Intensities: %f\n" % intensities[256][offset][0]

    comm.Barrier()
    #update edges
    if(rank==0):
        #top and bottom rows
        for i in range(1, arrayWidth):
            intensities[0][i][0]= G*intensities[0][i-1][0]
            intensities[511][i][0] = G*intensities[0][i+1][0]
        #sides
        for k in range(1, 511):
            intensities[k][0][0] = intensities[k][1][0]
    elif(rank==size-1):
        #top and bottom rows
        for i in range(0, arrayWidth-1):
            intensities[0][i][0] = G*intensities[0][i-1][0]
            intensities[511][i][0] = G*intensities[0][i+1][0]
        #sides
        for k in range(1, 511):
            intensities[k][arrayWidth-1][0] = intensities[k][arrayWidth-2][0]
    else:
        for i in range(0, arrayWidth):
            intensities[0][i][0] = G*intensities[0][i-1][0]
            intensities[511][i][0] = G*intensities[0][i+1][0]

    comm.Barrier()
    #update corners
    if(rank==0):
        intensities[0][0][0] = G*intensities[1][0][0]
        intensities[511][0][0] = G*intensities[510][0][0]
    elif(rank==size-1):
        intensities[0][arrayWidth-1][0] = G*intensities[0][arrayWidth-2][0]
        intensities[511][arrayWidth-1][0] = G*intensities[510][arrayWidth-2][0]
    #push everything down
    for i in range(0,512):
        for j in range (0, arrayWidth):
            intensities[i][j][2] = intensities[i][j][1]
            intensities[i][j][1] = intensities[i][j][0]
