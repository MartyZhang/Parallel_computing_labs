#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

import direction_calculation
import edge_detection

# Constants
n = 0.0002
p = 0.5
G = 0.75
rowLength = 512


# Create a 3D Array with  

# rowLength must be divisible by the number of processes

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#divide into 512 by 512/p arrays
T = int(sys.argv[1])
arrayWidth = int(rowLength/size)
assert rowLength % size == 0
intensities = numpy.zeros((rowLength, arrayWidth, 3), dtype=numpy.float32)

#we're trying to find the 257th slot in the array since [256][256] is technically 257,257
#subtract by 1 to get correct index
offset = 257%arrayWidth - 1
#special rank is the process which contains 257,257
specialRank = int(257/arrayWidth)
#initialize that slot to 1
if(rank==specialRank):
    if(rank==specialRank):
        print "Rank N/2 has Intensities: %f\n" % intensities[256][offset][0]
    #initialize the first drum hit
    intensities[256][offset][0] = 1.0
    intensities[256][offset][1] = 1.0
#buffers to send and receive
leftArray = numpy.zeros((512), dtype=numpy.float32)
rightArray = numpy.zeros((512), dtype=numpy.float32)

leftArrayReceive = numpy.zeros((512), dtype=numpy.float32)
rightArrayReceive = numpy.zeros((512), dtype=numpy.float32)

#from q1
def Send(rank, intensity, tag):
    comm.Send(intensity, dest=rank, tag=tag)
#from q1
def CalculateNewIntensity(top, left, right, bottom, lastTimeStep, lastLastTimeStep, p, n):
    return (p*(top + left + right + bottom - 4 * lastTimeStep) + 2 * lastTimeStep - (1-n)*lastLastTimeStep)/(1 + n)

#tags
right = 0
left = 1

for i in range(0, T):
    #print the intensity every time
    if(rank==specialRank):
        print "Rank N/2 has Intensities: %f\n" % intensities[256][offset][0]

    #first process special only sends to its right
    if(rank==0):
        #fill out buffer
        for i in range(0, 512):
            rightArray[i] = intensities[i][arrayWidth-1][1]
        #send buffer
        Send(rank+1, rightArray, right)
        #receive
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
    #last process also special only sends to its left
    elif(rank==size-1):
        #fill out buffer
        for i in range(0, 512):
            leftArray[i] = intensities[i][0][1]
        #send buffer
        Send(rank-1, leftArray, left)
        #receive
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)
    #everything else sends both ways
    else:
        for i in range(0, 512):
            leftArray[i] = intensities[i][0][1]
            rightArray[i] = intensities[i][arrayWidth-1][1]
        Send(rank-1, leftArray, left)
        Send(rank+1, rightArray, right)
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)

    #wait for everyone to send and receive before proceeding
    comm.Barrier()
    for i in range(0, 512):
        for j in range (0, arrayWidth):
            #first rank contains left edges
            if(rank==0):
                #pass if its top row, bottom row or if the left side edges
                if(i==0 or i==511 or j==0):
                    pass
                elif(j==(arrayWidth-1)):
                    top = intensities[i-1][j][1]
                    right = rightArrayReceive[i]
                    left = intensities[i][j-1][1]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
                else: 
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
            #last rank contains right edges
            elif(rank==size-1):
                #pass if its top row, bottom row, or right side edges
                if(i==0 or i==511 or j==arrayWidth-1):
                    pass
                elif (j==(0)):
                    #get from buffer
                    top = intensities[i-1][j][1]
                    right = intensities[i][j+1][1]
                    left = leftArrayReceive[i]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
                else: 
                    #normal processing
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
            else:
                #pass if its top or bottom row
                if(i==0 or i==511):
                    pass
                elif (j==(0)):
                    #get from buffer
                    top = intensities[i-1][j][1]
                    right = intensities[i][j+1][1]
                    left = leftArrayReceive[i]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
                elif (j==(arrayWidth-1)):
                    #get from buffer
                    top = intensities[i-1][j][1]
                    right = rightArrayReceive[i]
                    left = intensities[i][j-1][1]
                    bottom = intensities[i+1][j][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p , n)
                else: 
                    #normal processing
                    top = intensities[i-1][j][1]
                    bottom = intensities[i+1][j][1]
                    right = intensities[i][j+1][1]
                    left = intensities[i][j-1][1]
                    last = intensities[i][j][1]
                    last2 = intensities[i][j][2]
                    intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2,p ,n)

    comm.Barrier()
    #update edges
    if(rank==0):
        #top and bottom rows
        for i in range(1, arrayWidth):
            intensities[0][i][0]= G*intensities[1][i][0]
            intensities[511][i][0] = G*intensities[510][i][0]
        #sides
        for k in range(1, 511):
            intensities[k][0][0] = intensities[k][1][0]
    elif(rank==size-1):
        #top and bottom rows
        for i in range(0, arrayWidth-1):
            intensities[0][i][0] = G*intensities[1][i][0]
            intensities[511][i][0] = G*intensities[510][i][0]
        #sides
        for k in range(1, 511):
            intensities[k][arrayWidth-1][0] = intensities[k][arrayWidth-2][0]
    else:
        for i in range(0, arrayWidth):
            intensities[0][i][0] = G*intensities[1][i][0]
            intensities[511][i][0] = G*intensities[510][i][0]

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
