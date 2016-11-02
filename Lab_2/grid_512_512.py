#!/usr/bin/env python
from mpi4py import MPI
import numpy
import sys

# Constants
n = 0.0002
p = 0.5
G = 0.75
rowLength = 512

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#divide into rowLength by rowLength/p arrays
T = int(sys.argv[1])
arrayWidth = int(rowLength/size)

# rowLength must be divisible by the number of processes
assert rowLength % size == 0

intensities = numpy.zeros((rowLength, arrayWidth, 3), dtype=numpy.float32)
intensities = numpy.zeros((rowLength, arrayWidth, 3), dtype=numpy.float32)

#we're trying to find the 257th slot in the array since [256][256] is technically 257,257
#subtract by 1 to get correct index
slot = rowLength/2 + 1
offset = (slot)%arrayWidth - 1

#special rank is the process which contains 257,257
specialRank = int(slot/arrayWidth)
#initialize that slot to 1
if(rank==specialRank):
    if(rank==specialRank):
        print "%f" % intensities[slot-1][offset][0]
    #initialize the first drum hit
    intensities[slot-1][offset][0] = 1.0
    intensities[slot-1][offset][1] = 1.0
#buffers to send and receive
leftArray = numpy.zeros((rowLength), dtype=numpy.float32)
rightArray = numpy.zeros((rowLength), dtype=numpy.float32)

leftArrayReceive = numpy.zeros((rowLength), dtype=numpy.float32)
rightArrayReceive = numpy.zeros((rowLength), dtype=numpy.float32)

#from q1
def Send(rank, intensity, tag):
    comm.Send(intensity, dest=rank, tag=tag)
#from q1
def CalculateNewIntensity(top, left, right, bottom, lastTimeStep, lastLastTimeStep, p, n):
    return (p*(top + left + right + bottom - 4.0 * lastTimeStep) + 2.0 * lastTimeStep - (1.0-n)*lastLastTimeStep)/(1.0 + n)

#tags
right = 0
left = 1



for i in range(0, T):
    if(size==1):
        break
    comm.Barrier()
    #print the intensity every time
    if(rank==specialRank):
        print "%f" % intensities[slot - 1][offset][0]

    #first process special only sends to its right
    if(rank==0):
        #fill out buffer
        for i in range(0, rowLength):
            rightArray[i] = intensities[i][arrayWidth-1][1]
        #send buffer
        Send(rank+1, rightArray, right)
        #receive
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
    #last process also special only sends to its left
    elif(rank==size-1):
        #fill out buffer
        for i in range(0, rowLength):
            leftArray[i] = intensities[i][0][1]
        #send buffer
        Send(rank-1, leftArray, left)
        #receive
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)
    #everything else sends both ways
    else:
        for i in range(0, rowLength):
            leftArray[i] = intensities[i][0][1]
            rightArray[i] = intensities[i][arrayWidth-1][1]
        Send(rank-1, leftArray, left)
        Send(rank+1, rightArray, right)
        comm.Recv(rightArrayReceive, source=rank+1, tag = left)
        comm.Recv(leftArrayReceive, source=rank-1, tag = right)

    #wait for everyone to send and receive before proceeding
    for i in range(0, rowLength):
        for j in range (0, arrayWidth):
            #first rank contains left edges
            if(rank==0):
                #pass if its top row, bottom row or if the left side edges
                if(i==0 or i==rowLength-1 or j==0):
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
                if(i==0 or i==rowLength-1 or j==arrayWidth-1):
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
                if(i==0 or i==rowLength-1):
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

    #update edges
    if(rank==0):
        #top and bottom rows
        for i in range(1, arrayWidth):
            intensities[0][i][0]= G*intensities[1][i][0]
            intensities[rowLength-1][i][0] = G*intensities[rowLength-2][i][0]
        #sides
        for k in range(1, rowLength-1):
            intensities[k][0][0] = G*intensities[k][1][0]
    elif(rank==size-1):
        #top and bottom rows
        for i in range(0, arrayWidth-1):
            intensities[0][i][0] = G*intensities[1][i][0]
            intensities[rowLength-1][i][0] = G*intensities[rowLength-2][i][0]
        #sides
        for k in range(1, rowLength-1):
            intensities[k][arrayWidth-1][0] = G*intensities[k][arrayWidth-2][0]
    else:
        for i in range(0, arrayWidth):
            intensities[0][i][0] = G*intensities[1][i][0]
            intensities[rowLength-1][i][0] = G*intensities[rowLength-2][i][0]

    #update corners
    if(rank==0):
        intensities[0][0][0] = G*intensities[1][0][0]
        intensities[rowLength-1][0][0] = G*intensities[rowLength-2][0][0]
    elif(rank==size-1):
        intensities[0][arrayWidth-1][0] = G*intensities[0][arrayWidth-2][0]
        intensities[rowLength-1][arrayWidth-1][0] = G*intensities[rowLength-1][arrayWidth-2][0]
    #push everything down

    for i in range(0,rowLength):
        for j in range (0, arrayWidth):
            intensities[i][j][2] = intensities[i][j][1]
            intensities[i][j][1] = intensities[i][j][0]

for i in range(0, T):
    if(size!=1):
        break
    #print the intensity every time
    if(rank==specialRank):
        print "%f" % intensities[slot - 1][offset][0]    

    #wait for everyone to send and receive before proceeding
    for i in range(0, rowLength):
        for j in range (0, arrayWidth):
            #first rank contains left edges                #pass if its top row, bottom row or if the left side edges
            if(i==0 or i==rowLength-1 or j==0 or j==rowLength-1):
                pass
            else:
                top = intensities[i-1][j][1]
                bottom = intensities[i+1][j][1]
                right = intensities[i][j+1][1]
                left = intensities[i][j-1][1]
                last = intensities[i][j][1]
                last2 = intensities[i][j][2]
                intensities[i][j][0] = CalculateNewIntensity(top, left, right, bottom,last, last2, p, n)
    #update edges
    for k in range(1, rowLength-1):
        intensities[k][0][0] = G*intensities[k][1][0]
        #top and bottom rows
    for i in range(1, arrayWidth-1):
        intensities[0][i][0] = G*intensities[1][i][0]
        intensities[rowLength-1][i][0] = G*intensities[rowLength-2][i][0]
        #sides
    for k in range(1, rowLength-1):
        intensities[k][arrayWidth-1][0] = G*intensities[k][arrayWidth-2][0]

    #update corners
    intensities[0][0][0] = G*intensities[1][0][0]
    intensities[rowLength-1][0][0] = G*intensities[rowLength-2][0][0]
    intensities[0][arrayWidth-1][0] = G*intensities[0][arrayWidth-2][0]
    intensities[rowLength-1][arrayWidth-1][0] = G*intensities[rowLength-1][arrayWidth-2][0]
    #push everything down

    for i in range(0,rowLength):
        for j in range (0, arrayWidth):
            intensities[i][j][2] = intensities[i][j][1]
            intensities[i][j][1] = intensities[i][j][0]