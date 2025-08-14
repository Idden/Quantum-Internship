import math
from scipy.sparse import csr_matrix
import numpy as np

# task 1: make function that turns binary to decimal
def binToDeci(num):
    
    startBase = math.pow(2, len(num)-1)
    deciNum = 0

    for i in range(len(num)):
        deciNum += int(num[i]) * startBase
        startBase /= 2
        
    return int(deciNum)

# task 2: no consecutive ones in binary sequence
def binNoConsecOnes(N):

    if N == 0:
        return

    listNoConsecOnes = []
    consecOnes = None

    for i in range(int(math.pow(2, N))):
        
        currNum = str(bin(i))[2:]
        consecOnes = False

        if len(currNum) == 1:
            listNoConsecOnes.append(currNum)
            continue

        for j in range(len(currNum)-1):

            if currNum[j] == currNum[j+1] and currNum[j] == '1':
                consecOnes = True
                break
        
        if consecOnes:
            continue

        listNoConsecOnes.append(currNum)

    return listNoConsecOnes

# task 2: redo
def binNoConsecOnesEfficient(N):

    def recursiveBin(n, prevNum, currNum):

        #print(currNum, 'b')
        
        if n == 0:
            listNoConsecOnes.append(currNum)
            return
        
        recursiveBin(n - 1, '0', currNum + '0')

        #print(currNum, 'a')
        
        if prevNum != '1':
            recursiveBin(n - 1, '1', currNum + '1')

    listNoConsecOnes = []
    recursiveBin(N, None, '')
    
    return listNoConsecOnes

# task 3: sparse matrix set up
N = 5
basisList = binNoConsecOnesEfficient(N)
basisMap = {bitStr: i for i, bitStr in enumerate(basisList)}
basisLen = len(basisList)
flippedList = []

row = []
column = []

#print(basisList)

# function to flip bits
def flipBit(bit):

    if bit == '0':
        return '1'
    
    if bit == '1':
        return '0'

for i in range(basisLen):

    # add padding so that search doesnt go out of range
    paddedBitStr = '0' + basisList[i] + '0'
    copyBit = list(paddedBitStr)

    # apply the sum of r P_r-1 * sigma_x * P_r+1 operator
    for j in range(1, N+1):
        
        if paddedBitStr[j-1] == '0' and paddedBitStr[j+1] == '0':
            copyBit[j] = flipBit(paddedBitStr[j])
            flippedList.append(''.join(copyBit)[1:-1])
            copyBit = list(paddedBitStr)
        
    #print(flippedList)

    # adds row and column values for the sparse matrix
    for k in range(len(flippedList)):
        
        row.append(basisMap[flippedList[k]])
        column.append(i)
        
    flippedList.clear()

onesList = np.ones(len(row), dtype=int)

#print(row, column)

sparseHamiltonian = csr_matrix((onesList, (row, column)), shape=[basisLen, basisLen])
matrixHamiltonian = sparseHamiltonian.toarray()
#print(matrixHamiltonian)
