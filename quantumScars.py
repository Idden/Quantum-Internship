import math
from scipy.sparse import csr_matrix

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

            if currNum[j] == currNum[j+1] and currNum[j] == "1":
                consecOnes = True
                break
        
        if consecOnes:
            continue

        listNoConsecOnes.append(currNum)

    return listNoConsecOnes

# task 3: sparse matrix set up
sparseHamiltonian = csr_matrix(([0], ([0], [0])), shape=[100, 100])
