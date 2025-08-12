import math

binNum = "10010"

def binToDeci(num):
    
    startBase = math.pow(2, len(num))
    deciNum = 0

    for i in range(len(num)):
        deciNum += int(num) * startBase
        startBase /= 2

    return deciNum



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

    print(listNoConsecOnes)
