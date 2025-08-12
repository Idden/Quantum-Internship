import math

binNum = "10010"

startBase = math.pow(2, len(binNum)-1)
deciNum = 0

for i in range(len(binNum)):
    deciNum += int(binNum[i]) * startBase
    startBase /= 2
    
print(deciNum)
print(int(binNum, 2))