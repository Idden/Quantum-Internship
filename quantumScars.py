# test comment push

import math

binNum = "10010"

def binToDeci(num):
    
    startBase = math.pow(2, len(num))
    deciNum = 0

    for i in range(len(num)):
        deciNum += int(num) * startBase
        startBase /= 2

    return deciNum



