import shlex, os, sys
import numpy as np
import math as mathf
import sgsearch as sgs

from operator import itemgetter

nProc = 1                       #processor数

#  objectFunc
#    目的関数の値を取得
def objectFunc(x):

    x=x.reshape(-1,1)

#    return ((A+1.)**2 * (B+1.)**2 * (C-1.)**2 * (D+1.)**2 * (A+1.)**2 * (B+1.)**2 * (C+1.)**2 * (D-1.)**2)
#    f=0.0
#    for i in range(300):
#        f+=  -mathf.exp(-1.*((x[0]+1.)**2 + (x[1]-1.)**2 + (x[2]+1.)**2 + (x[3]-1.)**2)) - mathf.exp(-1.*((x[0]-1.)**2 + (x[1]+1.)**2 + (x[2]-1.)**2 + (x[3]+1.)**2)) 

    return ( -mathf.exp(-1.*((x[0]+1.)**2 + (x[1]-1.)**2 + (x[2]+1.)**2 + (x[3]-1.)**2)) - mathf.exp(-1.*((x[0]-1.)**2 + (x[1]+1.)**2 + (x[2]-1.)**2 + (x[3]+1.)**2)) )
#    return f

def objectFunc2(x):

    x=x.reshape(-1,1)

#    return ((A+1.)**2 * (B+1.)**2 * (C-1.)**2 * (D+1.)**2 * (A+1.)**2 * (B+1.)**2 * (C+1.)**2 * (D-1.)**2)
#    f=0.0
#    for i in range(300):
#        f+=  -mathf.exp(-1.*((x[0]+1.)**2 + (x[1]-1.)**2 + (x[2]+1.)**2 + (x[3]-1.)**2)) - mathf.exp(-1.*((x[0]-1.)**2 + (x[1]+1.)**2 + (x[2]-1.)**2 + (x[3]+1.)**2)) 

    return ( -mathf.exp(-1.*((x[0]+1.)**2 + (x[3]-1.)**2 + (x[10]+1.)**2 + (x[15]-1.)**2)) - mathf.exp(-1.*((x[1]-1.)**2 + (x[5]+1.)**2 + (x[8]-1.)**2 + (x[12]+1.)**2)) )
#    return f

def objectFunc3(x):

    x=x.reshape(-1,1)

#    return ((A+1.)**2 * (B+1.)**2 * (C-1.)**2 * (D+1.)**2 * (A+1.)**2 * (B+1.)**2 * (C+1.)**2 * (D-1.)**2)
#    f=0.0
#    for i in range(300):
#        f+=  -mathf.exp(-1.*((x[0]+1.)**2 + (x[1]-1.)**2 + (x[2]+1.)**2 + (x[3]-1.)**2)) - mathf.exp(-1.*((x[0]-1.)**2 + (x[1]+1.)**2 + (x[2]-1.)**2 + (x[3]+1.)**2)) 

    return ( -mathf.exp(-1.*((x[0]+1.)**2 + (x[3]-1.)**2 + (x[10]+1.)**2 + (x[15]-1.)**2)) - mathf.exp(-1.*((x[0]-1.)**2 + (x[3]+1.)**2 + (x[10]-1.)**2 + (x[15]+1.)**2)) )
#    return f


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        if sys.argv[1] == "-np":
            try:
                nProc = int(sys.argv[2])
            except:
                pass
    sgs.nProc = nProc               #processor数を設定
    result = sgs.pascal(func=objectFunc,
#                        lowerx=np.array([-2., -2., -2., -2.,-2., -2., -2., -2.,-2., -2., -2., -2.,-2., -2., -2., -2.]),
#                        upperx=np.array([2., 2., 2., 2.,2., 2., 2., 2.,2., 2., 2., 2.,2., 2., 2., 2.]),
                        lowerx=np.array([-2., -2., -2., -2.]),
                        upperx=np.array([2., 2., 2., 2.]),
                        Cands=2,
                        pattern="cross1_multi1",
                        depth=40,
                        linearmode_start=21)

#    result = sgs.pascal(func=objectFunc,
#                        lowerx=np.array([-2., -2., -2., -2.]),
#                        upperx=np.array([2., 2., 2., 2.]),
#                        Cands=2,
#                        pattern="all",
#                        depth=40,
#                        linearmode_start=21)

#    print(result)
    print(sgs.max_value[0])
    print(sgs.max_position[0])
    print(sgs.max_value[1])
    print(sgs.max_position[1])
