import shlex, os, sys
import numpy as np
import math as mathf
import sgsearch as sgs

from operator import itemgetter


#測定点のデータ形式
class Data():
    def __init__(self,pres,vol):
        self.pres=pres
        self.vol=vol

class Candidate():
    def __init__(self,idnum,indx,value,vel,force,depth):
        self.idnum=idnum
        self.indx=indx
        self.value=value
        self.vel=vel
        self.force=force
        self.depth=depth

#  objectFunc
#    目的関数の値を取得
def objectFunc(x):

    score=0.#測定数nからの2乗誤差の合計
    DataS = []#測定点のデータ入れ
    DataS.append(Data(pres=5.0,vol=0.707415))
    DataS.append(Data(pres=10.0,vol=1.003945))
    DataS.append(Data(pres=15.0,vol=1.563031))
    DataS.append(Data(pres=20.0,vol=2.296063))

#    DataS.append(Data(pres=5.0,vol=0.610435))
#    DataS.append(Data(pres=10.0,vol=1.075766))
#    DataS.append(Data(pres=15.0,vol=1.716214))
#    DataS.append(Data(pres=20.0,vol=2.42213))

    
    for dat in DataS:#各測定点と理論式からの誤差を合算する
        #print(dat.pres)
        #print(dat.vol)
        score=score + error_calc(x,dat)
    return score

def error_calc(x,data):
    error = 0.0
    x=x.reshape(-1,1)
    A = x[0]
    B = x[1]
    C = x[2]
    D = x[3]
    error = (A+B/(1+mathf.exp(-(data.pres-C)/D)) - data.vol)**2
    return error


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        if sys.argv[1] == "-np":
            try:
                nProc = int(sys.argv[2])
            except:
                pass
    sgs.nProc = nProc               #processor数を設定
    result = sgs.pascal(func=objectFunc,lowerx=np.array([-1., 1., 15., 2.]),upperx=np.array([1., 5., 30., 10.]),Cands=1,pattern="all", depth=70,linearmode_start=21)
    print(result)
    print(sgs.max_value[0])
    print(sgs.max_position[0])

'''
    result = sgs.pascal(func=objectFunc,
                        lowerx=np.array([-2., -2., -2., -2.]),
                        upperx=np.array([2., 2., 2., 2.]),
                        Cands=2,
                        pattern="all",
                        depth=50,
                        linearmode_start=51)
    print(result)
'''

#result = sgs.pascal(func=objectFunc,lowerx=np.array([-1., 1., 15., 2.]),upperx=np.array([1., 5., 30., 10.]),Cands=1,pattern="all", depth=100,linearmode_start=51)

