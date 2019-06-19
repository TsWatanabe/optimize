
import shlex, os, sys
import numpy as np
import math as mathf
import re

from multiprocessing import Process, Queue

from operator import itemgetter

nProc = 1           #processor数定義（モジュールの外から値を設定する）
final_result  = 0.
max_value = []
max_position = []

class Candidate():
    def __init__(self,idnum,indx,value,maxindx,maxvalue,prev_move1,prev_move2,depth):
        self.idnum=idnum
        self.indx=indx
        self.value=value
        self.maxindx=maxindx
        self.maxvalue=maxvalue
        self.prev_move1=prev_move1
        self.prev_move2=prev_move2
        self.depth=depth


def convertLists(tryindx, i,upperx,lowerx):
    return ((upperx-lowerx)*float(tryindx+1)/(i+2) + lowerx)

def onebit_rev(in_array,j):
    for k in range(0,len(in_array)):
        if(k==j):
            if (in_array[k]==0):
                in_array[k]=1
            else:
                in_array[k]=0
    return in_array

def move_mode(prev_move1,prev_move2):
    x = np.empty(len(prev_move1),dtype=int)
    for k in range(0,len(prev_move1)):
        if(prev_move1[k]==0 and prev_move2[k]==0):
            x[k]=0
        elif(prev_move1[k]==0 and prev_move2[k]==1):
            x[k]=1
        elif(prev_move1[k]==1 and prev_move2[k]==0):
            x[k]=0
        else:
            x[k]=1
    return x

def distance(a,b):
    i = 0
    for k in range(0,len(a)):
        if(a[k] != b[k]):
            i = i+1
    return i

def procJob2(func, cand, i, newN, upperx, lowerx, q):
    result_label = []
    for k in newN:
        tryindx = cand.indx - cand.prev_move1 + onebit_rev(cand.prev_move1,k)
        #             print(onebit_rev(cand.prev_move1,k))
        print(tryindx)
        tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        res = func(tryx)
        result_label.append([res,onebit_rev(cand.prev_move1,k)])
        print(res)
    q.put(result_label)

def calcMultiProc2(func, cand, i, N, upperx, lowerx):
    listN = list(range(N))
    newNs = getDistedList(listN, nProc)
    processes = []; queues = []
    for n in range(nProc):
        newN = newNs[n]
        q = Queue()
        p = Process(target=procJob2, args=(func, cand, i, newN, upperx, lowerx, q))
        processes.append(p)
        queues.append(q)
        p.start()
    result_label = []
    for n in range(nProc):
        processes[n].join()
        result = queues[n].get()
        result_label += result
    return [result_label, listN]

def onebitrev_search(func,Candidates,i,N,upperx,lowerx,Cands):


    result_label = []
    for cand in Candidates:
        print(cand.idnum)
        scoreup_param = 0

        linear = cand.prev_move1.copy()
        result_label.append([cand.value,cand.prev_move1])

        [result_label, kList] = calcMultiProc2(func, cand, i, N, upperx, lowerx)
        for n in range(len(result_label)):
            if result_label[n][0] < cand.value:
                k = kList[n]
                scoreup_param = scoreup_param + 1
                linear[k] = onebit_rev(cand.prev_move1,k)[k]

#        for k in range (0,N):
#            tryindx = cand.indx - cand.prev_move1 + onebit_rev(cand.prev_move1,k)
            #             print(onebit_rev(cand.prev_move1,k))
#            print(tryindx)
#            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
#            res = func(tryx)
#            result_label.append([res,onebit_rev(cand.prev_move1,k)])
#            print(res)
#            if res < cand.value:
#                scoreup_param = scoreup_param + 1
#                linear[k] = onebit_rev(cand.prev_move1,k)[k]
                #        print(linear)

        if (scoreup_param > 1):
            tryindx = cand.indx - cand.prev_move1 + linear
            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
            res = func(tryx)
            result_label.append([res,linear])
            print(cand.prev_move1)
            print(tryindx)
            print(tryx)
            print(res)

        result_label.sort(key=itemgetter(0))
        cand.indx = cand.indx - cand.prev_move1 + result_label[0][1]
        cand.value = result_label[0][0]
        cand.prev_move2 = result_label[0][1]
        cand.depth = i
        cand.prev_move1 = result_label[0][1]
        result_label.clear()

def getDistSize(value, n):
    delta = int(value / n)
    dVals = [delta for i in range(n)]
    mod = value - (delta * n)
    i = 0
    while i < mod:
        dVals[i] += 1
        i += 1
    return dVals

def getDistedList(dataList, n):
    """dataListを均等にn分割したリストを返す。"""
    dVals = getDistSize(len(dataList), n)
    distedList = []
    st = 0
    for n in dVals:
        distedList.append(dataList[st:st+n])
        st += n
    return distedList

#  各processのjob
def procJob(n, func, cand, i, procLines, lowerx, upperx, q):
    """各processorが処理するjobの内容。"""
    result = []
    for move in procLines:
        print(move)
        tryindx = cand.indx + move
        tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        res = func(tryx)
        # print("proc:"+str(n), tryindx)
        # print("proc:"+str(n), tryx)
        print("proc:"+str(n), res)
        result.append([res,move])
    q.put(result)

    #  並列処理
def calcMultiProc(func, cand, i, lines_numpy, lowerx, upperx):
    """nProc（processor数）毎に処理を分割して、並列処理させる。"""
    #各process毎にlines_numpyを分配する
    newLines = getDistedList(lines_numpy, nProc)
    #並列処理
    processes = []; queues = []
    #  各process分jobを投入
    for n in range(nProc):
        procLines = newLines[n]     #各processに渡すlines_numpy
        q = Queue()
        p = Process(target=procJob, args=(n, func, cand, i, procLines, lowerx, upperx, q))
        processes.append(p)
        queues.append(q)
        p.start()
    #  各process分の結果を取得
    result_label = []
    for n in range(nProc):
        processes[n].join()         #n番目のprocessが終了するまで待つ
        result = queues[n].get()    #n番目のprocessの結果を取得
        result_label += result
    return result_label


def comb_search(func,Candidates,i,lines_numpy,N,upperx,lowerx,Cands,pattern):

    #--- comb_search(main) ---
    result_label = []
    for cand in Candidates:
        print(cand.idnum)
        result_label = calcMultiProc(func, cand, i, lines_numpy, lowerx, upperx)
        # for move in lines_numpy:        #risuto
        #     print(move)
        #     tryindx = cand.indx + move
        #     tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        #     res = func(tryx)
        #     print(tryindx)
        #     print(tryx)
        #     print(res)
        #     result_label.append([res,move])
        result_label.sort(key=itemgetter(0))
        cand.indx = cand.indx + result_label[0][1]
        cand.value = result_label[0][0]
        cand.prev_move2 = cand.prev_move1
        cand.depth  = i+1
        cand.prev_move1 = result_label[0][1]
        if (i==1):
            cand.maxindx = cand.indx
            cand.maxvalue = cand.value
        else:
            if (cand.maxvalue > result_label[0][0]):
                cand.maxindx = cand.indx
                cand.max_value = cand.value              

        if(pattern !="all"):
            if(i==1):
                for j in range(1,Cands):
                    Candidates.append(Candidate(idnum=j,indx= result_label[j][1],prev_move1=result_label[j][1],value=result_label[j][0],prev_move2=result_label[j][1],depth=i+1,maxindx=result_label[0][1],maxvalue=result_label[0][0]))
                onebitrev_search(func,Candidates,i,N,upperx,lowerx,Cands)
                result_label.clear()
                return
            else:
                onebitrev_search(func,Candidates,i,N,upperx,lowerx,Cands)

        else:
            if(i==1):
                temp = cand.prev_move1
#        temp_result_label = result_label.copy()
                for j in range(1,Cands):
#            for k in range(0,N):
#                result_label.delete(onebit_rev(cand.prev_move1,j))
                    result_label = [e for e in result_label if distance(e[1],temp) > 1]
                    result_label.sort(key=itemgetter(0))
#                temp = result_label[0][1]
                    Candidates.append(Candidate(idnum=j,indx= result_label[0][1],prev_move1=result_label[0][1],value=result_label[0][0],prev_move2=result_label[0][1],depth=i+1,maxindx=result_label[0][1],maxvalue=result_label[0][0]))
                result_label.clear()
                return
        result_label.clear()

def procJob3(func, cand, i, newN, upperx, lowerx,linear ,q):
    result_label = []
    for k in newN:
        tryindx = cand.indx + onebit_rev(linear,k)
#        print(tryindx)
        tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        res = func(tryx)
        resArray = onebit_rev(linear,k)
        result_label.append([res,resArray])
#        print(res)
    q.put(result_label)

def calcMultiProc3(func, cand, i, N, upperx, lowerx,linear):
    listN = [j for j in range(N)]
    newNs = getDistedList(listN, nProc)
    processes = []; queues = []
    for n in range(nProc):
        newN = newNs[n]
        q = Queue()
        p = Process(target=procJob3, args=(func, cand, i, newN, upperx, lowerx,linear, q))
        processes.append(p)
        queues.append(q)
        p.start()
    result_label = []
    for n in range(nProc):
        processes[n].join()
        result = queues[n].get()
        result_label += result
    return [result_label, listN]


def neib_serach(func,Candidates,i,N,upperx,lowerx):

    result_label = []
    for cand in Candidates:
        print(cand.idnum)
        scoreup_param = 0

        linear = move_mode(cand.prev_move1,cand.prev_move2).copy()        
        tryindx = cand.indx + linear
        tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        res = func(tryx)
        result_label.append([res,linear])
#        print(linear)
#        print(tryindx)
#        print(tryx)
#        print(res)

        [result_label, kList] = calcMultiProc3(func, cand, i, N, upperx, lowerx,linear)
        for n in range(len(result_label)):
            if result_label[n][0] < res:
                k = kList[n]
                linear[k] = onebit_rev(cand.prev_move1,k)[k]
                scoreup_param = scoreup_param + 1

        #for k in range (0,N):
        #    tryindx = cand.indx + onebit_rev(move_mode(cand.prev_move1,cand.prev_move2),k)
        #    #             print(onebit_rev(cand.prev_move1,k))
        #    print(tryindx)
        #    tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        #    res = func(tryx)
        #    result_label.append([res,onebit_rev(move_mode(cand.prev_move1,cand.prev_move2),k)])
        #    print(res)
        #    if res < result_label[0][0]:
        #        linear[k] = onebit_rev(cand.prev_move1,k)[k]
        #        scoreup_param = scoreup_param + 1
                #         print(linear)

        if (scoreup_param > 1):
            tryindx = cand.indx + linear
            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
            res = func(tryx)
            result_label.append([res,linear])
            print(linear)
            print(tryindx)
            print(tryx)
            print(res)
            result_label.sort(key=itemgetter(0))
            cand.indx = cand.indx + linear
            cand.value = res
            cand.prev_move2 = cand.prev_move1
            cand.depth = i + 1
            cand.prev_move1 = linear

        else:
            result_label.sort(key=itemgetter(0))
            cand.indx = cand.indx + result_label[0][1]
            cand.value = result_label[0][0]
            cand.prev_move2 = cand.prev_move1
            cand.depth = i + 1
            cand.prev_move1 = result_label[0][1]
        if (cand.maxvalue > result_label[0][0]):
            cand.maxindx = cand.indx
            cand.max_value = cand.value
        result_label.clear()

def procJob4(func, cand, i, newN, upperx, lowerx,linear,q):
    result_label = []
    for k in newN:
        tryindx = 2*cand.indx + onebit_rev(linear,k)
        print(tryindx)
        tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
        res = func(tryx)
        resArray = onebit_rev(linear,k)
        result_label.append([res,resArray])
        print(res)
    q.put(result_label)

def calcMultiProc4(func, cand, i, N, upperx, lowerx,linear):
    listN = [j for j in range(N)]
    newNs = getDistedList(listN, nProc)
    processes = []; queues = []
    for n in range(nProc):
        newN = newNs[n]
        q = Queue()
        p = Process(target=procJob4, args=(func, cand, i, newN, upperx, lowerx,linear, q))
        processes.append(p)
        queues.append(q)
        p.start()
    result_label = []
    for n in range(nProc):
        processes[n].join()
        result = queues[n].get()
        result_label += result
    return [result_label, listN]



def neib_serach_acl(func,Candidates,i,N,upperx,lowerx):
    result_label = []
    for cand in Candidates:
        print(cand.idnum)
        scoreup_param = 0

        linear = move_mode(cand.prev_move1,cand.prev_move2).copy() 

        cand.depth = 2*cand.depth

        tryindx = 2*cand.indx + linear
        tryx = np.vectorize(convertLists)(tryindx,cand.depth,lowerx,upperx)
        res = func(tryx)
        result_label.append([res,linear])
        print(linear)
        print(cand.depth)
        print(tryindx)
        print(tryx)
        print(res)


        [result_label, kList] = calcMultiProc4(func, cand, cand.depth, N, upperx, lowerx,linear)
        for n in range(len(result_label)):
            if result_label[n][0] < res:
                k = kList[n]
                linear[k] = onebit_rev(linear,k)[k]
                scoreup_param = scoreup_param + 1


#        for k in range (0,N):
#            tryindx = 2*cand.indx + onebit_rev(move_mode(cand.prev_move1,cand.prev_move2),k)
#                        print(onebit_rev(cand.prev_move1,k))
#            print(tryindx)
#            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
#            res = func(tryx)
#            result_label.append([res,onebit_rev(move_mode(cand.prev_move1,cand.prev_move2),k)])
#            print(res)
#            if res < result_label[0][0]:
#                linear[k] = onebit_rev(move_mode(cand.prev_move1,cand.prev_move2),k)[k]
#                scoreup_param = scoreup_param + 1
#                        print(linear)
        if (scoreup_param > 1):
#            tryindx = 2*cand.indx + linear
#            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
#            res = func(tryx)
#            result_label.append([res,linear])
            print(linear)
            print(tryindx)
            print(tryx)
            print(res)
            result_label.sort(key=itemgetter(0))
            cand.indx = 2*cand.indx + linear
            cand.value = res
#            cand.depth = i + 1
            cand.prev_move2 = cand.prev_move1
            cand.prev_move1 = linear

        else:
            result_label.sort(key=itemgetter(0))
            cand.indx = 2*cand.indx + result_label[0][1]
            cand.value = result_label[0][0]
#            cand.depth = i + 1
            cand.prev_move2 = cand.prev_move1
            cand.prev_move1 = result_label[0][1]
        result_label.clear()

#        if ( scoreup_param > 1):
#            tryindx = 2*cand.indx + linear
#            tryx = np.vectorize(convertLists)(tryindx,i,lowerx,upperx)
#            res = func(tryx)
#            result_label.append([res,linear])
#            print(linear)
#            print(tryindx)
#            print(tryx)
#            print(res)

#        result_label.sort(key=itemgetter(0))
#        cand.indx = 2*cand.indx + result_label[0][1]
#        cand.value = result_label[0][0]
#        cand.prev_move2 = cand.prev_move1
#        cand.prev_move1 = result_label[0][1]
#        result_label.clear()


def pascal(func,upperx,lowerx,depth=20,linearmode_start=11,pattern="all",Cands=1):
    global max_value
    global max_position

    N=len(upperx)
    if (N == 0 or N != len(lowerx)):
        print ("upper or lower is wrong.")
        sys.exit(1)
    if os.path.exists(pattern):
        try:
            f=open(pattern)
        except:
            print("Pattern file is not exist.")
            exit()
        x = f.read()
        f.close()
        lines = [i for i in re.split(r'\n',x) if i != '']
        
        length = len(lines)
        lines_cut = []
        lines_merge = []
        lines_numpy = []

        for c_list in lines:#risuto
            c_list=c_list[0:N]
            lines_cut.append(c_list)
            print(c_list)
        lines_merge = list(set(lines_cut))
        print(lines_merge)
        lines_merge.sort()

        for c_list in lines_merge:
            c_list=list(c_list)
            c_list=[int(s) for s in c_list]   
            num_list=np.array(c_list)
            lines_numpy.append(num_list)
    else:
        sys.exit(1)

    Candidates = []
    origin =Candidate(idnum=0,indx=np.zeros(N, dtype = int),prev_move1=np.zeros(N, dtype = int),value=0.,prev_move2=np.zeros(N, dtype = int),depth=0,maxindx=np.zeros(N, dtype = int),maxvalue=0.)
    Candidates.append(origin)

    for i in range(1,depth+1):#kainokaisuu
        print ('start')
        if (i<linearmode_start):
            comb_search(func,Candidates,i,lines_numpy,N,upperx,lowerx,Cands,pattern)
        else:
            neib_serach(func,Candidates,i,N,upperx,lowerx)


    result = []
    for cand in Candidates:
        max_value.append(cand.value)
        max_position.append(np.vectorize(convertLists)(cand.indx,cand.depth,lowerx,upperx))
        result.append([cand.value,cand.indx,np.vectorize(convertLists)(cand.indx,cand.depth,lowerx,upperx)])
    return result
#        return cand.value,np.vectorize(convertLists)(cand.indx,cand.depth,lowerx,upperx)













