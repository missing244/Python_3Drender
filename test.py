import numpy as np
import time

class a : pass


def time_count(func) :
    time1 = time.time()
    def aaa() :
        func()
        time2 = time.time()
        print(time2 - time1)
    return aaa

A = a()

def test1() :
    for i in range(10000000) :
        if A.__class__ == a : continue

def test2() :
    for i in range(10000000) :
        if isinstance(A, a) : continue


time_count(test1)()
time_count(test2)()





#exit()
import os
for path,dir,file in os.walk( "D:\备份\Render3D" ) :
    if "__pycache__" not in path : continue
    for file1 in file : os.remove( os.path.join(path, file1) )
    os.removedirs(path)