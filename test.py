import os
import time
from time import sleep
from multiprocessing import Process
from threading import Thread

def f(name):
    sleep(3)
    print('hello', name)

def my_test():

    print("1--")

    p = Process(target=f, args=('bob_process',), daemon=True)
    p.start()
    # p.join()
    
    # thread = Thread(target=f, args=('bob_thread', ))
    # thread.daemon = True
    # thread.start()

    return "what"

if __name__ == '__main__':

    print("0--")
    r = my_test()
    print(r)
    print("2--")
