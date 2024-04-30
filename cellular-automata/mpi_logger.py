from mpi4py import MPI
import sys
from time import sleep

DEBUG = False
def setDebug(debug : bool):
    global DEBUG
    DEBUG = debug

def log(comm : MPI.Comm,  *args, **kwargs):
    if comm.Get_rank() == 0:
        print(*args, **kwargs)

def logW(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)

def err(comm : MPI.Comm, *args, **kwargs):
    if comm.Get_rank() == 0:
        print(*args, file=sys.stderr, **kwargs)

def debug(comm : MPI.Comm, *args, **kwargs):
    global DEBUG
    if DEBUG:
        sleep(comm.Get_rank() * 0.1)
        print("Rank:", comm.Get_rank(), *args, **kwargs)
