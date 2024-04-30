import argparse
from mpi4py import MPI

def parse_args(parser : argparse.ArgumentParser):
    args = None
    try:
        if MPI.COMM_WORLD.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(1)
    
    return args
