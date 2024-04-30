from math import log2
from time import sleep
import mpi4py.MPI as MPI
import numpy as np
import argparse

###
#################### UTILS #################################
###


def log(*args, **kwargs):
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Shear Sort",
        description="Snake like sort in distributed fashion",
        epilog="Happy sorting :)",
    )
    parser.add_argument("-r", "--rows", type=int, help="Number of rows", required=True)
    parser.add_argument(
        "-c", "--cols", type=int, help="Number of columns", required=True
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable verbose debug printing"
    )
    args = None
    try:
        if MPI.COMM_WORLD.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = MPI.COMM_WORLD.bcast(args, root=0)

    if args is None:
        exit(0)

    return args


##################################################################


class MPI_Shear_Sort:
    def __init__(self, args: argparse.Namespace):
        self._DEBUG = args.debug
        self._N_ROWS = args.rows
        self._N_COLS = args.cols

        # Duplicate Communicator to not missmatch
        self._COMM = MPI.COMM_WORLD.Dup()
        self._N_P = self._COMM.Get_size()
        self._RANK = self._COMM.Get_rank()

        # define widths, heights, and displacements
        self._widths = np.array(
            [
                self._N_COLS // self._N_P + 1
                if self._N_COLS % self._N_P > i
                else self._N_COLS // self._N_P
                for i in range(self._N_P)
            ],
            dtype=self._NP_TYPE,
        )

        self._heights = np.array(
            [
                self._N_ROWS // self._N_P + 1
                if self._N_ROWS % self._N_P > i
                else self._N_ROWS // self._N_P
                for i in range(self._N_P)
            ],
            dtype=self._NP_TYPE,
        )

        self._disp_row = np.array(
            [sum(self._widths[:i]) * self._widths.itemsize for i in range(self._N_P)],
            dtype=self._NP_TYPE,
        )
        self._disp_col = np.array(
            [sum(self._heights[:i]) * self._heights.itemsize for i in range(self._N_P)],
            dtype=self._NP_TYPE,
        )

        # Create np arrays for rows and cols
        self._rows = np.empty(
            shape=[self._heights[self._RANK], self._N_COLS], dtype=self._NP_TYPE
        )
        self._cols = np.empty(
            shape=[self._widths[self._RANK], self._N_ROWS], dtype=self._NP_TYPE
        )

        # Define new MPI types
        self._t_LL = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0], self._N_COLS
        ).Create_resized(0, self._rows.itemsize)
        self._t_LS = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0], self._N_COLS
        ).Create_resized(0, self._rows.itemsize)
        self._t_SL = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0] - 1, self._N_COLS
        ).Create_resized(0, self._rows.itemsize)
        self._t_SS = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0] - 1, self._N_COLS
        ).Create_resized(0, self._rows.itemsize)

        self._t_col_L = self._MPI_TYPE.Create_vector(
            self._widths[0], 1, self._N_ROWS
        ).Create_resized(0, self._rows.itemsize)
        self._t_col_S = self._MPI_TYPE.Create_vector(
            self._widths[0] - 1, 1, self._N_ROWS
        ).Create_resized(0, self._rows.itemsize)

        self._t_LL.Commit()
        self._t_LS.Commit()
        self._t_SL.Commit()
        self._t_SS.Commit()
        self._t_col_L.Commit()  # send/recv type from col->row / row->col
        self._t_col_S.Commit()

        # Create lists of types to send/receive to/from other PEs
        if self._N_ROWS % self._N_P == 0 or self._RANK < self._N_ROWS % self._N_P:
            # Thick height
            self._blockTypeList = np.array(
                [
                    self._t_LL
                    if (self._N_COLS % self._N_P == 0 or i < self._N_COLS % self._N_P)
                    else self._t_SL
                    for i in range(self._N_P)
                ]
            )
        else:
            # Slim height
            self._blockTypeList = np.array(
                [
                    self._t_LS
                    if (self._N_COLS % self._N_P == 0 or i < self._N_COLS % self._N_P)
                    else self._t_SS
                    for i in range(self._N_P)
                ]
            )
        self._blockCount = np.array([1 for i in range(self._N_P)])

        # Column type to send/recieve only depends on self._RANK and not who to send/receive from
        self._colTypeList = np.array(
            [
                self._t_col_L
                if (
                    self._N_COLS % self._N_P == 0
                    or self._RANK < self._N_COLS % self._N_P
                )
                else self._t_col_S
                for i in range(self._N_P)
            ]
        )

        ## Create initial matrix
        if self._RANK == 0:
            self._large_array = np.random.randint(
                low=0,
                high=self._N_COLS * self._N_ROWS,
                size=(self._N_ROWS, self._N_COLS),
                dtype=self._NP_TYPE,
            )
            self.debug(self._large_array)

    def sort(self):
        iterations = int(log2(self._N_ROWS) + 0.5)

        # OBS Barrier ONLY for more accurate timer.
        self._COMM.Barrier()
        time_start : float = MPI.Wtime()

        count = [self._N_COLS * h for h in self._heights]
        disp = [sum(count[:i]) for i in range(self._N_P)]
        self._COMM.Scatterv(
            [self._large_array, count, disp, self._MPI_TYPE], self._rows
        )

        startIsEven = bool(disp[self._RANK] % 2 == 0)
        for i in range(iterations):
            # Sort even rows acending
            # Odd rows decending
            if startIsEven:
                self._rows[::2].sort()
                self._rows[1::2, ::-1].sort()
            else:
                self._rows[::2, ::-1].sort()
                self._rows[1::2].sort()

            # Send data from rows to columns across all PEs
            self._COMM.Alltoallw(
                (self._rows, self._blockCount, self._disp_row, self._blockTypeList),
                (self._cols, self._heights, self._disp_col, self._colTypeList),
            )

            # Sort columns
            self._cols.sort()

            # Send data from columns to rows across all PEs
            self._COMM.Alltoallw(
                [self._cols, self._heights, self._disp_col, self._colTypeList],
                [self._rows, self._blockCount, self._disp_row, self._blockTypeList],
            )

        # Sort even/odd rows one last time
        self._rows[::2].sort()
        self._rows[1::2, ::-1].sort()

        self._COMM.Gatherv(self._rows, [self._large_array, count, disp, self._MPI_TYPE])

        time : float = MPI.Wtime() - time_start
        self._COMM.reduce(time, MPI.MAX)
        if self._RANK == 0:
            self.debug("Sorted array\n", self._large_array)
        log("Wall time: ", time, "sec")

    def debug(self, *args):
        if self._DEBUG:
            print("Rank: ", self._RANK, "\n", *args)

    _DEBUG: bool = False
    _large_array: np.ndarray = None
    _rows: np.ndarray = None
    _cols: np.ndarray = None
    _COMM: MPI.Comm = None
    _N_P: int = 0
    _N_ROWS: int = 0
    _N_COLS: int = 0
    _RANK: int = None

    _widths: np.ndarray = None
    _heights: np.ndarray = None
    _disp_row: np.ndarray = None
    _disp_col: np.ndarray = None
    _blockCount: np.ndarray = None

    _NP_TYPE = np.uint32
    _MPI_TYPE = MPI.UINT32_T

    # Types L - Large, S - Small, width/height
    _t_LL: MPI.Datatype = None
    _t_LS: MPI.Datatype = None
    _t_SL: MPI.Datatype = None
    _t_SS: MPI.Datatype = None
    _t_col_L: MPI.Datatype = None
    _t_col_S: MPI.Datatype = None
    _blockTypeList: np.ndarray = None
    _colTypeList: np.ndarray = None
