# from math import log2
# from time import sleep
from mpi4py import MPI
import numpy as np
from PIL import Image, ImageDraw
from mpi_logger import log, logW, err, debug
from time import sleep
import matplotlib.pyplot as plt
import numba

##################################################################

@numba.njit
def cyclicRule(cells : np.ndarray, newCells : np.ndarray, vmax):
    # This can probably be made better... the % is annoying 
    # Also remade to enable different neighborhoods
    for i,j in np.ndindex(cells.shape):
        newCells[i,j] = ((cells[i,j] + 2) % vmax) if np.any(cells[i-1:i+2,j-1:j+2] == ((cells[i,j] + 1) % vmax)) else cells[i,j]
    return newCells


class MPI_CellAutomata:
    # PE with rank 0 in comm_world will be responsible for drawing the gif. All others 
    # are put on a cartesian grid where they all get a part of canvas to apply the rule on 
    # Communication is done within the cartesian communicator, where ghostcells are distributed 
    # between neighbors. Each iteration starts with a a gather to the drawing PE. The 
    # biggest bottleneck for larger canvases and an increase in iterations is saving the gif at the end. 

    def __init__(self, 
                 outfile : str, 
                 width: int, 
                 height: int, 
                 colormap : str, 
                 rule : str, 
                 iterations : int = 400):
        self.SIZE = MPI.COMM_WORLD.Get_size()
        if self.SIZE == 1:
            err(MPI.COMM_WORLD, "Need at least -n2")
            exit(1)
        self.width = width
        self.height = height
        self._NP_TYPE = np.uint8  ## These need to be equivalent!
        self._MPI_TYPE = MPI.UINT8_T
        self._wComm = MPI.COMM_WORLD.Dup()
        self._worldRank = self._wComm.Get_rank()
        self.dims = MPI.Compute_dims(self.SIZE - 1, 2)
        tmpComm = self._wComm.Split(color=(self._worldRank != 0))

        self.outfile = outfile
        self.iterations = iterations
        self.vmax = 27
        if rule == "cyclic":
            self.rule = cyclicRule
        else:
            err("Rule not implemented")
            exit(1)

        self.colormap = plt.get_cmap(colormap, lut=self.vmax)
        
        self._heights = np.array(
            [
                height // self.dims[0] + 1
                if height % self.dims[0] > i
                else height // self.dims[0]
                for i in range(self.dims[0])
            ],
            dtype=np.int32,
        )
        self._widths = np.array(
            [
                width // self.dims[1] + 1
                if width % self.dims[1] > i
                else width // self.dims[1]
                for i in range(self.dims[1])
            ],
            dtype=np.int32,
        )
        dummy = np.ndarray(shape=[1, 1], dtype=self._NP_TYPE)

        # Vectors which are sent to drawing PE
        self.t_LL = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0], self._widths[0] + 2
        ).Create_resized(0, dummy.itemsize)
        self.t_SL = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0], self._widths[0] + 2
        ).Create_resized(0, dummy.itemsize)
        self.t_LS = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0] - 1, self._widths[0] + 1
        ).Create_resized(0, dummy.itemsize)
        self.t_SS = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0] - 1, self._widths[0] + 1
        ).Create_resized(0, dummy.itemsize)

        # Vectors received by drawing PE
        self.t_0_LL = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0], width
        ).Create_resized(0, dummy.itemsize)
        self.t_0_SL = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0], width
        ).Create_resized(0, dummy.itemsize)
        self.t_0_LS = self._MPI_TYPE.Create_vector(
            self._heights[0], self._widths[0] - 1, width
        ).Create_resized(0, dummy.itemsize)
        self.t_0_SS = self._MPI_TYPE.Create_vector(
            self._heights[0] - 1, self._widths[0] - 1, width
        ).Create_resized(0, dummy.itemsize)

        # Columns to send other PEs in cart comm
        self.t_col_LL = self._MPI_TYPE.Create_vector(
            self._heights[0] + 2, 1, self._widths[0] + 2
        ).Create_resized(0, dummy.itemsize)
        self.t_col_SL = self._MPI_TYPE.Create_vector(
            self._heights[0] + 1, 1, self._widths[0] + 2
        ).Create_resized(0, dummy.itemsize)
        self.t_col_LS = self._MPI_TYPE.Create_vector(
            self._heights[0] + 2, 1, self._widths[0] + 1
        ).Create_resized(0, dummy.itemsize)
        self.t_col_SS = self._MPI_TYPE.Create_vector(
            self._heights[0] + 1, 1, self._widths[0] + 1
        ).Create_resized(0, dummy.itemsize)

        self.t_LL.Commit()
        self.t_SL.Commit()
        self.t_LS.Commit()
        self.t_SS.Commit()
        self.t_0_LL.Commit()
        self.t_0_SL.Commit()
        self.t_0_LS.Commit()
        self.t_0_SS.Commit()
        self.t_col_LL.Commit()
        self.t_col_LS.Commit()
        self.t_col_SL.Commit()
        self.t_col_SS.Commit()

        if self._worldRank != 0:
            self._cartComm = tmpComm.Create_cart(
                dims=self.dims, periods=[True, True], reorder=False
            )
            self._cartRank = self._cartComm.Get_rank()
            self._coords = self._cartComm.Get_coords(self._cartRank)
            # Ghost cells in all directions
            self.cells = np.random.randint(
                low=0, high=self.vmax,
                size=[
                    self._heights[self._coords[0]] + 2,
                    self._widths[self._coords[1]] + 2,
                ],
                dtype=self._NP_TYPE,
            )
            self.newCells = np.empty(
                shape=[
                    self._heights[self._coords[0]] + 2,
                    self._widths[self._coords[1]] + 2,
                ],
                dtype=self._NP_TYPE,
            )
            # self.cells.fill(self._cartRank)
            # Send/Receive in cartiesian grid 
            class Neigbour:
                rank : int = -1
                sendDispl : int = 0
                receiveDispl : int = 0
                type : MPI.Datatype = self._MPI_TYPE
                count : int = 0

            # 4 Orthogonal neighbours, first send up and down, then left and right in order to get 
            #  each corner cell from diagonal neighbours
            up, down = self._cartComm.Shift(direction=0, disp=1)
            left, right = self._cartComm.Shift(direction=1, disp=1)
            
            self.upper = Neigbour()
            self.upper.rank = up
            self.upper.sendDispl = (self._widths[self._coords[1]] + 3) * dummy.itemsize
            self.upper.receiveDispl = 1 * dummy.itemsize
            self.upper.count = self._widths[self._coords[1]]
            self.upper.type = self._MPI_TYPE

            self.lower = Neigbour()
            self.lower.rank = down
            self.lower.sendDispl = (self._heights[self._coords[0]] * (self._widths[self._coords[1]] + 2) + 1) * dummy.itemsize
            self.lower.receiveDispl = ((self._heights[self._coords[0]] + 1) * (self._widths[self._coords[1]] + 2) + 1) * dummy.itemsize
            self.lower.count = self._widths[self._coords[1]]
            self.lower.type = self._MPI_TYPE

            self.left = Neigbour()
            self.left.rank = left
            self.left.sendDispl = 1 * dummy.itemsize
            self.left.receiveDispl = 0
            self.left.count = 1
            self.left.type = self._getRightColType()

            self.right = Neigbour()
            self.right.rank = right
            self.right.sendDispl = self._widths[self._coords[1]] * dummy.itemsize
            self.right.receiveDispl = (self._widths[self._coords[1]] + 1) * dummy.itemsize
            self.right.count = 1
            self.right.type = self._getRightColType()

        else:
            self._cartComm = MPI.COMM_NULL
            self._cartRank = -1
            self._coords = [-1, -1]
            self.cells = np.empty(shape=[height, width], dtype=self._NP_TYPE)
            self.frames = []

        # List of types to receive for drawing PE
        self._sendTypes: list[MPI.Datatype] = (
            []
            if self._worldRank != 0
            else [self._MPI_TYPE for i in range(self.SIZE)]
        )
        self._receiveTypes = [self._MPI_TYPE]
        for h in range(self.dims[0]):
            for w in range(self.dims[1]):
                if self._widths[w] == self._widths[0]:
                    # Wide
                    if self._heights[h] == self._heights[0]:
                        # Long
                        self._receiveTypes.append(self.t_0_LL)
                        if self._coords[0] == h and self._coords[1] == w:
                            self._sendTypes.append(self.t_LL)
                    else:
                        # Short
                        self._receiveTypes.append(self.t_0_SL)
                        if self._coords[0] == h and self._coords[1] == w:
                            self._sendTypes.append(self.t_SL)
                else:
                    # Narrow
                    if self._heights[h] == self._heights[0]:
                        # Long
                        self._receiveTypes.append(self.t_0_LS)
                        if self._coords[0] == h and self._coords[1] == w:
                            self._sendTypes.append(self.t_LS)
                    else:
                        # Short
                        self._receiveTypes.append(self.t_0_SS)
                        if self._coords[0] == h and self._coords[1] == w:
                            self._sendTypes.append(self.t_SS)
        self._sendCount = [0 for i in range(self.SIZE)]
        self._sendDispl = [0 for i in range(self.SIZE)]
        if self._worldRank != 0:
            self._sendTypes.extend([self._MPI_TYPE for i in range(self.SIZE - 1)])
            self._sendCount[0] = 1
            self._sendDispl[0] = (self._widths[self._coords[1]] + 3) * dummy.itemsize
            self._receiveTypes = [self._MPI_TYPE for i in range(self.SIZE)]

        # Displacements for drawing PE on where to receive
        self._receiveDispl = [0 for i in range(self.SIZE)]
        self._receiveDispl[0] = 0
        disp = 0
        for i in range(1, self.SIZE):
            self._receiveDispl[i] = disp
            disp = disp + self._widths[i % self.dims[1] - 1] * dummy.itemsize
            if i % self.dims[1] == 0:
                disp = disp + width * (self._heights[i // self.dims[1] - 1] - 1) * dummy.itemsize
        
        self._receiveCount = [
            0 if self._worldRank != 0 else 1 for i in range(self.SIZE)
        ]
        self._receiveCount[0] = 0  # Rank 0 does not receive from itself

    def _getRightColType(self)                                  :
        h = self._heights[self._coords[0]]
        w = self._widths[self._coords[1]]
        if h == self._heights[0]:
            if w == self._widths[0]:
                return self.t_col_LL
            else:
                return self.t_col_LS
        else:
            if w == self._widths[0]:
                return self.t_col_SL
            else:
                return self.t_col_SS
    

    def gather(self):
        # There is no GatherW.. so lets use non-blocking AllToAllW instead, but only send to 0
        # TODO, figure out why Ialltoallw doesn't work...
        self._wComm.Alltoallw(
            [self.cells, self._sendCount, self._sendDispl, self._sendTypes],
            [self.cells, self._receiveCount, self._receiveDispl, self._receiveTypes]
        )
        
    def distribute(self):
        if self._worldRank == 0:
            return
        # Distribute to neighbours. First send to upper and lower negbours, make sure you have received, then 
        # send to left and right neigbours. 
        self._cartComm.Isend([self.cells.flat[self.upper.sendDispl:], self.upper.count, self.upper.type], self.upper.rank, tag=1)
        req1 = self._cartComm.Irecv([self.cells.ravel()[self.lower.receiveDispl:], self.lower.count, self.lower.type], self.lower.rank, tag=1)
        
        self._cartComm.Isend([self.cells.flat[self.lower.sendDispl:], self.lower.count, self.lower.type], self.lower.rank, tag=2)
        req2 = self._cartComm.Irecv([self.cells.ravel()[self.upper.receiveDispl:], self.upper.count, self.upper.type], self.upper.rank, tag=2)
        
        req1.Wait()
        req2.Wait()

        # Now send to left and right
        self._cartComm.Isend([self.cells.flat[self.left.sendDispl:], self.left.count, self.left.type], self.left.rank, tag=3)
        req1 = self._cartComm.Irecv([self.cells.ravel()[self.right.receiveDispl:], self.right.count, self.right.type], self.right.rank, tag=3)

        self._cartComm.Isend([self.cells.flat[self.right.sendDispl:], self.right.count, self.right.type], self.right.rank, tag=4)
        req2 = self._cartComm.Irecv([self.cells.ravel()[self.left.receiveDispl:], self.left.count, self.left.type], self.left.rank, tag=4)
        
        req1.Wait()
        req2.Wait()

    def swap(self):
        tmp = self.cells
        self.cells = self.newCells
        self.newCells = tmp
    
    def draw(self):
        self.frames.append(Image.fromarray((self.colormap(self.cells)[:,:,:3] * 255).astype("uint8"), mode="RGB"))

    def saveAnimation(self):
        if self._worldRank == 0:
            logW("Saving gif...")
            self.frames[0].save(self.outfile, optimize=True, save_all=True, append_images=self.frames[1:], duration=10, loop=0)
            logW("Done!")

    def iterate(self):
        for i in range(self.iterations):
            self.gather()
            if self._worldRank == 0:
                if i > 50:
                    self.draw()
                if i % 10 == 0:
                    logW("Drew iteration", i)
            else:
                self.distribute()
                self.newCells = self.rule(self.cells, self.newCells, self.vmax)
                self.swap()
                if i % 10 == 0:
                    log(self._cartComm, "Rule applied")
        
    _gatherRequest = None

    iteration = 0
