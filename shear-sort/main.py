import numpy as np
import mpi_shear_sort 


def main():
    shear_sort_obj = mpi_shear_sort.MPI_Shear_Sort(mpi_shear_sort.parse_args())
    
    shear_sort_obj.sort()

if __name__ == "__main__":
    main()