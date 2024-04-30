import argparse
from mpi_argparse import parse_args
import mpi_logger as log
from mpi4py import MPI
from mpi_cell_automata import MPI_CellAutomata

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Cyclic cellular automata",
        epilog="Happy animating :)",
    )
    parser.add_argument("-w", "--width", type=int, help="width in pixels", default=800)
    parser.add_argument("-l", "--height", type=int, help="height in pixels", default=200)
    parser.add_argument("-c", "--cmap", type=str, help="Colormap", default="Pastel1")
    parser.add_argument("-r", "--rule", type=str, help="Rule", default="cyclic")
    parser.add_argument("-i", "--iterations", type=int, help="Number of iterations", default=400)
    parser.add_argument("-o", "--outfile", type=str, help="Name of output file", default="out.gif")
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable verbose debug printing"
    )
    args = parse_args(parser)
    log.setDebug(args.debug)
    cellAutomata = MPI_CellAutomata(args.outfile, args.width, args.height, args.cmap, args.rule, args.iterations)

    cellAutomata.iterate()

    cellAutomata.saveAnimation()


if __name__ == "__main__":
    main()