from __future__ import annotations
import random
import math
import os
from typing import List,Set, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import igraph as ig
import numpy as np




EMPTY_VALUE: int = 0  # Note this value is somewhat arbitrary, but this seems to work well
FULL_MASK: int = 511 # The value of a full bitset, that is all values (1...9) set

class Board:
    def __init__(
        self, 
        print_info: bool = False, 
        write_to_file: bool = False, 
        file_path: str = None,
    ):
        self.puzzle: List[List[int]] = [[EMPTY_VALUE]*9 for _ in range(9)]  # Stores the original puzzle
        self.state: List[List[int]] = [[EMPTY_VALUE]*9 for _ in range(9)]   # Stores the current board state

        self.file_path = file_path

        """
        Its useful to store rows columns and squares inside "bitsets" where each value is a set of 9 booleans, one for each value (1...9).
        Then when checking for whether a number can be put in a cell, it is just bitwise or operations, which are very fast.
        """

        # map 1..9 -> bits 0..8
        self.row_masks = [0]*9
        self.col_masks = [0]*9
        self.square_masks = [0]*9
        self.solved = False
        

        if file_path is not None:
            try:
                self.get_puzzle(file_path)  # Pulls in the puzzle from a file
            except FileNotFoundError:
                print('Puzzle file not found')
                return

            for row in range(9):  # Copies the puzzle into the board state
                for col in range(9):
                    self.set_cell(row, col, self.puzzle[row][col])

    def __str__(self) -> str:
        """
        :return: The format is the same as the input file
        """
        out: List[str] = []
        for line in self.state:
            out.append(','.join(str(c) if c != EMPTY_VALUE else '?' for c in line))
        return '\n'.join(out) + '\n'

    def pretty_print(self):
        """
        Prints this board in a human-readable form, with seperated squares
        """
        print("-"*25)
        for r in range(9):
            if r % 3 == 0 and r != 0:
                print("|"+'+'.join(["-"*7 for _ in range(3)])+"|")  # horizontal separator

            row_str = "| "
            for c in range(9):
                if c % 3 == 0 and c != 0:
                    row_str += "| "
                val = self.state[r][c]
                row_str += (str(val) if val != EMPTY_VALUE else ".") + " "
            print(row_str.strip()+" |")
        print("-"*25)

    def copy(self, board: Board):
        for row in range(9):
            for col in range(9):
                self.set_cell(row, col, board.get_cell(row, col))

    def get_cell(self, row: int, col: int) -> int:
        return self.state[row][col]

    def get_row(self, row: int) -> List[int]:
        return self.state[row]

    def get_col(self, col: int) -> List[int]:
        return [row[col] for row in self.state]

    def get_square(self, squareid: int) -> List[int]:
        """
        Gets one of the 9 3x3 squares on the Sudoku puzzle
        :param squareid: a value 0...8 representing the square
        :return: the values of the square in a list
        """
        r0 = (squareid // 3) * 3
        c0 = (squareid % 3) * 3
        return [self.state[r][c] for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)]

    def set_square(self, squareid: int, values: List[int]):
        r0 = (squareid // 3) * 3
        c0 = (squareid % 3) * 3
        # clear existing cells in the square from masks/state
        for row in range(r0, r0 + 3):
            for col in range(c0, c0 + 3):
                self.clear_cell(row, col)
        # set new values with mask updates
        i = 0
        for row in range(r0, r0 + 3):
            for col in range(c0, c0 + 3):
                value = values[i]
                if value != EMPTY_VALUE:
                    self.set_cell(row, col, value)
                i += 1

    def set_cell(self, row: int, col: int, value: int):
        self.clear_cell(row, col)
        self.state[row][col] = value
        if value != EMPTY_VALUE:
            bit = 1 << (value - 1)  # map 1..9 -> bits 0..8
            self.row_masks[row] |= bit
            self.col_masks[col] |= bit
            self.square_masks[(row//3)*3 + (col//3)] |= bit

    def clear_cell(self, row: int, col: int):
        original_value = self.state[row][col]
        if original_value != EMPTY_VALUE:
            bit = 1 << (original_value - 1)
            self.row_masks[row] &= ~bit
            self.col_masks[col] &= ~bit
            self.square_masks[(row//3)*3 + (col//3)] &= ~bit
        self.state[row][col] = EMPTY_VALUE

    def swap_cells(self, row1: int, col1: int, row2: int, col2: int):
        """
        Swaps two given cells
        """
        value1,value2 = self.get_cell(row1, col1), self.get_cell(row2, col2)
        self.set_cell(row2, col2, value1)
        self.set_cell(row1, col1, value2)

    def get_row_mask(self, row: int) -> int:
        return self.row_masks[row]

    def get_col_mask(self, col: int) -> int:
        return self.col_masks[col]

    def get_square_mask(self, row: int, col: int) -> int:
        return self.square_masks[(row//3)*3 + (col//3)]

    def get_cell_mask(self, row: int, col: int) -> int:
        """
        Returns the combination of the row, column, and square constrains in a single bitset (integer)
        """
        cell_mask = self.get_row_mask(row)
        cell_mask |= self.get_col_mask(col)
        cell_mask |= self.get_square_mask(row, col)
        return cell_mask

    def get_puzzle(self, filename: str):
        """
        Throws fileNotFound exception
        :param filename: Takes in a file name of a puzzle file
        Sets this board to the puzzle given in the file
        """
        with open(filename) as f:
            for y, line in enumerate(f.readlines()):
                for x, value in enumerate(line.split(',')):
                    value = value.strip()
                    if not value.isnumeric():
                        self.puzzle[y][x] = EMPTY_VALUE
                    else:
                        self.puzzle[y][x] = int(value)

    def output_puzzle_file(self) -> str:
        puzzle_name = ""
        for ch in reversed(PUZZLE_PATH):
            if ch in ('/', '\\'):
                break
            puzzle_name = ch + puzzle_name
        puzzle_name = puzzle_name.split('.')[0]  # Removes txt extension

        os.makedirs('output', exist_ok=True) # Makes sure output directory exists
        outfile_name = f'output/{GROUP_ID}_{ALGORITHM}_{PUZZLE_TYPE}_{puzzle_name}.txt'
        with open(outfile_name, 'w') as f:
            f.write(str(self))
        return outfile_name

    def check_solution(self) -> bool:
        """
        Separate checker from the solvers themselves, to ensure puzzles are correct.
        Not very optimized, but it doesn't really matter because it runs once per puzzle.
        :return: True is solved, False otherwise
        """
        # 1) No empties
        values: set[int] = set()
        for row in self.state:
            values.update(row)
        if EMPTY_VALUE in values:
            return False

        values = set()
        # 2) Rows and columns contain 9 distinct values (Sudoku assumes 1..9)
        for i in range(9):
            row = set(self.get_row(i))
            col = set(self.get_col(i))
            if len(row) != 9:
                return False
            if len(col) != 9:
                return False

            values = values.union(row)

            # 3) Each 3x3 square distinct
            r0 = (i // 3) * 3
            c0 = (i % 3) * 3
            square = [self.state[r][c] for r in range(r0, r0+3) for c in range(c0, c0+3)]
            if len(set(square)) != 9:
                return False

        # 4) Only legal values in the board (1...9)
        if len(values.intersection(set(range(1, 10)))) != 9:
            return False

        # 5) All original tiles from the puzzle are preserved
        for row in range(9):
            for col in range(9):
                if self.puzzle[row][col] != EMPTY_VALUE and self.state[row][col] != self.puzzle[row][col]:
                    return False

        return True


    def _domain_mask(self, r: int, c: int) -> int:
        """
        Bitmask of allowed values for cell (r,c), bits 0..8 correspond to 1..9.
        If cell is filled, returns mask with that single value
        """
        v = self.state[r][c]
        if v != EMPTY_VALUE:
            return 1 << (v - 1)

        used = self.get_cell_mask(r, c)          # bits used in row/col/box
        allowed = (~used) & ((1 << 9) - 1)       # keep only 9 bits
        return allowed

    @staticmethod
    def _popcount9(x: int) -> int:
        return x.bit_count()

    def to_constraint_graph_igraph(
        self,
        weight_mode: str = "inv_overlap",   # "inv_overlap", "overlap", "binary"
        eps: float = 1e-6,
        include_filled_edges: bool = True,
    ) -> ig.Graph:
        """
        Returns an igraph.Graph representing THIS board state.

        Vertices: 0..80 (id = 9*r + c)
        Edges: between peer cells (same row/col/box)
        Weights: derived from candidate-domain overlap (board-specific)

        Vertex attributes added:
          - "r","c","box"
          - "value" (EMPTY_VALUE if empty)
          - "fixed" (True if filled)
          - "dom_mask" (int bitmask)
          - "dom_size" (0..9)

        Edge attributes added:
          - "weight" (float) unless weight_mode == "binary"
          - "relation" one of {"row","col","box"} (if an edge is multiple types, it stores "multi")
        """
        n = 81
        full9 = (1 << 9) - 1

        def vid(r: int, c: int) -> int:
            return 9 * r + c

        def box_id(r: int, c: int) -> int:
            return (r // 3) * 3 + (c // 3)

        # Precompute per-cell domains
        dom = [0] * n
        val = [EMPTY_VALUE] * n
        fixed = [False] * n
        for r in range(9):
            for c in range(9):
                i = vid(r, c)
                v = self.state[r][c]
                val[i] = v
                fixed[i] = (v != EMPTY_VALUE)
                dom[i] = self._domain_mask(r, c) & full9

        # Build edges (peers) + relation type
        seen = {}  # (a,b) -> relation string
        edges = []
        relations = []

        def add_edge(i: int, j: int, rel: str):
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in seen:
                if seen[key] != rel:
                    seen[key] = "multi"
                return
            seen[key] = rel
            edges.append((a, b))
            relations.append(rel)

        for r in range(9):
            for c in range(9):
                i = vid(r, c)
                # Optionally skip edges incident to filled cells
                if (not include_filled_edges) and fixed[i]:
                    continue

                # same row
                for cc in range(9):
                    if cc == c:
                        continue
                    j = vid(r, cc)
                    if (not include_filled_edges) and fixed[j]:
                        continue
                    add_edge(i, j, "row")

                # same col
                for rr in range(9):
                    if rr == r:
                        continue
                    j = vid(rr, c)
                    if (not include_filled_edges) and fixed[j]:
                        continue
                    add_edge(i, j, "col")

                # same box
                br = (r // 3) * 3
                bc = (c // 3) * 3
                for rr in range(br, br + 3):
                    for cc in range(bc, bc + 3):
                        if rr == r and cc == c:
                            continue
                        j = vid(rr, cc)
                        if (not include_filled_edges) and fixed[j]:
                            continue
                        add_edge(i, j, "box")

        G = ig.Graph(n=n, edges=edges, directed=False)

        # Vertex attributes
        G.vs["r"] = [i // 9 for i in range(n)]
        G.vs["c"] = [i % 9 for i in range(n)]
        G.vs["box"] = [box_id(i // 9, i % 9) for i in range(n)]
        G.vs["value"] = val
        G.vs["fixed"] = fixed
        G.vs["dom_mask"] = dom
        G.vs["dom_size"] = [self._popcount9(m) for m in dom]

        # Edge attributes
        G.es["relation"] = [seen[(min(a,b), max(a,b))] for (a,b) in edges]

        if weight_mode != "binary":
            weights = []
            for (a, b) in edges:
                inter = dom[a] & dom[b]
                k = self._popcount9(inter)  # overlap size 0..9

                if weight_mode == "overlap":
                    w = k / 9.0
                elif weight_mode == "inv_overlap":
                    # smaller overlap => larger weight (tighter constraint coupling)
                    w = 1.0 / (k + eps)
                else:
                    raise ValueError("weight_mode must be one of: 'inv_overlap', 'overlap', 'binary'")

                weights.append(float(w))
            G.es["weight"] = weights

        return G

class BacktrackingSolver:
    def __init__(self, board: Board):
        self.decision_count: int = 0
        self.board: Board = board
        self.empty_squares: List[Tuple[int, int]] = [] # A list of all squares that need to be filled
        self.counted_numbers = {} # Counted values used in the puzzle so far, up to 9 of each (1...9) ignores empty squares
        self.get_all_empty()
        self.count_board_values()

    def get_all_empty(self):
        """
        Stores all empty squares into self.empty_squares
        """
        for row in range(9):
            for col in range(9):
                if self.board.state[row][col] == EMPTY_VALUE:
                    self.empty_squares.append((row, col))

    def count_board_values(self):
        """
        Count used values and store them in self.counted_numbers
        """
        self.counted_numbers = {i:0 for i in range(1, 10)}
        for row in range(9):
            for col in range(9):
                value = self.board.get_cell(row, col)
                if value != EMPTY_VALUE:
                    self.counted_numbers[value] += 1

    def sorted_values_by_use(self):
        return sorted(self.counted_numbers.items(),key=lambda x: x[1])

    def get_mrv(self) -> Tuple[int, int]:
        """
        mrv: most restricted value
        Look into the empty squares list, and finds the one with the most restrictions
        :return: Returns the mrv and pops it off the empty squares list
        """
        cell_index = -1
        best_restriction_count = -1

        for i, cell in enumerate(self.empty_squares):
            row, col = cell
            restriction_count = self.board.get_cell_mask(row, col).bit_count()

            if restriction_count > best_restriction_count:
                best_restriction_count = restriction_count
                cell_index = i
        return self.empty_squares.pop(cell_index)

    def solve(self) -> bool:
        """
        Solves the puzzle using backtracking. With a couple heuristics:
        1. mrv, start with whatever cell is the most restricted i.e. can only be a few values.
                This ensures that the DFS tree has as few starting branches as possible.
        2. least used value, start with the values that are least used in the puzzle so far:
                This limits the rest of the puzzle as minimally as possible.
        :return: Return the success of the solving algorithm
        """
        if len(self.empty_squares) == 0:
            return True

        row, col = self.get_mrv()
        for digit,_ in self.sorted_values_by_use():
            self.counted_numbers[digit] += 1
            self.decision_count += 1  # count each attempted assignment
            if self.is_digit_possible(row, col, digit):
                self.board.set_cell(row, col, digit)
                if self.solve():
                    return True
                self.board.clear_cell(row, col)
            self.counted_numbers[digit] -= 1
        self.empty_squares.append((row, col))
        return False

    def is_digit_possible(self, row: int, col: int, digit: int) -> bool:
        """
        Uses bitwise logic to quickly determine if a digit can go into a square.
        """
        bit = 1 << (digit - 1)
        return bool(bit & (~self.board.get_cell_mask(row, col)))

class ForwardCheckingSolver:
    def __init__(self, board: Board):
        self.decision_count: int = 0
        self.board: Board = board
        self.cell_domains: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)] # Bitsets for each cell in the puzzle, map 1..9 -> bits 0..8
        self.init_domains()
        self.empty_squares: Set[Tuple[int, int]] = set() # All squares that are empty in the puzzle before solving
        self.init_empty_squares()
        self.counted_numbers = {i:0 for i in range(1,10)} # Counted values used in the puzzle so far, up to 9 of each (1...9) ignores empty squares
        self.init_counted_numbers()

    def init_empty_squares(self):
        for row in range(9):
            for col in range(9):
                if self.board.get_cell(row,col) == EMPTY_VALUE:
                    self.empty_squares.add((row, col))

    def init_domains(self):
        for row in range(9):
            for col in range(9):
                self.cell_domains[row][col] = FULL_MASK & (~self.board.get_cell_mask(row, col))

    def init_counted_numbers(self):
        for row in range(9):
            for col in range(9):
                value = self.board.get_cell(row, col)
                if value != EMPTY_VALUE:
                    self.counted_numbers[value] += 1

    def sorted_values_by_use(self) -> List[Tuple[int, int]]:
        return sorted(self.counted_numbers.items(),key=lambda x: x[1])

    def update_domains(self, start: Tuple[int, int]) -> bool:
        row, col = start

        # Top left of the square containing the start value
        r0 = (row // 3) * 3
        c0 = (col // 3) * 3

        neighbors = set()

        # Row and column neighbors
        for i in range(9):
            neighbors.add((row, i))
            neighbors.add((i, col))

        # Box neighbors
        for dr in range(3):
            for dc in range(3):
                neighbors.add((r0 + dr, c0 + dc))

        valid_board = True
        # Update domains for each cell
        for r, c in neighbors:
            value = FULL_MASK & (~self.board.get_cell_mask(r, c))
            if value == 0 and (r,c) in self.empty_squares:
                valid_board = False
            self.cell_domains[r][c] = value

        return valid_board

    def get_mrv(self) -> Tuple[int, int]:
        """
        mrv: most restricted value
        Looks into the cell domains and finds the one with the most restrictions
        :return: Returns the mrv and pops it off the empty squares list
        """
        best_value = 99
        best_cell = None
        for cell in self.empty_squares:
            domain_size = self.cell_domains[cell[0]][cell[1]].bit_count()
            if domain_size < best_value:
                best_value = domain_size
                best_cell = cell
        self.empty_squares.remove(best_cell)
        return best_cell

    def solve(self) -> bool:
        if len(self.empty_squares) == 0:
            return True
        row,col = self.get_mrv()
        for value,_ in self.sorted_values_by_use():
            if not bool((1<<(value-1))&self.cell_domains[row][col]):
                continue
            self.board.set_cell(row, col, value)
            if self.update_domains((row,col)):
                self.decision_count += 1
                self.counted_numbers[value] += 1
                if self.solve():
                    return True
            self.counted_numbers[value] -= 1
            self.board.clear_cell(row, col)
            self.update_domains((row,col))
        self.empty_squares.add((row, col))
        return False

class AC3Solver:
    def __init__(self, board: Board):
        self.decision_count: int = 0
        self.board: Board = board
        self.empty_squares: Set[Tuple[int, int]] = set() # All squares that are empty in the puzzle before solving
        self.init_empty_squares()
        self.cell_domains: List[List[int]] = [[0 for _ in range(9)] for _ in range(9)] # Bitsets for each cell in the puzzle, map 1..9 -> bits 0..8
        self.init_domains()
        self.counted_numbers = {i:0 for i in range(1,10)} # Counted values used in the puzzle so far, up to 9 of each (1...9) ignores empty squares
        self.init_counted_numbers()
        self.neighbors: List[List[Set[Tuple[int, int]]]] = [list() for _ in range(9)] # A lookup table for all neighbors of all cells
        self.init_neighbors()



    def init_empty_squares(self):
        for row in range(9):
            for col in range(9):
                if self.board.get_cell(row,col) == EMPTY_VALUE:
                    self.empty_squares.add((row, col))

    def init_domains(self):
        for row in range(9):
            for col in range(9):
                value = self.board.get_cell(row,col)
                if value != EMPTY_VALUE:
                    self.cell_domains[row][col] = 1<<(value-1)
                else:
                    self.cell_domains[row][col] = FULL_MASK & (~self.board.get_cell_mask(row, col))

    def init_counted_numbers(self):
        for row in range(9):
            for col in range(9):
                value = self.board.get_cell(row, col)
                if value != EMPTY_VALUE:
                    self.counted_numbers[value] += 1

    @staticmethod
    def get_neighbors_onthefly(start: Tuple[int, int], ) -> Set[Tuple[int, int]]:
        """
        Finds all neighbors of the current cell, not including the current cell
        """
        row, col = start

        # Top left of the square containing the start value
        r0 = (row // 3) * 3
        c0 = (col // 3) * 3

        neighbors = set()

        # Row and column neighbors
        for i in range(9):
            neighbors.add((row, i))
            neighbors.add((i, col))

        # Square neighbors
        for dr in range(3):
            for dc in range(3):
                neighbors.add((r0 + dr, c0 + dc))

        neighbors.discard(start)
        return neighbors

    def init_neighbors(self):
        for row in range(9):
            for col in range(9):
                self.neighbors[row].append(AC3Solver.get_neighbors_onthefly((row,col)))


    def get_neighbors(self,start: Tuple[int,int]) -> Set[Tuple[int, int]]:
        return self.neighbors[start[0]][start[1]]


    def propagate_constraints(self) -> bool:
        """
        :return: success or failure in a boolean
        """
        contradiction = False
        propagation_queue = [(r,c) for r in range(9) for c in range(9)] # Values to be checked for their domain size
        while propagation_queue and not contradiction:
            row, col = propagation_queue.pop(0)
            domain = self.cell_domains[row][col]
            domain_size = domain.bit_count()
            if domain_size == 0: # Contradiction, something went wrong
                return False
            if domain_size > 1:
                continue

            # Domain size is known to be one at this point
            for neighbor_row,neighbor_col in self.get_neighbors((row,col)):
                neighbor_domain = self.cell_domains[neighbor_row][neighbor_col]
                self.cell_domains[neighbor_row][neighbor_col] &= ~domain # Remove this value from each neighbor
                if neighbor_domain != self.cell_domains[neighbor_row][neighbor_col]: # if something changes
                    if self.cell_domains[neighbor_row][neighbor_col].bit_count()==1:
                        propagation_queue.append((neighbor_row, neighbor_col))
                    if self.cell_domains[neighbor_row][neighbor_col].bit_count()==0:
                        return False

        return True

    def get_mrv(self) -> Tuple[int, int]:
        """
        mrv: most restricted value
        Looks into the cell domains and finds the one with the most restrictions
        :return: Returns the mrv and pops it off the empty squares list
        """
        best_value = 99
        best_cell = None
        for cell in self.empty_squares:
            domain_size = self.cell_domains[cell[0]][cell[1]].bit_count()
            if domain_size < best_value:
                best_value = domain_size
                best_cell = cell
        self.empty_squares.remove(best_cell)
        return best_cell

    def sorted_values_by_use(self):
        return sorted(self.counted_numbers.items(),key=lambda x: x[1])

    def solve(self) -> bool:
        if len(self.empty_squares) == 0:
            return True
        if not self.propagate_constraints():
            return False

        row,col = self.get_mrv()
        for value,_ in self.sorted_values_by_use():
            if not bool((1<<(value-1))&self.cell_domains[row][col]):
                continue

            board_domains = [row[:] for row in self.cell_domains] # Copies the board domains in case of a backtrack
            self.cell_domains[row][col] = 1 << (value-1)
            self.board.set_cell(row, col, value)

            self.decision_count += 1
            self.counted_numbers[value] += 1
            if self.solve(): # Backtrack
                return True

            self.cell_domains = [row[:] for row in board_domains] # Restores the values to what they were before the wrong guess
            self.counted_numbers[value] -= 1
            self.board.clear_cell(row, col)
        self.empty_squares.add((row, col))
        return False

def test_suite():
    algs = [BacktrackingSolver, ForwardCheckingSolver, AC3Solver]
    puzzle_dir = "custompuzzles"
    
    puzzles = [str(p.resolve()) for p in Path(puzzle_dir).rglob("*") if p.is_file()]
    boards = [Board(file_path=puzzle) for puzzle in puzzles]

    smallest_eigenvalues = ("", float('inf'))
    largest_eigenvalues = ("", float('-inf'))

    
    for board in boards[:1]:
        G = board.to_constraint_graph_igraph(weight_mode="inv_overlap", include_filled_edges=True)


        A = np.array(G.get_adjacency(attribute="weight").data, dtype=float)  # weighted adjacency
        D = np.diag(A.sum(axis=1))
        L = D - A  # weighted Laplacian

        print(len(A))

        evals = np.linalg.eigvalsh(L)
        
    


test_suite()