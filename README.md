# Project Description

## Overview
In this project, we will be analyzing the computational difficulty of different sudoku puzzles, and trying to find
a function that can predict the relative difficulty of a given puzzle.
Firstly we need to define a measure to be placed on a puzzle that has some dependence to the difficulty.

Sudoku is a classic example of a Constraint Satisfaction Problem (CSP) where the goal of the puzzle is to find
a combination of inputs such that all constraints of the problem are satisfied. For Sudoku:
- All rows contain 1..9
- All columns contain 1..9
- All boxes contain 1..9
Computationally CSP's quickly become very difficult, and are known to be np-complete.

Using the constraint model of Sudoku build a graph, where each edge is a constraint between two cells.
Each cell has an edge to each other cell on its row, column, and box. Regardless of the puzzle's starting state,
the relationships between cells are going to remain consistent, what will vary is the weight of these edges.

There is no single way to weight these edges, but one idea is to let the weight of the edge between two cells
be proportional to the intersection of their available domains. The more numbers they both can put in their cell,
the stronger the connection is. The inverse of this is also often considered, where the fewer numbers that exist
in both cells, the stronger these cells are related, as putting something into one of these cells will immediately
affect the other one. Other more complicated weight models are sometimes used, these are just some examples. In this project
we use many different weight models.


## Adjacency and Laplacean Matrix Descriptions
Now take a puzzle with some starting state, and define a graph with weights between all of the edges using some model
of weights. Turn this graph into matrix form call it A, since the graph is not directional, the matrix will be symetric. Each
cell in the matrix represents the strength of the relationship between two cells on the board. The board is 9x9 with
81 cells, the matrix is 81x81 representing every relationship. This matrix is sparse, as many cells do not have direct
coupling with one another, rather they only effect each other through propagation of their constraints.
Note that each diagonal entry of A is 0, this graph does not have self-loops, cells do not have relationships
with themselves.

Lets define a matrix D. D is a diagonal matrix where each element of the matrix is the sum of a row of A. These
elements give a value for how constrained a given cell is. The larger the value, the higher the connectivity to its
neighbors.

Using matrix D and matrix A define matrix L where L=D-A this is called the Laplacian matrix. It defines how strongly
cells resist being different from their neighbors. It's sort of a discrete diffusion opperator. In the Sudoku case,
it encodes how "connected" a puzzle is. A connected puzzle is one where changing the value of a cell has an immediate
effect on its neighbors constraints, while a loosely connected puzzle makes these relationships less obvious. Taking
the eigen values of this matrix gives characteristics about how constraints propogate throughout the puzzle. Specifically
the second smallest eval(L2), sometimes called the Fiedler value, is a strong predictor of such behaviors. Note, that
because of the setup of this matrix, L1 is always 0, which is much less helpful. Using L2 and spectral analysis generally
we should be able to predict compute time.

This is the key to trying to measure a puzzle's difficulty, the puzzles difficulty is related to its connectedness.
Experimentation is going to look like collecting data on many different puzzles with different levels of connectedness,
and comparing that to its compute time with different algorithms. I predict that with less "future-aware" algorithms like
simple backtracking, that very loosely connected puzzles are going to take much longer than ones whith stronger connections.

Once enough data is collected, we will use kernel regression methods to combine different functions to try to directly model
this relationship between "connectedness" and "compuational difficulty" for different algorithms.

## Puzzles
There are 1,000 puzzles on which these experiements were ran, all of
them were pulled from https://www.kaggle.com/datasets/rohanrao/sudoku,
- 750 were pulled randomly from this database
- 250 were pulled as the hardest to solve from the database
These files are named Custom for the 750, and "EvilTop" for the 250

They represent a large sample of types of sudoku puzzles, with
hard ones being selected for specifically as the central idea of this
project is in regards to the difficulty of computing hard sudokus.

## DATA
The db file has all experiments ran sorted into a sqlite table; all
of the same information is also output into a csv.
puzzle_info.db
sudoku_long.csv
In the db, there are 11 columns of data:
- id: is the id of the run as it is stored in the db

- Inverse Overlap: This encodes the opposite, assigning higher weights to smaller codomains.

- puzzle_path: This is where it is stored locally on my computer

- weight_mode: There are six weighting models, 2 directional, 4 undirectional
    - Binary: This encodes the rules of sudoku, all cell neighbor relationships are encoded with weight 1.
    - Overlap: This encodes how much two cells overlap in their domain.
    - Expected Fraction: This encodes the probability random assignment of one of the cells will invalidate the other.
    - Target Fraction: A similar alg to dir_expected_frac, but more aggressive, weighting fragile bottlenecks high
    - Directional Information: A measure of information influence, based on entropy models

- include_filled_edges: Whether or not filled cells will have influence on the graph. If true, the constraints
are all filled in, if false, it only checks constraints on empty cells.

- Algorithm: There are 3 increasingly complex treatments of the sudoku problem used:
    - Backtracking: Fill in a cell, check for constraint violations, if none continue, if yes try a different number. 
Bare minimum simplicity, naive approach but still effective
    - Forward Checking: Fill in a cell, check if any cell has domain size 0, if one does, try a different number.
More complex, while still being able to quickly make decisions.
    - Arc Consistency: This is forward checking with constraint propogation, check if any cell has domain size 1, remove it from 
the domain of its neighbors recursively. Then if any cell has domain size 0, try a different number.
This is the most complex solver, it can weed out contradictions quickly, but requires more compute per decision.

- Solved: Boolean Value if puzzle solved

- Solve Time: A course metric of how long it took to run the algorithm on the puzzle

- Decision Count: How many numbers were filled in, in the course of solving the puzzle. A more robust metric,
but it does favor slower decision making, even if it isn't overall faster

- Fiedler Value: The second smallest eigen value of the Laplacian Matrix. See above.

- Trace Laplacian: The trace of the Laplacian Matrix

- Created At: When that row was added to the Database

