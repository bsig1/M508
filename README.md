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
affect the other one. Other more complicated weight models are sometimes used, these are just some examples.

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
