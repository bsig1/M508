# Project Description

## Overview

In this project, we analyze the computational difficulty of different Sudoku puzzles and attempt to find a function that can predict the relative difficulty of a given puzzle.

First, we define a measurable quantity that depends on puzzle difficulty.

Sudoku is a classic example of a **Constraint Satisfaction Problem (CSP)** where the goal is to find an assignment of values such that all constraints are satisfied. For standard 9×9 Sudoku:

- All rows contain 1..9  
- All columns contain 1..9  
- All 3×3 boxes contain 1..9  

Computationally, CSPs quickly become very difficult and are known to be **NP-complete** in the generalized case.

Using the constraint model of Sudoku, we build a graph where each edge represents a constraint between two cells. Each cell has an edge to every other cell in its row, column, and box.

Regardless of the puzzle’s starting state, the structure of relationships between cells remains consistent. What varies is the **weight** of these edges.

There is no single correct way to weight edges. One approach is to let the weight between two cells be proportional to the intersection of their available domains. The more values two cells can both take, the stronger their connection. The inverse can also be considered: fewer shared values imply a stronger coupling, since assigning one cell immediately constrains the other.

More sophisticated weighting models are also used in this project.

---

## Adjacency and Laplacian Matrix Descriptions

Given a puzzle with a starting state, we define a weighted constraint graph and convert it into matrix form.

Let **A** be the adjacency matrix of this graph:

- The graph is undirected, so **A is symmetric**.
- The Sudoku board has 81 cells, so **A is 81×81**.
- Each entry represents the strength of the relationship between two cells.
- The matrix is sparse.
- Diagonal entries are zero (no self-loops).

Now define matrix **D**, a diagonal matrix where:

D_ii = sum_j A_ij

Each diagonal element represents the total connectivity of a cell. Larger values indicate stronger overall constraint coupling.

Define the **Laplacian matrix**:

L = D - A

The Laplacian describes how strongly cells resist differing from their neighbors. It acts as a discrete diffusion operator and encodes how constraints propagate through the puzzle.

Taking the eigenvalues of **L** reveals structural properties of the puzzle. In particular:

- The smallest eigenvalue λ₁ is always 0.
- The second smallest eigenvalue λ₂, called the **Fiedler value**, measures graph connectivity.

The Fiedler value is hypothesized to correlate strongly with computational difficulty.

A highly connected puzzle should propagate constraints quickly. A loosely connected puzzle may delay constraint propagation, making it harder for simple algorithms.

---

## Experimental Approach

We collect data across many puzzles with varying levels of connectivity and compare:

- Graph spectral measures (e.g., Fiedler value)
- Solver compute time
- Decision counts

The hypothesis:

- Simpler algorithms (e.g., naïve backtracking) will struggle more on loosely connected puzzles.
- More sophisticated algorithms (e.g., arc consistency) will be less sensitive to connectivity.

After collecting sufficient data, kernel regression methods will be used to model the relationship between spectral properties and computational difficulty.

---

## Puzzle Dataset

1,000 puzzles were used in this project.

- 750 were randomly selected  
- 250 were selected as the hardest puzzles  

All puzzles were pulled from:

Kaggle Dataset:  
https://www.kaggle.com/datasets/rohanrao/sudoku

Local organization:

- 750 random puzzles → `Custom/`
- 250 hardest puzzles → `EvilTop/`

---

## Data Files

The experiment outputs are stored in:

- SQLite database:  
  [puzzle_info.db` ](puzzle_info.db)

- CSV export:  
  [sudoku_long.csv](sudoku_long.csv)  

---

## Database Schema

The SQLite database contains the following columns:

### id
Unique ID of the run as stored in the database.

### Inverse Overlap
Assigns higher weights to smaller shared domains.

### puzzle_path
Local file path to the puzzle.

### weight_mode
**Weight Model Documentation:**  
[Weight Models PDF](./weighting_schema.pdf)
Six weighting models are implemented (2 directional, 4 undirectional):

- **Binary**  
  Encodes standard Sudoku rules with weight = 1 for all neighboring cells.

- **Overlap**  
  Weight proportional to domain overlap size.

- **Expected Fraction**  
  Probability that a random assignment to one cell invalidates another.

- **Target Fraction**  
  A more aggressive version of Expected Fraction that emphasizes fragile bottlenecks.

- **Directional Information**  
  Measures information influence using entropy-based models.

### include_filled_edges
Boolean flag indicating whether filled cells influence the graph:

- `true` → All constraints included  
- `false` → Only empty-cell constraints considered  

### Algorithm
Three solver implementations were tested:

- **Backtracking**  
  Basic fill-and-check recursion.

- **Forward Checking**  
  After assignment, ensure no cell has domain size 0.

- **Arc Consistency**  
  Forward checking + recursive constraint propagation on singleton domains.

### Solved
Boolean value indicating whether the puzzle was solved.

### Solve Time
Measured runtime of the algorithm.

### Decision Count
Number of assignments attempted during solving. More robust than runtime but biased toward slower per-decision algorithms.

### Fiedler Value
Second smallest eigenvalue of the Laplacian matrix.

### Trace Laplacian
Trace of the Laplacian matrix (sum of diagonal elements).

### Created At
Timestamp of database insertion.
