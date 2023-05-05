# Load library
from scipy import sparse
import numpy as np

# Create a vector as a row
vector_row = np.array([1, 2, 3])

# Create a vector as a column

vector_column = np.array([[1],
                          [2],
                          [3]])

# Load library

# Create a matrix
matrix = np.array([[1, 2],
                   [1, 2],
                   [1, 2]])

matrix_object = np.mat([[1, 2],
                        [1, 2],
                        [1, 2]])


# Load libraries
# from scipy import sparse

# Create a matrix
matrix = np.array([[0, 0],
                   [0, 1],
                   [3, 0]])

# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)
""" Sparse matrices only store nonzero elements and assume all other values
will be zero, leading to significant computational savings. In our
solution, we created a NumPy array with two nonzero values, then
converted it into a sparse matrix. If we view the sparse matrix we can
see that only the nonzero values are stored: """

# Create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)

# View original sparse matrix
print(matrix_sparse, '\nOriginal ^ New v')
# View larger sparse matrix
print(matrix_large_sparse)

# Create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])
print('\n mat:\n', matrix)
# View number of rows and columns
print("shape: ", matrix.shape)

# View number of elements (rows * columns)
print("size: ", matrix.size)


# View number of dimensions
print("ndim: ", matrix.ndim)

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Create function that adds 100 to something
# lambda function is a small anonymous function. A lambda function can take any number of arguments, but can only have one expression.
add_100 = lambda i: i + 100 # the equivalent functino is commented below
#def add_100(i): return i + 100


# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)

# Apply function to all elements in matrix
vectorized_add_100(matrix)

# adds 100 to all elements
matrix + 100
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print('\n', matrix)
# Return maximum element
print('np.max: ', np.max(matrix))

# Return minimum element
print('np.min: ', np.min(matrix))

# Find maximum element in each column
print('maxrow: ', np.max(matrix, axis=0))

# Find maximum element in each row
print('maxcol: ', np.max(matrix, axis=1))

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Return mean
print('mean: ', np.mean(matrix))

# Return variance
print('variance: ', np.var(matrix))

# Return standard deviation
print('standard deviation', np.std(matrix))

# Find the mean value in each column
print('meanofeachcol: ', np.mean(matrix, axis=0))
print('meanofeachrow: ', np.mean(matrix, axis=1))

# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

# Reshape matrix into 2x6 matrix
print(matrix)
print('reshape 2,6: ', matrix.reshape(2, 6))
print('reshape 1,-1: ', matrix.reshape(1, -1))
print('reshape 12', matrix.reshape(12))

# Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix)
# Transpose matrix
print('Transpose {m.T}: ', matrix.T)

# Flatten matrix
print('mat.flatten: ', matrix.flatten())

# Create matrix
matrix = np.array([[1, 1, 1],
                   [1, 1, 10],
                   [1, 1, 15]])

# Return matrix rank
np.linalg.matrix_rank(matrix)

# Create matrix
matrix = np.array([[1, 2, 3],
                   [2, 4, 6],
                   [3, 8, 9]])

# Return determinant of matrix
print(matrix)
print('det: ', np.linalg.det(matrix))

# Return diagonal elements
print('diag: ', matrix.diagonal())
print('diag offset 1: ', matrix.diagonal(offset=1))
print('diag offset -1: ', matrix.diagonal(offset=-1))

# Return trace
print('trace: ', matrix.trace())

# Return diagonal and sum elements
print('sumdiag: ', sum(matrix.diagonal()))

# Create matrix
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

# View eigenvalues
eigenvalues

# View eigenvectors
eigenvectors

# Create two vectors
vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

# Calculate dot product
np.dot(vector_a, vector_b)

# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])

# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])

# Add two matrices
np.add(matrix_a, matrix_b)


# Subtract two matrices
np.subtract(matrix_a, matrix_b)

# Add two matrices
matrix_a + matrix_b

# Create matrix
matrix_a = np.array([[1, 1],
                     [1, 2]])

# Create matrix
matrix_b = np.array([[1, 3],
                     [1, 2]])

# Multiply two matrices
np.dot(matrix_a, matrix_b)

# alternatively py3.5+, use @ to multiply matrices
matrix_a @ matrix_b

# Multiply two matrices element-wise
matrix_a * matrix_b

# Create matrix
matrix = np.array([[1, 4],
                   [2, 5]])

# Calculate inverse of matrix
np.linalg.inv(matrix)

# Multiply matrix and its inverse
matrix @ np.linalg.inv(matrix)

# Set seed
np.random.seed(0)

# Generate three random floats between 0.0 and 1.0
np.random.random(3)
#array([ 0.5488135 ,  0.71518937,  0.60276338])

# Generate three random integers between 0 and 10
np.random.randint(0, 11, 3)

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)

# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)

# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)