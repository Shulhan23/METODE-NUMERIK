# import metode dari numpy dan scipy 
import numpy as np
import scipy.linalg

# rumus metode matriks balikan
def solve_using_inverse(matrix_A, vector_b):
    A_inv = np.linalg.inv(matrix_A)
    x = np.dot(A_inv, vector_b)
    return x

# Rumus metode dekomposisi LU Gauss
def solve_using_lu_gauss(matrix_A, vector_b):
    P, L, U = scipy.linalg.lu(matrix_A)
    y = np.linalg.solve(L, np.dot(P, vector_b))
    x = np.linalg.solve(U, y)
    return x

# Rumus metode dekomposisi Crout
def solve_using_crout(matrix_A, vector_b):
    LU, piv = scipy.linalg.lu_factor(matrix_A)
    L = np.tril(LU, k=-1) + np.eye(len(matrix_A))
    U = np.triu(LU)
    y = np.linalg.solve(L, vector_b)
    x = np.linalg.solve(U, y)
    return x

def penyelesaian():
    
    matrix_A = np.array([[1, 1, 2 ], [2, 4, -3], [3, 6, -5]])
    vector_b = np.array([9, 1, 0])
    
    # Testing metode matriks balikan
    print("")
    print("METODE MATRIKS BALIKAN",)
    print("")
    x_inverse = solve_using_inverse(matrix_A, vector_b)
    print("Solusi menggunakan metode matriks balikan:", x_inverse)

    # Testing metode dekomposisi LU Gauss
    print("")
    print("METODE DEKOMPOSISI LU GAUSS",)
    print("")
    x_lu_gauss = solve_using_lu_gauss(matrix_A, vector_b)
    print("Solusi menggunakan metode dekomposisi LU Gauss:", x_lu_gauss)
    
    # Testing metode dekomposisi Crout
    print("")
    print("METODE DEKOMPOSISI CROUT",)
    print("")
    x_crout = solve_using_crout(matrix_A, vector_b)
    print("Solusi menggunakan metode dekomposisi Crout:", x_crout)

if __name__ == "__main__":
    penyelesaian()
