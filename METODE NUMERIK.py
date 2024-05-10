# import metode dari numpy dan scipy 
import numpy as np
import scipy.linalg

# rumus metode matriks balikan
def Matriks_balikan(matrix_A, Vector):
    A_inv = np.linalg.inv(matrix_A)
    x = np.dot(A_inv, Vector)
    return x

# Rumus metode dekomposisi LU Gauss
def Dekomposis_LU_Gauss(matrix_A, vector_b):
    P, L, U = scipy.linalg.lu(matrix_A)
    y = np.linalg.solve(L, np.dot(P, vector_b))
    y = np.linalg.solve(U, y)
    return y

# Rumus metode dekomposisi Crout
def Dekomposisi_Crout(matrix_A, vector_b):
    LU, piv = scipy.linalg.lu_factor(matrix_A)
    L = np.tril(LU, k=-1) + np.eye(len(matrix_A))
    U = np.triu(LU)
    y = np.linalg.solve(L, vector_b)
    x = np.linalg.solve(U, y)
    return x

def Persamaan():
    
    matrix_A = np.array([[1, 1, 2 ], [2, 4, -3], [3, 6, -5]])
    vector_b = np.array([9, 1, 0])
    
    # Testing metode matriks balikan
    print("\n" "Shulhan Aziz")
    print("")
    print("METODE MATRIKS BALIKAN",)
    x_inverse = Matriks_balikan(matrix_A, vector_b)
    print("Solusi dari persamaan menggunakan metode matriks balikan:", x_inverse)

    # Testing metode dekomposisi LU Gauss
    print("")
    print("METODE DEKOMPOSISI LU GAUSS",)
    x_lu_gauss = Dekomposis_LU_Gauss(matrix_A, vector_b)
    print("Solusi dari persamaan menggunakan metode dekomposisi LU Gauss:", x_lu_gauss)
    
    # Testing metode dekomposisi Crout
    print("")
    print("METODE DEKOMPOSISI CROUT",)
    x_crout = Dekomposisi_Crout(matrix_A, vector_b)
    print("Solusi dari persamaan menggunakan metode dekomposisi Crout:", x_crout)

if __name__ == "__main__":
    Persamaan()
