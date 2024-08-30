"""latex_from_numpy"""
import numpy as np
import math

def latex(matrix: np.ndarray):
    """latex"""
    latex_text = "\n\\begin{bmatrix}\n"
    for i in matrix:
        print(i, end=" ")
        latex_text += " & ".join(map(str, i)) + " \\\\\n"
    latex_text = latex_text[:-3] + "\n\\end{bmatrix}"
    print(latex_text)

def main():
    """latex_from_numpy"""
    import numpy as np
    matrix = np.array([
    [math.e, math.pi],
    [2, math.sqrt(2)],
    [-7, 4]
])
    print(matrix)
    latex(matrix)

main()
