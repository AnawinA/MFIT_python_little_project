"""main"""
import numpy as np
from mfit_ende_matrix import decode_matrix

ENCODE_MATRIX = np.array([
    [1, -2, 2],
    [-1, 1, 3],
    [1, -1, -4]
])

def main() -> None:
    """main"""
    encoded = [7, -22, 54, -1, -7, 43, 10, -19, 14, 14, -27, 22, 36, -50, -61, -5, -4, 60]
    decode_text = decode_matrix(encoded, ENCODE_MATRIX)
    print(decode_text)

if __name__ == '__main__':
    main()
