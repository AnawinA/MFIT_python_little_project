import numpy as np
import mfit_class as mf


# print(mf.triangle_area(
#     (1, 0),
#     (2, 2),
#     (4, 3),
#     True
# ))

# print(mf.two_points_distance(
#     (2, 4),
#     (-1, 3),
#     help=True
# ))

print(mf.tetrahedron_volume(
    (0, 4, 1),
    (4, 0, 0),
    (3, 5, 2),
    (2, 2, 5),
    help=True
))