import numpy as np

# MFIT week 1

def size(a: np.ndarray, help=False):
    shape =  a.shape
    if help:
        print(f"[size info] {shape[0]} is rows v & {shape[1]} is columns >")
    return f"{shape[0]} x {shape[1]}"

def add(a: np.ndarray, b: np.ndarray):
    return a + b

def mul(a: np.ndarray | int, b: np.ndarray | int, help=False):
    if isinstance(a, int) or isinstance(b, int):
        if help:
            print("[mul info] a * b: scalar x matrix")
        return a * b
    else:
        if help:
            print("[mul info] a @ b: matrix x matrix")
            print(f"> check: (_,{a.shape[1]}) == ({b.shape[0]},_)")
            print(f"> get: [{a.shape[0]} x {b.shape[1]}]")
            print("---calc---")
            for i in range(a.shape[0]):
                for j in range(b.shape[1]):
                    print(f"({i+1} x {j+1}): {str(a[i]):<7} * {str(b[:,j]):<7} => {str(a[i] * b[:,j]):<9} = {str(a[i] @ b[:,j]):>2}")
                print("---")
            print()
        return np.dot(a, b)

def dot(a: np.ndarray, b: np.ndarray, help=False):
    return mul(a, b, help)

def pow(a: np.ndarray, b: int, help=False):
    ai = a
    for i in range(b - 1):
        ai = mul(ai, a, help)
        
    return np.linalg.matrix_power(a, b)
def i(size: int):
    return np.identity(size, dtype=int)

def t(a: np.ndarray):
    return np.transpose(a)

def dir_week(week: int):
    dir_list = None
    match week:
        case 1:
            dir_list = [
                'size', 'add', 'mul', 'pow', 'i', 't',
            ]
        case _:
            raise ValueError(f"can not find this week: {week} (or maybe under developing)")
    return dir_list

def total_week() -> int:
    return 2

# MFIT week 2

def minor(matrix, i, j):
  sub_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
  sub_det = np.linalg.det(sub_matrix)
  return sub_det

def m(matrix, i, j):
  sub_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
  return sub_matrix

def cofactor(matrix, i, j):
  sub_minor = minor(matrix, i, j)
  return ((-1) ** (i+j)) * sub_minor

def all_index_for(a: np.ndarray, func: callable, help=False):
    new_matrix = np.zeros(a.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            get_value = func(a, i, j)
            if help:
                print(f"[{i},{j}] = {get_value}")
            new_matrix[i, j] = get_value
    return new_matrix

def to_minor(a: np.ndarray, help=False):
    a_minor = all_index_for(a, minor, help)
    return a_minor

def to_cofactor(a: np.ndarray, help=False):
    a_cofactor = all_index_for(a, cofactor, help)
    return a_cofactor

def det(a: np.ndarray, help=False):
    if help:
        print("[det info] det(a) or |a|")
        print("[1.] 2x2-3x3: product\n[2.] 4x4 or more: cofactor")
        if a.shape[0] != a.shape[1]:
            raise ValueError("this is not a square matrix")
        match a.shape[0]:
            case 2:
                print("> 2x2 then use product:")
                up_det = a[0][1] * a[1][0]
                print("up:", up_det)
                down_det = a[0][0] * a[1][1]
                print("down:", down_det)
                print(f"= result: ({down_det} - {up_det}) = ", down_det - up_det)
            case 3:
                print("> 3x3 then use product:")
                down = [a[0][0] * a[1][1] * a[2][2], a[0][1] * a[1][2] * a[2][0], a[0][2] * a[1][0] * a[2][1]]
                down_det = sum(down)
                print("down:", down, "=", down_det)
                up = [a[2][0] * a[1][1] * a[0][2], a[2][1] * a[1][2] * a[0][0], a[2][2] * a[1][0] * a[0][1]]
                up_det = sum(up)
                print("up:", up, "=", up_det)
                print(f"=> result: ({down_det} - {up_det}) = ", down_det - up_det)
            case 0:
                raise ValueError("this is not a matrix")
            case 1:
                return a[0][0]
            case _:
                print("> 4x4 or more then use cofactor:")
    
    return np.round(np.linalg.det(a), 3)

def adj(a: np.ndarray, help=False):
    get_cofactor = to_cofactor(a, help)
    return get_cofactor.T

def inv(a: np.ndarray, help=False):
    adjoint = to_cofactor(a).T
    get_det = det(a)
    if help:
        print("det: ", get_det)
        print("adjoint:\n", adjoint)
        print("inv 1/|a| * adj(a)")
    return np.round((1 / get_det) * adjoint, 3)

def triangle_area(xy1: tuple | list, xy2: tuple | list, xy3: tuple | list, help=False):
    vertices = np.array([
        [xy1[0], xy1[1], 1],
        [xy2[0], xy2[1], 1],
        [xy3[0], xy3[1], 1]
    ])
    det_ver = det(vertices)
    if help:
        print("vertices matrix:\n", vertices)
        print("det: ", det_ver)
        print("triangle area = abs(1/2 * det([x, y, 1]))")
    return np.abs(0.5 * det_ver)

def two_points_distance(xy1: tuple | list, xy2: tuple | list, help=False):
    vertices = np.array([
        [1, 1, 1],
        [xy1[0], xy1[1], 1],
        [xy2[0], xy2[1], 1]
    ])
    x = m(vertices, 0, 0)
    y = m(vertices, 0, 1)
    z = m(vertices, 0, 2)
    det_x = det(x)
    det_y = det(y)
    det_z = det(z)    
    if help:
        print("vertices matrix:\n", np.array([
            ["x", "y", 1],
            [xy1[0], xy1[1], 1],
            [xy2[0], xy2[1], 1]
        ]))
        print("M(x):\n", x)
        print("M(y):\n", y)
        print("M(z):\n", z)
        print(f"dets: {det_x}x - {det_y}y + {det_z} = 0")
        print("distance = x - y + 1")
    return f"{det_x}x - {det_y}y + {det_z} = 0"

def tetrahedron_volume(xyz1: tuple | list, xyz2: tuple | list, xyz3: tuple | list, xyz4: tuple | list, help=False):
    vertices = np.array([
        [xyz1[0], xyz1[1], xyz1[2], 1],
        [xyz2[0], xyz2[1], xyz2[2], 1],
        [xyz3[0], xyz3[1], xyz3[2], 1],
        [xyz4[0], xyz4[1], xyz4[2], 1]
    ])
    det_ver = det(vertices)
    if help:
        print("vertices matrix:\n", vertices)
        print("det: ", det_ver)
        print("triangle area = abs(1/6 * det([x, y, z, 1]))")
    return np.abs((1/6) * det_ver)

# week 3

def split_last(a: np.ndarray):
    return np.hsplit(a, [-1])

def join_last(a1: np.ndarray, a2: np.ndarray):
    return np.hstack((a1, a2))

def el(a: np.ndarray, flat=True, help=False):
    a1, a2 = split_last(a)
    a1_inv = inv(a1, help)
    if help:
        print("a1:\n", a1)
        print("a1 inv:\n", a1_inv)
        print("a2:\n", a2)
        print("a1_inv @ a2:\n")
    a1_dot = a1_inv @ a2
    if flat:
        return a1_dot.flatten()
    return a1_dot



class GuassElimination:
    def __init__(self, a: np.ndarray, always_print=False, label=False, start_index=1):
        self.a = a
        self.always_print = always_print
        self.label = label
        self.si = start_index
    
    def gswap(self, ra: int, rb: int):
        ra -= self.si
        rb -= self.si
        self.a[[ra, rb]] = self.a[[rb, ra]]
        if self.label:
            print(f"R{ra + self.si} <-> R{rb + self.si}")
        if self.always_print:
            print(self.a)

    def gmul(self, r: int, c: int):
        r -= self.si
        self.a[[r]] = self.a[[r]] * c
        if self.label:
            print(f"({c})R{r + self.si} -> R{r + self.si}")
        if self.always_print:
            print(self.a)
    
    def gadd(self, ra: int, rb: int, c: int=1):
        ra -= self.si
        rb -= self.si
        self.a[[ra]] = self.a[[ra]] + self.a[[rb]] * c
        if self.label:
            if c == 1:
                print(f"R{ra + self.si} + R{rb + self.si} -> R{ra + self.si}")
            else:
                print(f"R{ra + self.si} + ({c})R{rb + self.si} -> R{ra + self.si}")
        if self.always_print:
            print(self.a)
    
    
    def substitute(self, print_process=False, keys=False):
        a01, a02 = split_last(self.a)

        a1 = a01.copy()
        a2 = a02.flatten()
        
        result = np.zeros(a1.shape[1], dtype=int)
        for i in range(a1.shape[1] - 1, -1, -1):
            result[i] = a2[i] - np.sum(a1[i, i+1:])
            a1[:, i] = a1[:, i] * result[i]
            if print_process:
                print(a1)
        if keys:
            return {chr(k + 120): v for k, v in enumerate(result)}
        return result

def el_cramer(a: np.ndarray, x: int, help=False):
    a1, a2 = split_last(a)
    a_det = det(a1)
    x0 = a1.copy()
    x0[:, [x-1]] = a2
    x_det = det(x0)
    if help:
        print("a1:\n", a1)
        print("a2:\n", a2)
        print("|a1|:\n", a_det)
        print(f"a(x={x}):\n", x0)
        print(f"|a(x={x})|:\n", x_det)
        print(f"|a(x={x})| / |a1|\n")
    a_cramer = x_det / a_det
    return a_cramer

def gswap(a: np.ndarray, ra: int, rb: int):
    a[[ra, rb]] = a[[rb, ra]]

def gmul(a: np.ndarray, rb: int, c: int):
    a[[rb]] = a[[rb]] * c

def gadd(a: np.ndarray, ra: int, rb: int, c: int):
    a[[ra]] = a[[ra]] + a[[rb]] * (1/2)

# week 4

def norm(v: np.ndarray, help=False):
    v_sum = 0
    v_str = "sqrt("
    for i in v:
        v_sum += i**2
        v_str += f"{i}² + "
    v_norm = np.sqrt(v_sum)
    v_str = v_str[:-3] + ")"
    if help:
        print("norm ||v||:")
        print(v_str)
    return v_norm

def normc(v: np.ndarray, c:int, help=False):
    v_sum = 0
    v_str = f"|{c}|*sqrt("
    for i in v:
        v_sum += i**2
        v_str += f"{i}² + "
    v_norm = np.sqrt(v_sum)
    v_str = v_str[:-3] + ")"
    if help:
        print("norm ||cV|| to")
        print("=> |c|*||v||:")
        print(v_str)
    return abs(c) * v_norm

def unit_vector(v: np.ndarray, help=False):
    v_norm = norm(v, help)
    if help:
        print("unit vector:")
        print(f"v / ||v||")
        print(f"{v} / {v_norm}")
    return v / v_norm

def unit_vector_check(v: np.ndarray, help=False):
    v_norm = norm(v, help)
    v_unit_vector = norm(v / v_norm)
    if help:
        print("unit vector check:")
        print("if unit vector is true\n= then norm is 1")
        print(f"|| v / ||v|| ||")
        print(f"||{v} / {v_norm}||")
        print(f"||unit v|| == 1 => {v_unit_vector == 1}")
    return v_unit_vector

def distance(v1: np.ndarray, v2: np.ndarray, help=False):
    v_distance = norm(v1 - v2, help)
    if help:
        print("distance:")
        print(f"d(u, v) = ||u - v||")
        print(f"||{v1} - {v2}||")
    return v_distance

def angle(v1: np.ndarray, v2: np.ndarray, help=False):
    v_dot = dot(v1, v2)
    v_norm = np.round(norm(v1) * norm(v2))
    v_div = v_dot / v_norm
    v_angle = np.arccos(v_div)
    if help:
        print("angle:")
        print("u · v:", v_dot)
        print("||u|| * ||v||:", v_norm)
        print("u·v / ||u||*||v||:", v_div)
        print(f"arccos(u · v / ||u|| * ||v||)")
    return v_angle

def is_orthogonal(v1: np.ndarray, v2: np.ndarray, help=False):
    v_dot = dot(v1, v2)
    if help:
        print("is orthogonal:")
        print("u · v == 0")
        print("=> u == v")
    return v_dot == 0

def cross(v1: np.ndarray, v2: np.ndarray, help=False):
    v_cross = np.cross(v1, v2)
    if help:
        print("cross product:")
        print("u x v:\n",
f"[[i, j, k]\n\
  {v1}\n\
  {v2}]")
    return v_cross

def parallelogram_area(v1: np.ndarray, v2: np.ndarray, help=False):
    v_cross = cross(v1, v2)
    v_cross_norm = norm(v_cross, help)
    if help:
        print("area of parallelogram:")
        print("||u x v|| == ||u||*||v||sin(0):")
        print(f"||{v1} x {v2}||")
        print(f"||{v_cross}||")
    return v_cross_norm

# week 5

def lin_independent_m(a: np.ndarray, t: False, add_zero=True, help=False):
    aa = a.copy()
    if add_zero:
        aa = join_last(a, np.array([0] * a.shape[0]).reshape(a.shape[0], 1))
    if t:
        aa = aa.T
    print(aa)
    a_el = el(aa)
    is_idp = all(a_el == 0)
    if help:
        print("linear independent (matrix):")
        print("a:\n", aa)
        print("el(a):\n", a_el)
        print("c1 = c2 = c3 = 0:", is_idp)
        print(a_el == 0)
    return is_idp

def lin_independent(*v: np.ndarray, help=False):
    if help:
        print(*v)
    a = np.array([*v]).T
    return lin_independent_m(a, False, True, help)

# week 6

def remove_parentheses(text: str):
    if text[0] == "(" and text[-1] == ")":
        return text[1:-1]
    return text

def image_r2(v1v2: tuple, eval_str: str, help=False):
    v1, v2 = v1v2
    eval_str = remove_parentheses(eval_str)
    calc = eval(eval_str)
    if help:
        print("image of:")
        print(f"({eval_str})")
        print(f"=> {eval_str.replace("v1", str(v1)).replace("v2", str(v2))})")
    return calc

def preimage_r2(v1v2: tuple, eval_str: str, help=False):
    try:
        from sympy import symbols, Eq
        from sympy.solvers import solve
        from sympy.parsing.sympy_parser import parse_expr
    except ModuleNotFoundError:
        print("sympy module not found!!")
        print("use 'pip install sympy' to install it")
    else:
        eval_str = remove_parentheses(eval_str)
        x_ex, y_ex = eval_str.split(',') 
        x_ex = f"{x_ex.strip()}={v1v2[0]}"
        y_ex = f"{y_ex.strip()}={v1v2[1]}"
        xy_ex = (x_ex, y_ex)
        v1, v2 = symbols('v1 v2')
        v = []
        i = 0
        while i < 2:
            a1 = parse_expr(xy_ex[i].split('=')[0])
            a2 = parse_expr(xy_ex[i].split('=')[1])
            v.append(Eq(a1,a2))
            del a1,a2
            i+=1
        solved = solve(v)
        return (solved[v1], solved[v2])


# special function

def diagonalize(a: np.ndarray, p:np.ndarray, help=False):
    p_inv: np.ndarray = inv(p, help)
    b: np.ndarray = p_inv @ a @ p
    if help:
        print("[diagonalize info] b = p^−1 @ a @ p")
        print("p^-1:\n", p_inv)
        print("a @ p:\n", a @ p)
        print("p^−1 @ a @ p")
    return b


def tproduct(a: np.ndarray, b: np.ndarray, help=False):
    if help:
        print("[tproduct info] (ab)T = bT.aT")
    return b.T @ a.T

def normdbl(v: np.ndarray, is_dot=True, help=False):
    if help:
        print("||v||² == v · v")
        if is_dot:
            print(f"{v} · {v}")
        else:
            print(f"||{v}||²")
    if not is_dot:
        return norm(v, help) ** 2 
    return v @ v

def prop_cross_negative(v1: np.ndarray, v2: np.ndarray, help=False):
    v_cross = cross(v1, v2)
    v_cross_neg = cross(v2, v1)
    if help:
        print("cross product:")
        print("u x v:\n", v_cross)
        print("v x u:\n", v_cross_neg)
        print("-(v x u):", -v_cross_neg)
        print("u x v == -(v x u):")
    return v_cross == -v_cross_neg

def prop_cross_group(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, help=False):
    vc1 = cross(v2, v3)
    vc2 = cross(v1, v2)
    if help:
        print("cross product:")
        print("(v x w):\n", vc1)
        print("(u x v):\n", vc2)
        print("u · (v x w) == (u x v) · w")
        print(f"=> {dot(v1, vc1) == dot(vc2, v3)}")
    return dot(v1, vc1)