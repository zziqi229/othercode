import numpy as np
from PIL import Image
from math import sqrt
import EntropyHub as EH
import copy

n = 224
dir = np.array([-1, -1, -1, 0, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 0, -1]).reshape([8, 2])

np.random.seed(233)
A = np.random.randint(0, high=2, size=[n, n], dtype=np.uint8)

ff = np.zeros([8, n, n], np.int32)
gg = np.ones([8, n, n, 2], np.int32) * -1

ts = 30

color = np.zeros([n, n], np.int32)


def transimg(A):  # 0 白    1黑    2蓝
    R = np.zeros_like(A)
    G = np.zeros_like(A)
    B = np.zeros_like(A)
    R[A == 0] = 1
    G[A == 0] = 1
    B[A == 0] = 1

    R[A == 1] = 0
    G[A == 1] = 0
    B[A == 1] = 0

    R[A == 2] = 0
    G[A == 2] = 0
    B[A == 2] = 1
    img = Image.fromarray(np.stack([R, G, B], -1) * 255)
    return img


def inbound(x, y):
    if x >= 0 and x < n and y >= 0 and y < n: return True
    return False


def dfs1(a, b, dir_id=[5, 6]):
    if f[a, b] != 0:
        return f[a, b]
    f[a, b] = 1
    for i in dir_id:
        x, y = a + dir[i, 0], b + dir[i, 1]
        if not inbound(x, y): continue
        if A[x, y] != 1: continue
        t = dfs1(x, y, dir_id)
        if t + 1 > f[a, b]:
            f[a, b] = t + 1
            g[a, b] = [x, y]
    return f[a, b]


def get_points(a, b):
    res = [(a, b)]
    if g[a, b, 0] != -1:
        res += get_points(g[a, b, 0], g[a, b, 1])
    return res


def draw():
    rx, ry = 100, 100
    r = 50
    for i in range(rx - r, rx + r + 1):
        for j in range(ry - r, ry + r + 1):
            if abs((i - rx) ** 2 + (j - ry) ** 2 - r * r) <= 50:
                A[i, j] = 1
    for i in range(100):
        A[i, 200] = 1
    for i in range(50):
        x = i
        y = int(sqrt(i))
        A[x + 100, 20 + y] = 1


def dfs2(a, b, id):
    color[a, b] = id
    for d in dir:
        x, y = a + d[0], b + d[1]
        if not inbound(x, y): continue
        if A[x, y] == 1 and color[x, y] == 0:
            dfs2(x, y, id)


def loss(lines):
    lines = np.array(lines)
    C = np.zeros_like(lines)
    for i in range(1, len(C)):
        C[i, 0] = lines[i, 0] - lines[i - 1, 0]
        C[i, 1] = lines[i, 1] - lines[i - 1, 1]
    ent = 0
    Ap, Phi = EH.ApEn(C[1:, 0], m=2, r=0.15)
    ent += Ap[-1]
    Ap, Phi = EH.ApEn(C[1:, 1], m=2, r=0.15)
    ent += Ap[-1]
    return ent


def dfs3(a, b, cur, dir_id):
    if len(cur) >= ts:
        if True or loss(cur) < 1:
            for x, y in cur:
                A[x, y] = 2
        else:
            return

    vis[a, b] = True
    for k in dir_id:
        d = dir[k]
        x, y = a + d[0], b + d[1]
        if not inbound(x, y): continue
        if A[x, y] == 0 or vis[x, y]: continue
        nex = copy.copy(cur)
        nex.append([x, y])
        dfs3(x, y, nex, dir_id)
    pass


def getid(a, b):
    return a * n + b


def find(x):
    if x != uset[x]:
        uset[x] = find(uset[x])
    return uset[x]


if __name__ == '__main__':
    draw()

    for d in range(8):
        f = ff[d]
        g = gg[d]
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1:
                    dfs1(i, j, [d, (d + 1) % 8])

    for d in range(8):
        f = ff[d]
        g = gg[d]
        uset = [i for i in range(n * n)]
        mx = [0] * (n * n)
        for a in range(n):
            for b in range(n):
                if g[a, b, 0] == -1: continue
                u = getid(a, b)
                v = getid(g[a, b, 0], g[a, b, 1])
                uset[find(u)] = find(v)
        for a in range(n):
            for b in range(n):
                x = getid(a, b)
                mx[find(x)] = max(mx[find(x)], f[a, b])
        for a in range(n):
            for b in range(n):
                if f[a, b] >= ts and f[a, b] == mx[find(getid(a, b))]:
                    lines = get_points(a, b)
                    lines = np.array(lines)
                    C = np.zeros_like(lines)
                    for i in range(1, len(C)):
                        C[i, 0] = lines[i, 0] - lines[i - 1, 0]
                        C[i, 1] = lines[i, 1] - lines[i - 1, 1]
                    ent = 0
                    Ap, Phi = EH.ApEn(C[1:, 0], m=3, r=0.15)
                    ent += Ap[-1]
                    Ap, Phi = EH.ApEn(C[1:, 1], m=3, r=0.15)
                    ent += Ap[-1]
                    if True or ent < 0.6:
                        for i, j in lines:
                            A[i, j] = 2

    # A[A != 2] = 0
    # A[A == 2] = 1

    img = transimg(A)
    img.show()

    # id = 0
    # for i in range(n):
    #     for j in range(n):
    #         if A[i, j] == 1 and color[i, j] == 0:
    #             id += 1
    #             dfs2(i, j, id)
    #
    # for i in range(n):
    #     for j in range(n):
    #         if A[i, j] == 0: continue;
    #         vis = np.zeros([n, n], np.bool)
    #         dfs3(i, j, [], dir_id=[7, 0, 1])
    # img = transimg(A)
    # img.show()
    #
    # A[A != 2] = 0
    # A[A == 2] = 1
    # for i in range(n):
    #     for j in range(n):
    #         if A[i, j] == 0: continue;
    #         vis = np.zeros([n, n], np.bool)
    #         dfs3(i, j, [], dir_id=[3, 4, 5])
    # img = transimg(A)
    # img.show()
