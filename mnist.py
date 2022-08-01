import copy

import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
from PIL import Image

n = 28
dir = np.array([-1, -1, -1, 0, -1, 1, 0, 1, 1, 1, 1, 0, 1, -1, 0, -1]).reshape([8, 2])
ff, gg = 0, 0

f, g = 0, 0


def getdata(ts=50):
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),  # 转换成张量
                                       ]))
    X = train.train_data.numpy()
    y = train.train_labels.numpy()
    X[X < ts] = 0
    X[X > 0] = 1
    return X, y


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


def getbound(G):
    t = np.where(G > 0)
    return (t[0].min(), t[1].min()), (t[0].max(), t[1].max())


def inbound(x, y):
    if x >= 0 and x < n and y >= 0 and y < n: return True
    return False


def dfs(G, a, b, dir_id=[5, 6]):
    global f, g, ff, gg
    if f[a, b] != 0:
        return f[a, b]
    f[a, b] = 1
    for i in dir_id:
        x, y = a + dir[i, 0], b + dir[i, 1]
        if not inbound(x, y): continue
        if G[x, y] == 0: continue
        t = dfs(G, x, y, dir_id)
        if t + 1 > f[a, b]:
            f[a, b] = t + 1
            g[a, b] = [x, y]
    return f[a, b]


def get_points(a, b):
    res = [(a, b)]
    if g[a, b, 0] != -1:
        res += get_points(g[a, b, 0], g[a, b, 1])
    return res


def judge1(G, vis=False):
    global f, g, ff, gg
    ff = np.zeros([8, n, n], np.int32)
    gg = np.ones([8, n, n, 2], np.int32) * -1
    lu, rd = getbound(G)

    for d in range(8):
        dir_id = [(d + i) % 8 for i in range(2)]
        f = ff[d]
        g = gg[d]
        for a in range(n):
            for b in range(n):
                if G[a, b] > 0:
                    dfs(G, a, b, dir_id)

    ts = int((rd[0] - lu[0]) * 0.9)

    f = ff[5]
    g = gg[5]
    GG = copy.deepcopy(G)
    for a in range(n):
        for b in range(n):
            if f[a, b] > ts:
                lines = get_points(a, b)
                lines = np.array(lines)
                GG[lines[:, 0], lines[:, 1]] = 2
    if vis:
        transimg(GG).show()
    ok1 = f.max() > ts
    r = 4
    for a in range(n):
        for b in range(n):
            if G[a, b] == 0: continue
            flag = False
            for i in range(-r, r):
                for j in range(-r, r):
                    x, y = a + i, b + j
                    if inbound(x, y):
                        if GG[x, y] == 2:
                            flag = True
                            break
            if flag == False:
                ok1 = False

    f = ff[4]
    g = gg[4]
    GG = copy.deepcopy(G)
    for a in range(n):
        for b in range(n):
            if f[a, b] > ts:
                lines = get_points(a, b)
                lines = np.array(lines)
                GG[lines[:, 0], lines[:, 1]] = 2
    if vis:
        transimg(GG).show()
    ok2 = f.max() > ts
    r = 4
    for a in range(n):
        for b in range(n):
            if G[a, b] == 0: continue
            flag = False
            for i in range(-r, r):
                for j in range(-r, r):
                    x, y = a + i, b + j
                    if inbound(x, y):
                        if GG[x, y] == 2:
                            flag = True
                            break
            if flag == False:
                ok2 = False

    return ok1 or ok2


if __name__ == '__main__':
    X, y = getdata()
    X1 = X[y == 1]
    Xn1 = X[y != 1]
    res = np.array([judge1(G) for G in X1[:1000]])
    print(res.shape, res.sum())
    # print(np.where(res)[0])
    res = np.array([judge1(G) for G in Xn1[:1000]])
    print(res.shape, res.sum())
    # print(np.where(res)[0])

    # ok = judge1(Xn1[1], vis=True)
    # print(ok)
```
老师，只使用搜索算法，当前我在MNIST数据集中尝试对判别是否是1的真正率82.6%，真负率89.3%，如果引入更复杂的判别技巧会更高

现在的做法是在图片中仿照人手写数字游走出一条轨迹，大致是数字的形状，如果存在一条游走轨迹的像素点是图片中原有数字区域像素点的子集，则判为正例。但是这只是正例的必要非充分条件，例如会把4、7也认定为1，因此又要求原有数字像素点的邻域内必须存在游走轨迹像素点。

遇到的问题是对数字形状没有准确的形式化定义，只是在使用一些技巧进行判别，因此会存在许多反例。对于判别是否是1应该相对简单，如果继续提高精度要引入许多复杂的规则。同时数字的形状越复杂，需要的判别规则就越多。



轨迹的长度和角度需要手动调节


对于比较相似的数字，可能需要特别的技巧进行区分，比如判别是否为1，负例是7或9的时候

继续提高精度的方法是查看判断错误
```