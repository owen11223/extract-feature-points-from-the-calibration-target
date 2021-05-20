import cv2 as cv
import numpy as np
import random


def read_config():
    filename = input("please enter config file name")
    with open(filename, 'r') as config:
        probability = float(config.readline().split()[0])
        max_draw = int(config.readline().split()[0])
        min_fit = int(config.readline().split()[0])
        max_fit = int(config.readline().split()[0])
        #print(max_fit)
        #print(min_fit)
    return probability, max_draw, min_fit, max_fit


def getdata():
    worldfilename = input("please enter world file name")
    imagefilename = input("please enter image file name")
    world = []
    image = []

    with open(worldfilename) as file:
        data = file.readlines()
        for line in data:
            seg = line.split()
            world.append([float(w) for w in seg[:3]])

    with open(imagefilename) as igfile:
        igdata = igfile.readlines()
        for line in igdata:
            seg = line.split()
            image.append([float(w) for w in seg[:2]])
    #print(world)
    #print(image)
    return world, image


def mtx_world_image(world, image):
    mtx = []
    zr = np.zeros(4)
    for a, b in zip(world, image):
        con = np.concatenate([np.array(a), [1]])
        x = np.concatenate([con, zr, -b[0] * con])
        y = np.concatenate([zr, con, -b[1] * con])
        mtx.append(x)
        mtx.append(y)
    return np.array(mtx)


def out(mtx):
    u, s, vh = np.linalg.svd(mtx, full_matrices=True)
    matrix = vh[-1].reshape(3, 4)
    return matrix


def median(world, image, matrix):
    sqr = []
    mtx1 = matrix[0][:4]
    mtx2 = matrix[1][:4]
    mtx3 = matrix[2][:4]

    for a, b in zip(world, image):
        arr_b0 = b[0]
        arr_b1 = b[1]
        con = np.append(np.array(a), 1)
        mt1dotcon = mtx1.T.dot(con)
        mt2dotcon = mtx2.T.dot(con)
        mt3dotcon = mtx3.T.dot(con)
        sqr1 = ((arr_b0 - (mt1dotcon / mt3dotcon)) ** 2 + (arr_b1 - (mt2dotcon / mt3dotcon)) ** 2)
        sqr.append((np.sqrt(sqr1)))
    m = np.median(sqr)
    #print(m)
    return m, sqr


def dist(world, image, matrix):
    sqr = []
    mtx1 = matrix[0][:4]
    mtx2 = matrix[1][:4]
    mtx3 = matrix[2][:4]

    for a, b in zip(world, image):
        arr_b0 = b[0]
        arr_b1 = b[1]
        con = np.append(np.array(a), 1)
        mt1dotcon = mtx1.T.dot(con)
        mt2dotcon = mtx2.T.dot(con)
        mt3dotcon = mtx3.T.dot(con)
        sqr1 = ((arr_b0 - (mt1dotcon / mt3dotcon)) ** 2 + (arr_b1 - (mt2dotcon / mt3dotcon)) ** 2)
        sqr.append((np.sqrt(sqr1)))
    #print(m)
    return sqr


def rans(world, image, median):
    probability, max_draw, min_fit, max_fit = read_config()
    k = 0.5      #probability that a point is an inlier
    np.random.seed(0)
    count = 0
    inlier = 0
    sigma = 1.5 * median
    rand = random.randint(min_fit, max_fit)

    while count < max_draw:
        choice = np.random.choice(len(image), rand)
        worldran = np.array(world)[choice]
        imageran = np.array(image)[choice]
        mtx = mtx_world_image(worldran, imageran)
        t = out(mtx)
        jrt = dist(world, image, t)
        inl = []
        for a, jrt in enumerate(jrt):
            if jrt < sigma:
                inl.append(a)
        if len(inl) >= inlier:
            inlier = len(inl)
            Q = mtx_world_image(worldran, imageran)
            bestone = out(Q)
        if not(k == 0):
            k = float(len(inl)) / float(len(image))
            max_draw = float(np.log(1 - probability)) / np.absolute(np.log(1 - k ** rand))
        count += 1
    return inlier, bestone


def non_coplanar_calibration(mat):
    a1 = mat[0][:3].T
    a2 = mat[1][:3].T
    a3 = mat[2][:3].T
    b = []
    for i in range(len(mat)):
        b.append(mat[i][3])
    b = np.reshape(b, (3, 1))
    p_nor = 1 / (np.linalg.norm(a3.T))
    a1_dot_a3 = a1.T.dot(a3)
    u0 = p_nor**2 * a1_dot_a3
    a2_dot_a3 = a2.T.dot(a3)
    v0 = p_nor**2 * a2_dot_a3
    av = np.sqrt(p_nor**2 * a2.T.dot(a2) - v0**2)
    s = p_nor**4 / av * (np.cross(a1.T, a3.T)).dot(np.cross(a2.T, a3.T))
    au = np.sqrt(p_nor**2 * a1.T.dot(a1) - s**2 - u0**2)
    k = np.array([[au, s, u0], [0, av, v0], [0, 0, 1]])
    epsilon = np.sign(b[2])
    t = epsilon * p_nor * np.linalg.inv(k).dot(b).T
    r3 = epsilon * p_nor * a3
    r1 = p_nor**2 / av * np.cross(a2.T, a3.T)
    r2 = np.cross(r3, r1)
    r = np.array([r1.T, r2.T, r3.T])
    print('\n--------------------------\n')
    print('u0 = %f\n' % u0)
    print('v0 = %f\n' % v0)
    print('Alpha_u = %f\n' % au)
    print('Alpha_v = %f\n' % av)
    print('s = %f\n' % s)
    print('T* = %s\n' % t)
    print('R* = %s\n' % r)
    print('\n--------------------------\n')


def main():
    world, image = getdata()
    mtx = mtx_world_image(world, image)
    t = out(mtx)
    m, sqr = median(world, image, t)
    ln, best = rans(world, image, m)
    non_coplanar_calibration(best)


if __name__ == '__main__':
    main()