import numpy as np
import cv2 as cv


def non_coplanar_calibration(a1, a2, a3, b):
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


def L2_loss(world, image, matrix):
    sqr = 0
    mtx1 = matrix[0][:4]
    mtx2 = matrix[1][:4]
    mtx3 = matrix[2][:4]

    for a, b in zip(world, image):
        arr_b0 = b[0]
        arr_b1 = b[1]
        con = np.concatenate([np.array(a), [1]])
        mt1dotcon = mtx1.T.dot(con)
        mt2dotcon = mtx2.T.dot(con)
        mt3dotcon = mtx3.T.dot(con)
        sqr += ((arr_b0 - (mt1dotcon / mt3dotcon)) ** 2 + (arr_b1 - (mt2dotcon / mt3dotcon)) ** 2)
    sqr = sqr / len(image)
    print('\n--------------------------\n')
    print('MSE = %s\n' % sqr)
    print('\n--------------------------\n')


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
    a1 = matrix[0][:3].T
    a2 = matrix[1][:3].T
    a3 = matrix[2][:3].T
    b = []
    for a in range(len(matrix)):
        b.append(matrix[a][3])
    b = np.reshape(b, (3, 1))
    return a1, a2, a3, b, matrix


def capture_image():
    filename = input("enter a filename")

    if len(filename) > 1:
        image = cv.imread(filename)
    else:
        print('enter a filename')

    return image


def feature_points(pic):
    termination = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objective = np.zeros((7 * 7, 3), np.float32)
    objective[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)

    world = []
    image = []

    togrey = cv.cvtColor(pic, cv.COLOR_RGBA2GRAY)
    val, corners = cv.findChessboardCorners(togrey, (7, 7), None)

    if val:
        world.append(objective)
        dou = cv.cornerSubPix(togrey, corners, (11, 11), (-1, -1), termination)
        image.append(corners)
        cv.drawChessboardCorners(pic, (7, 7), dou, val)
        cv.imshow('image', pic)

    world_file = open("worldpoint.txt", "w")
    for p in world:
        world_file.write("%s\n" % p)
    world_file.close()

    image_file = open("imagepoint.txt", "w")
    for p in image:
        image_file.write("%s\n" %p)
    image_file.close()
    print("press 'x' to quit ")
    while True:
        k = cv.waitKey()
        if k == ord('x'):
            cv.destroyAllWindows()


def main():
    print("enter 'w' to get parameters and MSE ")
    print("enter 't' to show chess board and save feature points ")
    print("enter 'q' to quit ")
    selection = input("enter selection: ")
    if selection == 'w':
        world, image = getdata()
        mtx = mtx_world_image(world, image)
        a1, a2, a3, b, matrix = out(mtx)
        non_coplanar_calibration(a1, a2, a3, b)
        L2_loss(world, image, matrix)
        main()

    elif selection == 't':
        pic = capture_image()
        feature_points(pic)

    elif selection == 'q':
        cv.destroyAllWindows()

    else:
        print("invalid choice. please re-enter")
        main()


if __name__ == '__main__':
    main()





