import pickle
import cv2
import numpy as np

# 可视化predict和gt


def bev_predictBox_with_gtBox(points, gt, predict_result, score_threshold = 0.1, image_size=2048, save_filename='bev_visualize.jpg'):

    x,y,z,_,_ = points.T

    pc_range = 60
    x = x / pc_range
    y = y / pc_range

    # 缩放到图像大小，并平移到图像中心
    # 图像坐标系为：
    # 0------>x
    # |
    # |
    # y

    # lidar坐标系俯视图为
    #          x
    #          |
    #          |
    #    y<----0
    half_image_size = image_size / 2
    center_x = half_image_size
    center_y = half_image_size

    x = -x * half_image_size + center_x
    y = -y * half_image_size + center_y

    # opencv的图像，可以用numpy进行创建
    image = np.zeros((image_size, image_size, 3), np.uint8)

    for ix, iy, iz in zip(x, y, z):
        ix = int(ix)
        iy = int(iy)

        # 判断是否在图像范围内
        if ix >= 0 and ix < image_size and iy >= 0 and iy < image_size:
            image[ix, iy] = 255, 255, 255


    for j in gt.corners:
        corner = [j[0], j[3], j[7], j[4]]
        corner = [(-i * half_image_size / pc_range + half_image_size)[:2] for i in corner]

        # 由于plot的x,y是正常的x,y而不是opencv image里的x,y
        # 所以这里需要交换x,y
        #实验
        # image[int(rec[0][0]),int(rec[0][1])] = 255,0,0
        # cv2.line(image,(0,100),(1024,0),(0,255,0),3)
        corner = [tuple(reversed(i.int().tolist())) for i in corner]

        cv2.line(image, corner[0], corner[1], (0, 255, 0), 2)
        cv2.line(image, corner[1], corner[2], (0, 255, 0), 2)
        cv2.line(image, corner[2], corner[3], (0, 255, 0), 2)
        cv2.line(image, corner[3], corner[0], (0, 255, 0), 2)

    for index,j in enumerate(predict_result['boxes_3d'].corners):
        if predict_result['scores_3d'][index] < score_threshold:
            continue
        corner = [j[0], j[3], j[7], j[4]]
        corner = [(-i * half_image_size / pc_range + half_image_size)[:2] for i in corner]

        # 由于plot的x,y是正常的x,y而不是opencv image里的x,y
        # 所以这里需要交换x,y
        # 实验
        # image[int(rec[0][0]),int(rec[0][1])] = 255,0,0
        # cv2.line(image,(0,100),(1024,0),(0,255,0),3)
        corner = [tuple(reversed(i.int().tolist())) for i in corner]

        # BGR
        red = (0,0,255)
        cv2.line(image, corner[0], corner[1], red, 2)
        cv2.line(image, corner[1], corner[2], red, 2)
        cv2.line(image, corner[2], corner[3], red, 2)
        cv2.line(image, corner[3], corner[0], red, 2)

    cv2.imwrite(save_filename, image)
    # cv2.imshow('image', image)  # 建立名为‘image’ 的窗口并显示图像
    # k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    # if k == 27:  # 键盘上Esc键的键值
    #     cv2.destroyAllWindows()



def discard():
    # 打开文件，这里假设文件名为 'data.pkl'
    with open('pkl/transfusion_nusc_voxel_L_bs4_3layer_new_pipeline.py_batch_0_new.pkl', 'rb') as file:
        #     # 使用pickle.load()从文件中加载对象
        data = pickle.load(file)

    x, y, z, _,_= data['points'].T

    # 设置图像的尺寸1024x1024
    image_size = 1024

    # 数据归一化
    pc_range = 60
    x = x / pc_range  # [-1,1]
    y = y / pc_range

    # 缩放到图像大小，并平移到图像中心
    # 图像坐标系为：
    # 0------>x
    # |
    # |
    # y

    # lidar坐标系俯视图为
    #          x
    #          |
    #          |
    #    y<----0
    half_image_size = image_size / 2
    center_x = half_image_size
    center_y = half_image_size

    x = -x * half_image_size + center_x
    y = -y * half_image_size + center_y

    # opencv的图像，可以用numpy进行创建
    image = np.zeros((image_size, image_size, 3), np.uint8)

    for ix, iy, iz in zip(x, y, z):
        ix = int(ix)
        iy = int(iy)

        # 判断是否在图像范围内
        if ix >= 0 and ix < image_size and iy >= 0 and iy < image_size:
            image[ix, iy] = 255, 255, 255

    # 从liarInstanceBox拿到定义
    """torch.Tensor: Coordinates of corners of all the boxes
            in shape (N, 8, 3).
    
            Convert the boxes to corners in clockwise order, in form of
            ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``
            需要的rectangle对角线就是(3,4) (0,7)
    
            .. code-block:: none
    
                                               up z
                                front x           ^
                                     /            |
                                    /             |
                      (x1, y0, z1) + -----------  + (x1, y1, z1)
                                  /|            / |
                                 / |           /  |
                   (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                                |  /      .   |  /
                                | / origin    | /
                left y<-------- + ----------- + (x0, y1, z0)
                    (x0, y0, z0)
            """
    for j in data['corners']:
        corner = [j[0], j[3], j[7], j[4]]
        corner = [(-i * half_image_size / pc_range + half_image_size)[:2] for i in corner]

        # 由于plot的x,y是正常的x,y而不是opencv image里的x,y
        # 所以这里需要交换x,y
        #实验
        # image[int(rec[0][0]),int(rec[0][1])] = 255,0,0
        # cv2.line(image,(0,100),(1024,0),(0,255,0),3)
        corner = [tuple(reversed(i)) for i in corner]
        cv2.line(image, corner[0], corner[1], (0, 255, 0), 2)
        cv2.line(image, corner[1], corner[2], (0, 255, 0), 2)
        cv2.line(image, corner[2], corner[3], (0, 255, 0), 2)
        cv2.line(image, corner[3], corner[0], (0, 255, 0), 2)

    cv2.imwrite("centerpoint.jpg", image)
    cv2.imshow('image', image)  # 建立名为‘image’ 的窗口并显示图像
    k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
    if k == 27:  # 键盘上Esc键的键值
        cv2.destroyAllWindows()