from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import sys
import os
import io
import traceback
from scipy.spatial import ConvexHull
from datetime import datetime

#ImageFile.LOAD_TRUNCATED_IMAGES = True
camera_channels = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT']

def resize_image(image, target_size):
    """
    调整图像大小到目标尺寸。
    :param image: PIL.Image对象
    :param target_size: 目标尺寸，格式为(width, height)
    :return: 调整大小后的图像
    """
    image = image.convert('RGB')
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    return resized_image


def render_map_on_image(nusc, nusc_map, sample_token, camera_channel):
    sample_record = nusc.get('sample', sample_token)
    cam_token = sample_record['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)
    cam_path = nusc.get_sample_data_path(cam_token)

    # 检查文件是否存在
    if not os.path.exists(cam_path):
        raise FileNotFoundError(f"Image file not found at {cam_path}")

    # 如果文件存在，继续处理
    img = Image.open(cam_path)

    map_img, _ = nusc_map.render_map_in_image(nusc, sample_token, camera_channel=camera_channel,alpha=1)

    plt.close(map_img)  # 关闭图形，释放资源

    # 转换Figure到PIL.Image（用于后续的图像处理）
    buf = io.BytesIO()
    dpi = 1600 / 9  # 假设figsize的宽度为9英寸，想要得到1600像素的宽度
    map_img.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)

    buf.seek(0)
    pil_img = Image.open(buf)

    # 返回原始图像和转换后的地图图像
    return img, pil_img


def identify_road_area(original_img, map_img):
    # 将两个图像转换为numpy数组
    original_np = np.array(original_img)
    map_np = np.array(map_img)

    # 如果map_img具有4个通道（即RGBA），则仅保留前三个通道（RGB）
    if map_np.shape[2] == 4:
        map_np = map_np[:, :, :3]

    # 现在两个数组具有相同的形状，可以安全进行操作
    diff = np.abs(original_np - map_np)

    # 将差异转换为灰度图，然后应用阈值来识别路面区域
    diff_gray = np.mean(diff, axis=2)

    road_mask = diff_gray > np.mean(diff_gray) * 0.3  # 这个阈值可能需要调整以适应不同的场景和数据集

    return road_mask


# def create_road_mask_by_color_thresholds(map_img, color_thresholds):
#     """
#     创建一个基于多个颜色阈值的路面掩码。
#     :param map_img: 包含渲染地图的PIL.Image对象。
#     :param color_thresholds: 一个包含(r, g, b)颜色阈值的列表。
#     :return: 一个布尔型numpy数组，其中True代表路面。
#     """
#     map_np = np.array(map_img.convert('RGB'))  # 确保图像是RGB格式
#
#     # 初始化一个形状和地图图像相同的布尔型数组，用False填充
#     road_mask = np.zeros(map_np.shape[:2], dtype=bool)
#
#     # 对每个颜色阈值创建一个掩码，并将其合并到road_mask中
#     for color in color_thresholds:
#         lower_bound = np.array([max(0, c - 10) for c in color])  # 设定颜色的下界
#         upper_bound = np.array([min(255, c + 10) for c in color])  # 设定颜色的上界
#
#         # 使用logical_or更新road_mask以包括所有在颜色范围内的像素
#         road_mask = np.logical_or(road_mask, np.all(np.logical_and(map_np >= lower_bound, map_np <= upper_bound), axis=2))
#
#     return road_mask


def apply_road_cover(original_img, road_mask, cover_color=(255, 255, 255)):
    # 将原始图像转换为numpy数组
    original_img_np = np.array(original_img)

    # 遮罩转换为布尔型，大于0的为True
    road_mask_bool = road_mask > 0

    # 分别对每个通道应用遮罩和颜色
    for i in range(3):  # 遍历R、G、B三个颜色通道
        original_img_np[:, :, i][road_mask_bool] = cover_color[i]

    # 将修改后的numpy数组转换回PIL.Image对象
    covered_img = Image.fromarray(original_img_np)

    return covered_img

def align_images(im1, im2):
    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测ORB特征并计算描述符。
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)

    # 使用Hamming距离进行匹配并排序匹配结果
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # 使用Hamming距离进行匹配
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # 只保留好的匹配点，不再调用 sort 方法
    numGoodMatches = int(len(matches) * 0.15)
    matches = sorted(matches, key=lambda x: x.distance)[:numGoodMatches]  # 直接对 matches 进行排序和截取

    # 从好的匹配点中提取位置
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # 寻找两个图像之间的变换矩阵
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # 使用变换矩阵将图像 im1 对齐到图像 im2
    height, width, channels = im2.shape
    im1_aligned = cv2.warpPerspective(im1, h, (width, height))

    return im1_aligned

def get_boxes(sample_data_token):
    _, boxes, camera_intrinsic = nusc.get_sample_data(sample_data_token, BoxVisibility.ANY)
    print(f"Boxes: {boxes}, Camera Intrinsic: {camera_intrinsic}")
    return boxes, camera_intrinsic


def modify_road_mask_with_boxes(road_mask, boxes, camera_intrinsic, img_size, sample_token, camera_channel):
    combined_boxes_mask = np.zeros(road_mask.shape[:2], dtype=np.uint8)

    for box in boxes:
        box_corners_3d = box.corners()
        box_corners_2d = view_points(box_corners_3d, camera_intrinsic, normalize=True)[:2, :]

        # 只选择在相机前方的点进行投影
        in_front_of_camera = box_corners_3d[2, :] > 0
        box_corners_2d = box_corners_2d[:, in_front_of_camera]

        if box_corners_2d.shape[1] >= 3:
            projected_corners = box_corners_2d.T
            hull = ConvexHull(projected_corners)
            hull_points = projected_corners[hull.vertices].astype(np.int32)
            cv2.fillPoly(combined_boxes_mask, [hull_points], color=(255))

    # 确保 road_mask 是 uint8 类型
    if road_mask.dtype != np.uint8:
        road_mask = (road_mask * 255).astype(np.uint8)

    # 确保 inverted_combined_boxes_mask 也是 uint8 类型
    inverted_combined_boxes_mask = cv2.bitwise_not(combined_boxes_mask)
    if inverted_combined_boxes_mask.dtype != np.uint8:
        inverted_combined_boxes_mask = inverted_combined_boxes_mask.astype(np.uint8)

    # 在进行按位与操作之前，检查数组是否为空
    if road_mask.size == 0 or inverted_combined_boxes_mask.size == 0:
        raise ValueError("One of the masks is empty, cannot perform bitwise and operation")

    road_mask_modified = cv2.bitwise_and(road_mask, inverted_combined_boxes_mask)

    return road_mask_modified


def process_dataset(nusc,dataroot, save_dir):
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化地图缓存字典
    map_cache = {}

    # 遍历所有的样本
    for sample in nusc.sample:
        sample_token = sample['token']

        # 获取样本的地图名称
        scene_record = nusc.get('scene', sample['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        map_name = log_record['location']

        # 检查地图是否已经被加载
        if map_name not in map_cache:
            # 如果地图未加载，加载地图并添加到缓存字典
            map_cache[map_name] = NuScenesMap(dataroot=dataroot, map_name=map_name)

        nusc_map = map_cache[map_name]

        # 对每个样本中的所有摄像头通道图像进行处理
        for camera_channel in camera_channels:
            # 创建相应的目录
            channel_dir = os.path.join(save_dir, camera_channel)
            if not os.path.exists(channel_dir):
                os.makedirs(channel_dir)

            sample_record = nusc.get('sample', sample_token)
            cam_token = sample_record['data'][camera_channel]
            cam_data = nusc.get('sample_data', cam_token)

            # 获取时间戳并转换格式
            timestamp = cam_data['timestamp']
            date_time = datetime.utcfromtimestamp(timestamp / 1e6)
            date_time_str = date_time.strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]

            try:
                print(f"Processing sample {sample_token}, channel {camera_channel}")

                original_img, map_img = render_map_on_image(nusc, nusc_map, sample_token, camera_channel)
                original_np = np.array(original_img)
                map_np = np.array(map_img)
                aligned_map = align_images(map_np, original_np)

                road_mask = identify_road_area(original_img, aligned_map)

                road_mask_uint8 = (road_mask * 255).astype(np.uint8)
                road_mask_image = Image.fromarray(road_mask_uint8)

                try:
                    boxes, camera_intrinsic = get_boxes(cam_token)
                except KeyError as e:
                    print(f"Token {e} not found in dataset. Skipping...")
                    continue

                road_mask_modified = modify_road_mask_with_boxes(road_mask, boxes, camera_intrinsic, original_img.size,
                                                                 sample_token, camera_channel)
                # 组合符合NuScenes命名规范的文件名
                base_name = f"{log_record['vehicle']}_{date_time_str}"
                filename = f"{base_name}__{camera_channel}__{timestamp}.jpg"
                file_path = os.path.join(channel_dir, filename)

                covered_img = apply_road_cover(original_img, road_mask_modified, cover_color=(255, 255, 255))
                covered_img.save(file_path, format='JPEG')

                print(f'Saved processed image to {file_path}')

            except Exception as e:
                print(f"Error processing sample {sample_token}, channel {camera_channel}: {str(e)}")
                traceback.print_exc()
            finally:
                plt.close('all')  # 确保关闭所有打开的图像


# # 调用函数处理整个数据集，指定数据集版本token
# dataroot = '/Users/fmy/Desktop/test/v1.0-mini'
# save_dir = '/Users/fmy/Desktop/test/processed'
# version = 'v1.0-mini'
# verbose=True
# # 初始化NuScenes实例
# nusc = NuScenes(version, dataroot, verbose)
# process_dataset(dataroot, save_dir)
def main():
    if len(sys.argv) < 4:
        print("Usage: python process_snow.py <dataroot> <img_path> <save_path>")
        sys.exit(1)

    dataroot = sys.argv[1]
    img_path = sys.argv[2]
    save_path = sys.argv[3]

    # 确保保存路径的目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 初始化NuScenes实例
    version = 'v1.0-trainval'  # 或者根据需要调整
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # 调用处理函数，传递NuScenes实例
    process_dataset(nusc, img_path, save_path)

if __name__ == "__main__":
    main()
