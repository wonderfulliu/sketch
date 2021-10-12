import numpy as np
import shutil
from PIL import Image
import cv2
import os
from cv2 import VideoWriter_fourcc

# 定义变量
# 要转换的视频的名字
video_name = 'video.mp4'

res_dic = 'result'
img_dic = 'images'
ske_dic = 'sketch'
res_video_name = 'result'

# 定义文件夹路径
# 当前文件所在路径
root = os.getcwd()
# 存储处理文件的路径
result_path = root + '/' + res_dic
# 存储分割的图片的路径
img_path = root + '/' + res_dic + '/' + img_dic
# 存储素描图片的路径
sketch_path = root + '/' + res_dic + '/' + ske_dic
# 存储素描视频的路径
video_path = root + '/' + res_dic + '/' + res_video_name + '.mp4'


# 判断文件夹是否存在: result、images、sketch, 不存在则创建相应的文件夹
def init_directory():
    print('初始化文件夹...')

    # 存在存储处理文件的文件夹则删除
    if os.path.exists(result_path):
        clear_dir(result_path)

    # 创建result文件夹
    os.mkdir(result_path)
    # 创建存储分割图片的文件夹
    os.mkdir(img_path)
    # 创建存储素描图片的文件夹
    os.mkdir(sketch_path)

    print('初始化文件夹结束')


# 清空存储分割后图片的文件夹
def clear_dir(path):
    print('清理文件夹')
    shutil.rmtree(path)


# 视频转原始图片
def video_to_pic(imgPath):
    print('切割视频...')
    cap = cv2.VideoCapture(video_name)
    success = cap.isOpened()
    frame_count = 0
    i = 0

    while success:
        frame_count += 1
        success, frame = cap.read()

        if frame_count % 2.5 == 0:
            i += 1
            cv2.imwrite(imgPath + '/%d.jpg' % i, frame)
    cap.release()
    print('切割视频完成')


# 原始图片转素描图片
def pic_to_sketch(imgPath, sketchPath):
    print('原图转素描...')
    images = os.listdir(imgPath)
    for num_count in range(len(images)):
        a = np.asarray(Image.open(imgPath + '/' + str(num_count + 1) + '.jpg').convert('L')).astype('float')

        depth = 10.
        grad = np.gradient(a)
        grad_x, grad_y = grad

        grad_x = grad_x * depth / 100.
        grad_y = grad_y * depth / 100.

        A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)
        uni_x = grad_x / A
        uni_y = grad_y / A
        uni_z = 1. / A
        vec_el = np.pi / 2.2
        vec_az = np.pi / 4.
        dx = np.cos(vec_el) * np.cos(vec_az)
        dy = np.cos(vec_el) * np.sin(vec_az)
        dz = np.sin(vec_el)
        b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
        b = b.clip(0, 255)
        im = Image.fromarray(b.astype('uint8'))
        im.save(sketchPath + '/' + str(num_count + 1) + '.jpg')
    print('原图转素描完成')


# 素描图片转视频
def sketch_to_video(sketchPath, videoPath):
    print('素描转视频...')
    images = os.listdir(sketchPath)
    images.sort(key=lambda x: int(x[:-4]))

    fps = 12

    fourcc = VideoWriter_fourcc(*'mp4v')
    image = Image.open(sketchPath + '/' + images[0])

    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)

    for im_name in range(len(images)):
        frame = cv2.imread(sketchPath + '/' + images[im_name])
        videoWriter.write(frame)

    print("素描转视频完成")
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 初始化文件夹
    init_directory()

    # 第一步
    video_to_pic(img_path)

    # 第二步
    pic_to_sketch(img_path, sketch_path)

    # 第三步
    sketch_to_video(sketch_path, video_path)
