from PIL import Image
import cv2
import os
from cv2 import VideoWriter_fourcc

# 定义文件夹路径
# 当前文件所在路径
root = os.getcwd()
# 存储素描图片的路径†
sketch_path = root + '/郑荷花_20220107110351611'
# 视频帧率
video_fps = 24
# 存储素描视频的路径
video_path = root + '/result-' + str(video_fps) + '.mp4'


# 素描图片转视频
def sketch_to_video(sketchPath, videoPath, fps=12):
    print('转换开始...')
    images = os.listdir(sketchPath)
    images.sort()
    fourcc = VideoWriter_fourcc(*'mp4v')
    image = Image.open(sketchPath + '/' + images[0])

    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)

    for im_name in range(len(images)):
        frame = cv2.imread(sketchPath + '/' + images[im_name])
        videoWriter.write(frame)

    print("转换完成")
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 第三步
    sketch_to_video(sketch_path, video_path, video_fps)
