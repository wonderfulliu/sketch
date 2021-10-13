import cv2
import os

src_image = 'img.png'

style = 2

img_rgb = cv2.imread(src_image)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

if style == 1:
    # 风格1
    img_gray = cv2.medianBlur(img_gray, 5)
    img_edge = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=3, C=2)
if style == 2:
    # 风格2
    img_blur = cv2.GaussianBlur(img_gray, ksize=(11, 11), sigmaX=0, sigmaY=0)
    img_edge = cv2.divide(img_gray, img_blur, scale=255)


cv2.imwrite(os.getcwd() + '/result.jpg', img_edge)
