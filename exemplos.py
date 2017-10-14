
import cv2
from matplotlib import pyplot as plt

def segmentation_limiar(path_img):

    # lê a imagem
    img = cv2.imread(path_img, 0)

    # limiar global
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # limiar Otsu's
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # limiar do Otsu's depois do filtro Gaussian
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # plot todas as imagens e seus histogramas
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]

    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':

    segmentation_limiar('/home/pavic/PycharmProjects/PDI-Medical/data/normal/1 (1).png')