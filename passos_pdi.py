"""
script para aquisição de imagens
"""
import glob as g
import numpy as np
import cv2
from matplotlib import pyplot as plt

def aquisicao_imagens(path, extensao):
    '''
    :param path: caminho do diretório que contém as imagens
    :param extensao: extensão das imagens. EX: .png
    :return: lista com o caminho de todas as imagens do diretório
    '''
    lista = g.glob(pathname=path+'*'+extensao)
    return lista

def _ver_imagem(path_img):
    '''
    :param path_img: caminho da imagem
    :return:
    '''
    try:
        nome_img = path_img.split('/')
        nome_img = nome_img[len(nome_img)-1]
    except:
        nome_img = path_img

    print('nome da imagem = ', nome_img)
    imagem  = cv2.imread(path_img)
    print('type = ', type(imagem))
    cv2.imshow(nome_img, imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmentar_com_otsu(path_img):
    img = cv2.imread(path_img, 0)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    return img

def segmentar_imagem_com_kmeans(imagem):
    Z = imagem.reshape((-1,3))

    #convert Z to float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((imagem.shape))

    return res2


if __name__ == '__main__':
    ""
    #Arquisição de Imagens
    #caminho_imagens = aquisicao_imagens(path='/media/pavic/Arquivos/data/dataBaseRetina/RIM ONE/RIM-ONE 1/deep_src/',  extensao='*.bmp')


    #segmentação
    #segmentar_imagem_com_kmeans(cv2.imread('/home/pavic/PycharmProjects/PDI-Medical/data/normal/1 (1).png'))

    segmentar_com_otsu('/home/pavic/PycharmProjects/PDI-Medical/data/normal/1 (1).png')

    #sobreposição



