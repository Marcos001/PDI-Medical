#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

"""
        nome_img = lista[i].split('/')
        nome_img = nome_img[len(nome_img)-1]
        p = (((i+1 + inc) * 100) / size)
        print(i, ' -> descrevendo imagem ' + nome_img + ' da classe ' + str(classe), ' => ',  '%.1f%s' %(p,'%') )
"""

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def get_maior_pixel(imagem):
    maior = 0
    for i in range(imagem.shape[0]): # percorre as linhas
        for j in range(imagem.shape[1]): # percorre as colunas
            if imagem[i][j] > maior:
                maior = imagem[i][j]
    return maior

def normalize(valor):
    return int(valor) / 255


def getMinimum(image):
    ''''''
    menor = 255
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] < menor:
                menor = image[i][j]
    return menor


def getMaximum(image):
    ''''''
    maior = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > maior:
                maior = image[i][j]
    return maior


def ver_imagem(path_img, nome_imagem=None):
    '''
    :param path_img: caminho da imagem ou matriz de imagem
    :param nome_img: nome da imagem para exibir quando a imagem for exibida (opcional)
    :return: vazio
    '''

    if nome_imagem is None:
        nome_img = 'Visualizado Imagem com CV2'
    else:
        nome_img = nome_imagem

    if type(path_img) == str:

        imagem = cv2.imread(path_img)

        try:
            nome_img = path_img.split('/')
            nome_img = nome_img[len(nome_img) - 1]
        except:
            nome_img = path_img
    else:
        imagem = path_img


    cv2.imshow(nome_img, imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segmentar_imagem_com_kmeans(imagem):
    Z = imagem.reshape((-1,3))

    #converte Z para float32
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