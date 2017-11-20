
import glob as g # pacote para obter path de arquivos em um diretorio
import cv2 # opencv
from descritor_textura import extrair_caracteristicas # script de extraao de features
from classificador import classificar # script de classificacao

import os

def aquisicao_imagens(path, extensao):
    '''
    :param path: caminho do diretorio que contem as imagens
    :param extensao: extensao das imagens. EX: .png
    :return: lista com o caminho de todas as imagens do diretorio
    '''
    lista = g.glob(pathname=path+'*'+extensao)
    return lista


def sobrepor(imagem, mascara):
    '''
    :param imagem: imagem origem
    :param mascara: imagem que contem a mascara apontando a regiao de interese
    :return: a imagem sobreposta
    '''
    for i in range(mascara.shape[0]): # percorre as linhas
        for j in range(mascara.shape[1]): # percorre as colunas
            if mascara[i][j] == 0:
               imagem[i][j] = 0
    return imagem


def segmentar_com_otsu(path_img):
    '''
    :param path_img: caminho da imagem a ser segmentada
    :return: retorna a imagem segmentada e a imagem origem nos 3 canais RGB
    '''

    # le a imagem em tons de cinza
    img = cv2.imread(path_img, 0)
    # le a imagem nos tres canais
    img_channel = cv2.imread(path_img)
    # segmenta a imagem atraves de um limiar obtido pelo algoritmo OTSU
    ret, imagem_segmentada_por_limiar = cv2.threshold(img, 127 , 255, cv2.THRESH_OTSU)
    # retorna a imagem original e a segmentada
    return img_channel,  imagem_segmentada_por_limiar


if __name__ == '__main__':

    """
    # diretorio onde contem as imagens
    dir_in = '/home/nig/PycharmProjects/PDI-Medical/data/imagens/normal'
    # diretorio onde as imagens segmentadas e sobrepostas serao salvas
    dir_out = '/home/nig/PycharmProjects/PDI-Medical/data/segmentadas/normal'+str('/')

    # variavel para encontrar o nome da imagem
    separador = dir_in.split('/')
    separador = separador[len(separador)-1]

    # obtem uma lista com o caminho de todas as imagens
    caminho_imagens = aquisicao_imagens(path=dir_in+'/',  extensao='.png')

    # ciclo que realiza o processamento
    for i in caminho_imagens:

        # segmenta a e sobrepoe a imagem
        img_seg = segmentar_com_otsu(path_img=i)

        # obtem o nome da imagem
        nome_img = i.split('/'+separador+'/')[1]
        print(' segmentando e sobrepondo ', nome_img)

        # salva a imagem
        cv2.imwrite(filename=dir_out+nome_img, img=sobrepor(img_seg[0], img_seg[1]))
"""
    # realizando a extracao de caracteristicas
    extrair_caracteristicas(path_normal='/home/nig/PycharmProjects/PDI-Medical/data/imagens/mama/Benigna/',
                            path_doente='/home/nig/PycharmProjects/PDI-Medical/data/imagens/mama/Maligna/',
                            path_arquivo_descritor='/home/nig/PycharmProjects/PDI-Medical/data/descritor/features_mama.libvsm',
                            extensao='.png')

    #classificar
    classificar('/home/nig/PycharmProjects/PDI-Medical/data/descritor/features_mama.libvsm')
    print()



