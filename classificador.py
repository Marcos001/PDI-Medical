
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import normalize

from util.util import plot_confusion_matrix, confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from validacao import main_validacao

def get_data(path_data):
    """
    formata os dados do arquivo dos descritores
    para converter em uma lista de FEATURES e LABELS
    """

    FEATURES = [] # lista de caracteristicas
    LABELS = [] # lista de rotulos

    file_data = open(path_data, 'r')

    for i in file_data:
        lista = str(i).split(' ')
        F = []
        for j in range(len(lista)):
            if j == 0:
                LABELS.append(lista[j])
            else:
                if lista[j] != '\n':
                    value = lista[j].split(':')[1]
                    F.append(value)
        FEATURES.append(F)
    return LABELS, FEATURES

def gerar_matrix_confusao(y_test, y_pred ):
    # calcula a matrix de confusao
    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    np.set_printoptions(precision=2)

    class_names = [] # rotulos para o grafico da matrix de confusao
    class_names.append('Normal')
    class_names.append('Doente')

    # grafico de matrix de confusao com dados nao normalizados
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # grafico de matrix de confusao com dados normalizados
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    main_validacao(tn, fp, fn, tp) # exibi os valores de acuracia, sensibilidade e especificidade
    plt.show() # exibe o grafico


def classificar_com_SVM(X_train, X_test, y_train, y_test):

    print("SVM")

    c_svm = SVC() # cria uma instancia do modelo SVM

    c_svm.fit(X_train, y_train) # treina o modelo

    y_pred =  c_svm.predict(X_test) # faz a predicao sobre os dados de teste

    gerar_matrix_confusao(y_test, y_pred) #exibe o grafico com a matrix de confusao


def classificar_com_RandomForest(X_train, X_test, y_train, y_test):
    print("RandomForest")
    c_rf = RandomForestClassifier()

    c_rf.fit(X_train, y_train)
    rf_pred =  c_rf.predict(X_test)
    gerar_matrix_confusao(y_test, rf_pred)


def classificar_com_AdaBoostClassifier(X_train, X_test, y_train, y_test):
    print('AdaBoost')
    csf_ad = AdaBoostClassifier()
    csf_ad.fit(X_train, y_train)
    prediction = csf_ad.predict(X_test)
    gerar_matrix_confusao(y_test, prediction)



def model(LABELS, FEATURES):

    TRAIN = 0.2
    TEST = 1 - TRAIN

    # formata os dados de terino e teste
    X_train, X_test, y_train, y_test = train_test_split(FEATURES, LABELS, test_size=TEST)

    classificar_com_SVM(X_train, X_test, y_train, y_test) # classificador SVM

    classificar_com_RandomForest(X_train, X_test, y_train, y_test) # classificador Random Forest

    classificar_com_AdaBoostClassifier(X_train, X_test, y_train, y_test) # classificador AdaBoostClassifier


def classificar(path_arquivo_descritor):

    # obtem a lista de labels e caracteristicas
    L, F = get_data(path_arquivo_descritor)

    # normalizando os dados
    F = normalize(F, axis=0, norm='max')

    # cria o modelo, classifica e gera as matrizes de confusao
    model(L, F)
