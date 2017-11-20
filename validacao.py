
def acuracia(vn, fp, fn, vp):
    return (vp+vn) / (vp+vn+fp+fn)

def especificidade(vn, fp):
    return (vn) / (vn+fp)

def sensibilidade(fn, vp):
    return (vp) / (vp+fn)


def main_validacao(vn, fp, fn, vp):
    """
    :param vn: verdadeiros positivos
    :param fp: falsos positivos
    :param fn: falsos negativos
    :param vp: verdadeiros positivos
    :return:
    """
    print(' Acuracia______________[ %.2f%s]' %((acuracia(vn, fp, fn, vp)*100), "%"))
    print(' Especificidade________[ %.2f%s ]' %((especificidade(vn, fp)*100), "%"))
    print(' Sensibilidade_________[ %.2f%s ]' %((sensibilidade(fn, vp)*100),"%"))