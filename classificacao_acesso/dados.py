import csv

def carregar_acessos():
    x = []
    y = []

    arquivo = open('acesso.py', 'r')
    leitor = csv.reader(arquivo)

    next(leitor)

    for home, como_funciona, contato, comprou in leitor:
        dado = [int(home), int(como_funciona), int(contato)]
        x.append(dado)
        y.append(int(comprou))

    return x, y
