from sklearn.naive_bayes import MultinomialNB

# [é gordinho ?, tem perninhas curta ?, faz auau ?]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
marcacoes = [1, 1, 1, -1, -1, -1]

# Qual é o animal ?
misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

teste = [misterioso1, misterioso2, misterioso3]

marcacoes_teste = [-1, 1, -1]

resultado = modelo.predict(teste)
diferencas = resultado - marcacoes_teste
acertos = [d for d in diferencas if d == 0]

total_acertos = len(acertos)
total_elementos = len(teste)
taxa_acerto = 100 * total_acertos / total_elementos

print('a taxa de acertos é de ', taxa_acerto)
