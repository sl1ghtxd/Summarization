from gensim.models import Word2Vec

VECTOR = Word2Vec.load("olega.bin")

word = input('Введите слово для получения схожего: ')
n = int(input('Введите количество желаемых слов: '))

ans = []
for i in range(n):
    ans.append(VECTOR.wv.most_similar(word, topn=n)[i][0])
print(ans)
