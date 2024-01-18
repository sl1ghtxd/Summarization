# -*- coding: utf-8 -*-
import pymorphy2, json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

morph = pymorphy2.MorphAnalyzer()
VECTOR = Word2Vec.load("olega.bin")


def tokenizer(text):
    global tokenized
    for comment in text.split("."):
        tokenized.append(simple_preprocess(comment, deacc=True, min_len=1))

###МЕЖДУ РЕШЕТКАМИ РЕАЛИЗОВАНО СЧИТЫВАНИЕ ИЗ TXT И JSONL ФАЙЛОВ###
tokenized = []
with open("edu.txt", "r", encoding="utf-8") as f:
    s = f.read()
# with open("edu.jsonl", "r") as f:
#     x = list(f)
# fulldata = []
# for json_str in x:
#     data = json.loads(json_str)
#     fulldata.append(data)
# sorteddata = [[0] for i in range(len(fulldata))]
# comments = []
# posts = 0
# for i in range(len(fulldata)):
#     if 'root_id' not in fulldata[i]:
#         sorteddata[posts][0] = fulldata[i]
#         posts += 1
#     else:
#         comments.append(fulldata[i])
# s = ""
# for i in range (len(comments)):
#     s += str(comments[i]['text']) + " "
###МЕЖДУ РЕШЕТКАМИ РЕАЛИЗОВАНО СЧИТЫВАНИЕ ИЗ TXT И JSONL ФАЙЛОВ###

tokenizer(s)
normal = tokenized
for i in range(len(tokenized)):
    if len(tokenized[i]) != 0:
        for j in range(len(tokenized[i])):
            if len(tokenized[i][j]) != 0:
                normal[i][j] = morph.parse(tokenized[i][j])[0].normal_form
print(normal)
VECTOR.build_vocab(normal, update=True)
VECTOR.train(normal, total_examples=VECTOR.corpus_count, epochs=10)
VECTOR.save("olega.bin", sep_limit=1000000000) 
