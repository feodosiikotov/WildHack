import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import torch
from torch import nn
from letters_dicts import token_dict, idx2token
from top_words import top_words
from Levenshtein import distance
from torch import nn
import torch


query_popularity = pd.read_csv('query_popularity.csv')
query_popularity['query'] = query_popularity['query'].str.lower()

model = Word2Vec.load("word2vec.model")

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.pred = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids):
        embs = self.emb(input_ids)
        output, _ = self.lstm(embs)
        return self.pred(output)
    
model_lstm = LSTM(len(token_dict), 128, 128).cpu()
model_lstm.load_state_dict(torch.load('wildhack_lstm3.pth'))

def predict(x, one_word = True):
  # Функция для дополнения нейронной сетью слова  
  x0 = x
  x = [3] + [token_dict[i] for i in x]
  ans = x + [0]*100
  for i in range(len(x), 100):
    with torch.no_grad():
        pred = model_lstm(torch.tensor(ans, dtype=int).unsqueeze(0)).argmax(2).squeeze(0)
    ans[i] += int(pred[i-1])
  ans = [idx2token[i] for i in ans][1:]
  if '<eos>' in ans:
    eos = ans.index('<eos>')
    ans = ans[:eos]
  if one_word:
    init_len = len(x0.split())
    ans = ''.join(ans)
    ans = ans.split()
    ans = ans[:init_len]
    ans = ' '.join(ans)
  else:
    ans = ''.join(ans)
  return ans

def get_recomendation(text, how_many=100):
    # Рекомендация моделью w2v
    recomendations = model.wv.most_similar(text, topn=300)
    good_ones = [x[0] for x in recomendations if text in x[0]]
    result = query_popularity[query_popularity['query'].isin(good_ones)].sort_values('query_popularity', ascending=False)['query'].drop_duplicates()[:how_many]
    return np.array(result)

def remove_duplicates(recomendations, treshold=0.5):
    # Удаление похожих запросов
    queries_list = set()
    minus_set = set()
    for i in range(len(recomendations)):
        for j in range(i+1, len(recomendations)):
            tmp0 = []
            tmp1 = []
            for w in set(recomendations[i].split()) | set(recomendations[j].split()):
                if w not in recomendations[i]:
                    tmp1.append(w)
                elif w not in recomendations[j]:
                    tmp0.append(w)

            queries_list |= {recomendations[i], recomendations[j]}
            n = len(tmp0)*len(tmp1)
            if n == 0:
                if len(tmp0) == 0:
                    minus_set |= {recomendations[i]}
                else:
                    minus_set |= {recomendations[j]}
    queries_list -= minus_set
    queries_list = list(queries_list)
    a0 = []
    a1 = []
    a2 = []
    for i in range(len(queries_list)):
        for j in range(i+1, len(queries_list)):
            a0.append(queries_list[i])
            a1.append(queries_list[j])
            tmp0 = []
            tmp1 = []
            for w in set(queries_list[i].split()) | set(queries_list[j].split()):
                if w not in queries_list[i]:
                    tmp1.append(w)
                elif w not in queries_list[j]:
                    tmp0.append(w)
            n = len(tmp0)*len(tmp1)
            if n == 0:
                a2.append(1)
                continue
            s = 0
            for w1 in (tmp1):
                for w2 in (tmp0):
                    try:
                        s += model.wv.similarity(w1, w2)
                    except:
                        s += 0
            a2.append(s/n)
    df = pd.DataFrame({'w1': a0, 'w2': a1, 'w3': a2}).sort_values(by='w3')
    df = df[df['w3'] <= treshold]
    ans = list((set(df.w1) | set(df.w2)))
    ans = query_popularity[query_popularity['query'].isin(ans)].sort_values('query_popularity', ascending=False)['query'].drop_duplicates()[:10]
    return np.array(ans)
    
def correct_mistakes(word):
    # Исправление опечаток
    min_distance = float('inf')
    closest_word = None
    for w in top_words:
        d = distance(word, w)
        if d == 0:
            return w
        if d < min_distance:
            min_distance = d
            closest_word = w
    return closest_word

def recommend(text):
    if text == '':
        return np.random.choice(query_popularity[query_popularity['query_popularity'] == 1]['query'], 10, replace=False)
    new_text = predict(text)
    if len(new_text) > len(text):
        new_text = new_text.split()
        for i in range(len(new_text)):
            new_text[i] = correct_mistakes(new_text[i])
        new_text = ' '.join(new_text)
    try:
        recs = get_recomendation(new_text)
        recs = remove_duplicates(recs)
    except KeyError:
        recs = [new_text]
    return recs
        
        