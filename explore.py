import numpy as np
import pickle
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import jsonlines
from langdetect import detect
from collections import OrderedDict
from operator import itemgetter
import steamfront
import itertools

client = steamfront.Client()

X = []
y = []
z = []
games = []

def detect_text(text):
    try:
        language = detect(text)
    except:
        language = "unknown"

    return language

with jsonlines.open('data/reviews.txt', 'r') as f:
    for item in f:
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])
        games.append(item['appid'])
'''

ids = []
for g in games:
    ids.append(g)


id_freq = {}
for id in ids:
    if (id in id_freq):
        id_freq[id] += 1
    else:
        id_freq[id] = 1

games_list = []

games_freq = OrderedDict(sorted(id_freq.items(), key = itemgetter(1), reverse = True))
for k, v in itertools.islice(games_freq.items(), 20):

    game = client.getApp(appid=k)
    games_list.append((game.name, v))

print(games_list[:30])
'''

'''
[('Counter-Strike: Global Offensive', 428), ("PLAYERUNKNOWN'S BATTLEGROUNDS", 132), ('Grand Theft Auto V', 93),
 ('Dota 2', 82), ("Tom Clancy's Rainbow Six® Siege", 59), ('Rocket League®', 55),
 ('Rust', 49), ('Z1 Battle Royale', 48), ('Team Fortress 2', 43), ('PAYDAY 2', 41),
 ("Garry's Mod", 35), ('Paladins®', 33), ('Dead by Daylight', 33), ('Destiny 2', 29),
 ('ARK: Survival Evolved', 28), ('DayZ', 25), ('Fall Guys: Ultimate Knockout', 25),
 ('Among Us', 24), ('Unturned', 24), ('Euro Truck Simulator 2', 22)]
'''

xs = ["CSGO", "PUBG", "GTAV", "DOTA2", "R6", "RL", "TF2", "RUST", "H1", "PD2", "GRY", "PLD", "DBD", "DTNY", "ARK", "DayZ", "FG", "AU", "UNTR", "ETS"]
ys = [428, 132, 93, 82, 59, 55, 49, 48, 43, 41, 35, 33, 33, 29, 28, 25, 25, 24, 24, 22]

plt.title("Most common games in review dataset")
plt.scatter(xs,ys)
plt.xticks([])
plt.ylabel("Frequency")
# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = f"{x}"

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.show()
'''
langs = []
for text in X:
    langs.append(detect_text(text))

freq = {}
for item in langs:
    if (item in freq):
        freq[item] += 1
    else:
        freq[item] = 1

print(OrderedDict(sorted(freq.items(), key = itemgetter(1), reverse = True)))
'''
xs = ["EN", "RU", "DE", "TR", "PT", "ES", "PL", "CN", "UNK", "KO", "SO", "FR", "HU", "RO", "BG"]
ys = [2099, 907, 219, 215, 201, 154, 113, 110, 108, 89, 79, 59, 48, 41, 40]

plt.title("Most common detected languages in review dataset")
plt.scatter(xs,ys)
plt.xticks([])
plt.ylabel("Frequency")
# zip joins x and y coordinates in pairs
for x,y in zip(xs,ys):

    label = f"{x}"

    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.show()

'''


OrderedDict([('en', 2099), ('ru', 907), ('de', 219),
             ('tr', 215), ('pt', 201), ('es', 154),
             ('pl', 113), ('zh-cn', 110), ('unknown', 108),
             ('ko', 89), ('so', 79), ('fr', 59), ('hu', 48),
             ('ro', 41), ('bg', 40), ('tl', 39), ('af', 37),
             ('nl', 33), ('it', 32), ('ja', 30), ('uk', 30),
             ('no', 27), ('sl', 26), ('da', 23), ('ca', 23),
             ('th', 22), ('mk', 22), ('cy', 21), ('et', 20),
             ('cs', 20), ('sk', 20), ('id', 17), ('hr', 13),
             ('fi', 12), ('sq', 11), ('vi', 10), ('sw', 9),
             ('sv', 6), ('lt', 4), ('lv', 4), ('zh-tw', 3),
             ('el', 1), ('fa', 1), ('he', 1), ('ar', 1)])'''
