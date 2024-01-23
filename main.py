import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv("train_cpy.csv")

df['HomePlanet'] = df['HomePlanet'].fillna(pd.Series(np.where(df['VIP'] == True, 'Europa', 'Earth'), index=df.index))

df['CryoSleep'] = df['CryoSleep'].fillna(
    (df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] != 0.0).any(axis=1))

df = df.dropna(subset=["Age"])
df = df.dropna(subset=["Cabin"])
df = df.dropna(subset=["VIP"])
df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

cols = "PassengerId Name Cabin".split()
df = df.drop(columns=cols)

df['Destination'] = df['Destination'].fillna('Unknown')

df_VIP = df.loc[((df["VIP"] == True) & (df["CryoSleep"] == False))]
df_com = df.loc[((df["VIP"] == False) & (df["CryoSleep"] == False))]

cols = "RoomService FoodCourt ShoppingMall Spa VRDeck".split()
for c in cols:
    mean_com = df_com[c].mean()
    mean_VIP = df_VIP[c].mean()
    df[c] = df.apply(
        lambda row: 0 if row['CryoSleep'] else (mean_com if ~row['VIP'] else mean_VIP) if pd.isnull(row[c]) else row[c],
        axis=1)

decks = sorted(df.Deck.unique())
mapdictdeck = dict()
for i in decks:
    mapdictdeck[i] = decks.index(i)
df.Deck = df.Deck.map(mapdictdeck)

sides = sorted(df.Side.unique())
mapdictside = dict()
for i in sides:
    mapdictside[i] = sides.index(i)
df.Side = df.Side.map(mapdictside)

listVip = [False, True]
mapVip = dict()
for i in listVip:
    mapVip[i] = listVip.index(i)
df.VIP = df.VIP.map(mapVip)

listCryoSleep = [False, True]
mapCryoSleep = dict()
for i in listCryoSleep:
    mapCryoSleep[i] = listCryoSleep.index(i)
df.CryoSleep = df.CryoSleep.map(mapCryoSleep)

listTransported = [False, True]
mapTransported = dict()
for i in listTransported:
    mapTransported[i] = listTransported.index(i)
df.Transported = df.Transported.map(mapTransported)

listHomePlanet = "Earth Europa Mars".split()
mapHomePlanet = dict()
for i in listHomePlanet:
    mapHomePlanet[i] = listHomePlanet.index(i)
df.HomePlanet = df.HomePlanet.map(mapHomePlanet)

listDestination = "TRAPPIST-1e,55 Cancri e,PSO J318.5-22,Unknown".split(",")
mapDestination = dict()
for i in listDestination:
    mapDestination[i] = listDestination.index(i)
df.Destination = df.Destination.map(mapDestination)

df['AllWastes'] = 0.0
listWastes = "RoomService FoodCourt ShoppingMall Spa VRDeck".split()

for index, row in df.iterrows():
    summ = 0
    for waste_type in listWastes:
        summ += row[waste_type]
df.at[index, 'AllWastes'] = summ

df = df.drop(columns=listWastes)

target = "Transported"
y = df[target]
X = df.drop(columns=target).values
depthes = list(range(1,21))
accs=[]
for i in depthes:
    model = DecisionTreeClassifier(max_depth=i)
    model.fit(X, y)
    pred = model.predict(X)
    accs.append((pred == y).mean())
plt.plot(depthes,accs)
# +Планеты кодируются: земля = 0, европа = 1, марс = 2
# +Назначение кодируем: TRAPPIST-1e = 0, Cancri e = 1, J318.5-22 = 2
# +vip cryptosleep, Transported : 0, 1
# +Кабина(палуба и сторона) кодируются так же как города в демке
# Траты объединяем в 1 колонку
