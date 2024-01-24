import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

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

def k_fold(X, y, k):
    N = X.shape[0]
    n = N // k
    x_list = []
    y_list = []
    for i in range(k):
        lo = i * n
        hi = (i + 1) * n

        if i == k - 1:
            hi = N

        x_list.append(X[lo:hi])
        y_list.append(y[lo:hi])

    for i in range(k):
        xl_copy = x_list.copy()
        yl_copy = y_list.copy()
        x_val = xl_copy.pop(i)
        y_val = yl_copy.pop(i)
        x_train = np.vstack(xl_copy)
        y_train = np.hstack(yl_copy)

        yield (x_train, y_train), (x_val, y_val)
    return

depthes = list(range(1,50))
d_accs = []
for d in depthes:
    acclist = []
    for (x_train, y_train), (x_val, y_val) in k_fold(X, y, 10):
        model = DecisionTreeClassifier(max_depth = d) # оптимальная глубина = 7
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        acc = np.mean(pred == y_val)
        acclist.append(acc)
    d_accs.append(np.mean(acclist))

plt.plot(depthes,d_accs)
plt.show()

models = dict(tree=DecisionTreeClassifier(max_depth=7),
                forest=RandomForestClassifier(),
                booster=GradientBoostingClassifier())
for name, model in models.items():
    accs = []
    for (x_train, y_train), (x_val, y_val) in k_fold(X, y, 10):
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        accs.append(np.mean(pred == y_val) * 100)
    print(f'Для модели {name} среднняя accuracy = {np.mean(accs):.2f}%')

# +Планеты кодируются: земля = 0, европа = 1, марс = 2
# +Назначение кодируем: TRAPPIST-1e = 0, Cancri e = 1, J318.5-22 = 2
# +vip cryptosleep, Transported : 0, 1
# +Кабина(палуба и сторона) кодируются так же как города в демке
# Траты объединяем в 1 колонку
