import pandas as pd

df = pd.read_csv("red_wines_after_over_sampling.csv")


def percentage(part, whole):
    return (float(part) * float(whole))/100


rnd = int(percentage(15, df['citric acid'].size))
data = df['citric acid'].head(len(df)).sample(rnd)

for el in data.index[0:rnd][0:rnd]:
    df.loc[el, 'citric acid'] = ''

#df.to_csv('data.csv')