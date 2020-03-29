if __name__ == '__main__':
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    red_wines = pd.read_csv('winequality-red.csv')
    red_wines.head()

    X = red_wines.iloc[:, :-1]
    Y = red_wines.iloc[:, -1:]

    for column in X:
        sns.boxplot(x='quality', y=column, data=red_wines)
        plt.show()
