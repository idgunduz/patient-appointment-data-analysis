import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pandas as pd

path_to_excel = r'/Users/zeyna/Documents/TFG/dataframes/df_decision_trees.xlsx'


def without_hue(plot, feature):
    total = len(feature)
    for p in plot.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height()
        plot.annotate(percentage, (x, y), size=12)
    plt.show()


def main_exploratory(columns_count, df):
    for cc in columns_count:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=cc, hue="TIPSAL", data=df).set(title=cc, ylabel='Total Pacientes')
        plt.show()
    """for cc in columns_count:
        fig, ax = sns.countplot(x=cc, hue="TIPSAL", data=df).set(title=cc, ylabel='Total Pacientes')
        for s in ax:
            without_hue(s, df[cc])
        plt.show()"""


if __name__ == '__main__':
    path_to_excel = r'dfs/df_to_explore.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    columns_count = ["ANTERIOR", "TVISITA", "LIBRELEC", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                     "SERVICIO", "GR_ETARIO", "SEXO", "RANGOEDAD"]
    main_exploratory(columns_count, dataframe)