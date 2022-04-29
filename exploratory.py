import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import pandas as pd


def main(columns_count, df):
    for cc in columns_count:
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(x=cc, hue="TIPSAL", data=df).set(title=cc, ylabel='Total Pacientes')
        a = [p.get_height() for p in ax[0].axes.patches]
        patch = [p for p in ax[0].axes.patches]
        number_of_categories = int(len(a)/2)
        for i in range(number_of_categories):
            total = a[i] + a[(i + number_of_categories)]
            for j in range(2):
                percentage = '{:.1f}%'.format(100 * a[(j * number_of_categories + i)] / total)
                x = patch[(j * number_of_categories + i)].get_x() + \
                    patch[(j * number_of_categories + i)].get_width() / 3
                y = patch[(j * number_of_categories + i)].get_y() + patch[(j * number_of_categories + i)].get_height()
                if percentage != "nan%":
                    ax[0].axes.annotate(percentage, (x, y), size=12)
        plt.show()
    plt.show()

    plt.figure(figsize=(14, 6))
    sns.countplot(x="RANGOEDAD", hue="TIPSAL", data=df).set(title="RANGOEDAD", ylabel='Total Pacientes')
    plt.show()


if __name__ == '__main__':
    path_to_excel = r'dfs/df_to_explore.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    columns_count = ["ANTERIOR", "TVISITA", "LIBRELEC", "TURNO", "CIRPRES", "ULTESP", "TIPENTR",
                     "SERVICIO", "GR_ETARIO", "SEXO"]
    main(columns_count, dataframe)