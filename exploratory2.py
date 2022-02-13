import matplotlib.pyplot as plt
import concurrent.futures
import seaborn as sns
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from common import *

path_to_excel = r'/Users/zeyna/Documents/TFG/dataframes/df_decision_trees.xlsx'


def main_exploratory(columns_count, df):

    for cc in columns_count:
        sns.countplot(x=cc, hue="TIPSAL", data=df).set(title=cc, ylabel='Total Pacientes')
        plt.show()
    """sns.countplot(x="TIPPRES", hue="TIPSAL", data=df).set(title='TIPPRES', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="CIRPRES", hue="TIPSAL", data=df).set(title='CIRPRES', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="ULTESP", hue="TIPSAL", data=df).set(title='ULTESP', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TIPENTR", hue="TIPSAL", data=df).set(title='TIPENTR', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="LIBRELEC", hue="TIPSAL", data=df).set(title='LIBRELEC', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TVISITA_P", hue="TIPSAL", data=df).set(title='TVISITA_P', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TVISITA_S", hue="TIPSAL", data=df).set(title='TVISITA_S', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SERVICIO_GASB", hue="TIPSAL", data=df).set(title='SERVICIO_GASB', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SERVICIO_GASC", hue="TIPSAL", data=df).set(title='SERVICIO_GASC', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SERVICIO_GASE", hue="TIPSAL", data=df).set(title='SERVICIO_GASE', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SERVICIO_GASH", hue="TIPSAL", data=df).set(title='SERVICIO_GASH', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="ANTERIOR_N", hue="TIPSAL", data=df).set(title='ANTERIOR_N', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="ANTERIOR_S", hue="TIPSAL", data=df).set(title='ANTERIOR_S', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SOSPECHA_N", hue="TIPSAL", data=df).set(title='SOSPECHA_N', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="SOSPECHA_S", hue="TIPSAL", data=df).set(title='SOSPECHA_S', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TURNO_N", hue="TIPSAL", data=df).set(title='TURNO_N', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TURNO_M", hue="TIPSAL", data=df).set(title='TURNO_M', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="TURNO_T", hue="TIPSAL", data=df).set(title='TURNO_T', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREFERENTE_NO", hue="TIPSAL", data=df).set(title='PREFERENTE_NO', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREFERENTE_SI", hue="TIPSAL", data=df).set(title='PREFERENTE_SI', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="GR_ETARIO_IA", hue="TIPSAL", data=df).set(title='GR_ETARIO_IA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="GR_ETARIO_A", hue="TIPSAL", data=df).set(title='GR_ETARIO_A', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_COLONOSCOPIA DIAGNOSTICA", hue="TIPSAL", data=df).set(title='PREST_COLONOSCOPIA DIAGNOSTICA',
                                                                                 ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_CONSULTA PRIMERA", hue="TIPSAL", data=df).set(title='PREST_CONSULTA PRIMERA',
                                                                                 ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_EXCLUSIVO PREVECOLON", hue="TIPSAL", data=df).set(title='PREST_EXCLUSIVO PREVECOLON',
                                                                                 ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_EXCLUSIVO PREVECOLON CON ANESTESISTA", hue="TIPSAL", data=df)\
        .set(title='PREST_EXCLUSIVO PREVECOLON CON ANESTESISTA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_PANENDOSCOPIA ORAL DIAGNOSTICA", hue="TIPSAL", data=df)\
        .set(title='PREST_PANENDOSCOPIA ORAL DIAGNOSTICA', ylabel='Total Pacientes')
    plt.show()

    sns.countplot(x="PREST_RECTOSIGMOIDOSCOPIA DIAGNOSTICA", hue="TIPSAL", data=df)\
        .set(title='PREST_RECTOSIGMOIDOSCOPIA DIAGNOSTICA', ylabel='Total Pacientes')
    plt.show()"""

