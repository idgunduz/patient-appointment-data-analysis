import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from common import *
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
mpl.rcParams.update(mpl.rcParamsDefault)


def neural_networks(df):
    target = df["TIPSAL"]
    df = df.drop(["TIPSAL"], axis=1)
    x, y = df.values, target.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    undersample = NearMiss(version=2, n_neighbors=3)
    x_under, y_under = undersample.fit_resample(x_scaled, y)
    x_train, x_test, y_train, y_test = train_test_split(x_under, y_under, test_size=0.2,
                                                        stratify=y_under, random_state=0)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=38))
    #model.add(Dense(512, activation='relu', input_dim=40)) Cuando ejecutamos el df1
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100)
    sns.set()
    acc = hist.history['accuracy']
    val = hist.history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, '-', label='Training accuracy')
    plt.plot(epochs, val, ':', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.plot()
    y_pred = model.predict(x_test)
    preds = np.round(y_pred, 0)
    print(confusion_matrix(y_test, preds))
    print(metrics.classification_report(y_test, preds))


if __name__ == '__main__':
    path_to_excel = r'dfs/df2.xlsx'
    dataframe = pd.read_excel(path_to_excel)
    neural_networks(dataframe)