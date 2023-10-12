# -*- coding: utf-8 -*-
"""analyzeFraud.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V7_LhXbf2KDYdrOai1n1Cky6jjwdXYSA
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

"""El parámetro index_col=False ayuda a indicar explicitamente que no se tome la primera columna como un index"""

df = pd.read_csv('/content/drive/MyDrive/fraudDataset/transactions_train.csv', index_col=False)

"""# Datos generales acerca del dataframe

Encontramos a partir del análisis del dataset que este se compone de 6351193 registros con 10 columnas, ninguno de estos registros presenta un NaN o valores nulos, sin embargo, existen gran número de registros con valores en 0.
"""

df.head()

"""Columnas existentes dentro del dataframe"""

df.columns

"""Número de registros y columnas"""

df.shape

"""Tipo de variable de cada columna"""

df.info()

"""Número de NaN o valores nulos dentro del dataset"""

df.isna().sum()

"""**Revisión de valores cero en columnas**

Encontramos que en las columnas oldbalanceOrig, newbalanceOrig, oldbalanceDest y newbalanceDest, existen la siguiente cantidad de 0 como registro.
"""

zerosA = (df['amount'] == 0).sum()

zerosB = (df['oldbalanceOrig'] == 0).sum()

zerosC = (df['newbalanceOrig'] == 0).sum()

zerosD = (df['oldbalanceDest'] == 0).sum()

zerosE = (df['oldbalanceDest'] == 0).sum()

zeroValues = dict(amount = zerosA, oldbalanceOrig = zerosB, newbalanceOrig = zerosC, oldbalanceDest = zerosD, newbalanceDest = zerosE)
zeros = pd.DataFrame(zeroValues, index=['Valores de 0'])

keys=['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
total = pd.DataFrame({key: len(df) for key in keys}, index=['Total de valores'])

values = pd.concat([zeros, total])

values = values.transpose()

values.plot(kind='bar', stacked=True,
        title='Comparativa de total de datos vs Datos cuyo valor es 0.0', cmap='crest')

values

fraud = (df['isFraud'] == 1).sum()
notFraud = (df['isFraud'] == 1).sum()
isFraudValues = []

ax = plt.pie(typeCount, labels=typeCount.index, autopct='%1.0f%%')
plt.show()

"""# Análisis de columna type

La única columna con datos de tipo categórico es 'type' que expresa el tipo de transacción hecha. Encontramos que el tipo CASH_OUT representa un 35% del dataframe, seguido de PAYMENT con 34%, CASH_IN con 22%, TRANSFER con 8% y finalmente DEBIT con 1%.
"""

sns.set_palette(sns.color_palette('viridis'))

ax = sns.countplot(x='type', data=df)
ax.set_title('Distribución de tipos de transacción (type)')
ax.set_xlabel('')
ax.ticklabel_format(style='plain', axis='y')
plt.show()

typeCount = df['type'].value_counts()
print(typeCount)
ax = plt.pie(typeCount, labels=typeCount.index, autopct='%1.0f%%')
plt.show()

"""# Relación entre variables

**Correlación entre variables en un mapa de calor**

Encontramos que existe una baja relación entre columnas, dado el propósito del análisis se puede concluir que la relación más fuerte (0.46) que corresponde a amount y newbalanceDest no tiene un uso práctico, pues naturalmente la cantidad que es parte de la transacción refleja un cambio en el total crediticio de la cuenta destino.
"""

plt.figure(figsize=(8, 6))
ax = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='magma', linewidths=1);