# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:21:26 2021

@author: simon
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
#leemos el csv
df = pd.read_csv('C:/Users/simon/Documents/Projects/Marketing/marketing_data.csv')

#%%
df['Age'] = 2020 - df['Year_Birth']
#%%
#Exploración de datos
print(df.info())

#%%
#Eliminamos espacios en el nombre de las columnas
df.columns = df.columns.str.replace(' ', '')
#%%
#Transformamos la columna 'Income' (ingresos) a numerica

df['Income'] = df['Income'].str.replace('$', '')
df['Income'] = df['Income'].str.replace(',','')
df['Income'] = df['Income'].astype('float')
#%%
#Podemos reemplazar los valores nulos de los clientes ya sea por el promedio o la mediana
#para conocer por cual de ellos es mejor reemplazar los valores nulos graficamos los valores de 
#ingresos
print(df.isnull().sum())

#%%
#mostramos la distribución de ingresos y es posible observar que existen algunos ingresos muy altos
#comparados con los demas
sns.distplot(df['Income'], kde = False, hist =True)

#%%
#para comprobar lo anterior utilizamos un boxplot y se observan algunos outliers, por lo que la 
#mejor opción de reemplazo de valores nulos en el income es por la mediana, ya que el promedio de
#los ingresos puede estar afectado por los outliers, mientras que la mediana no.
plt.figure(figsize=(3,5))
sns.boxplot(x=df['Income'], orient = 'v')

#%%
print(df.Income.median(), df.Income.mean())
#%%
df.fillna({'Income':df['Income'].median()}, inplace = True)
#%%
print(df.info())
#%%
plots = df.drop(columns = ['ID', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                           'AcceptedCmp5', 'Response', 'Complain']).select_dtypes(include = np.number)
#%%
plots.plot(subplots = True, layout = (4,4), kind = 'box',figsize =(12,12), patch_artist = True)
#Estas graficas nos mostraron algunos datos irregulares en la columna de edad (1900) lo cual 
#Puede deberse a un error en la captura de datos, por lo tanto se eliminan esos outliers
#%%
df_clean = df[df['Year_Birth']>1900]

#%%
df_clean.Year_Birth.plot(kind = 'box', patch_artist = True)
#%%
#transformaremos dt_costume a datetime (contiene fechas)
df_clean['Dt_Customer'] = pd.to_datetime(df_clean['Dt_Customer'])

#%%
df_clean.info()

#%%
#número total de dependientes
df_clean['Dependents'] = df_clean.Kidhome + df_clean.Teenhome

#%%
print(df_clean[df_clean.columns[df_clean.columns.str.contains('Mnt')]]) 
#%%
# sumamos todos los gastos y creamos una nueva columna llamada Mnt que hace referencia a 'Amount'
df_clean['Mnt'] = df_clean[df_clean.columns[df_clean.columns.str.contains('Mnt')]].sum(axis = 1)

#%%
# sumamos todas las compras realizadas y creamos nueva columna 'Purchase'
df_clean['Purchase'] = df_clean[df_clean.columns[df_clean.columns.str.contains('Purchase')]].sum(axis = 1)

#%%
print(df_clean[df_clean.columns[df_clean.columns.str.contains('Cmp')]])
#%%
#eliminamos columna llamada cmp, recordar el inplace, de otro modo no elimina la columna
df_clean.drop(columns = ['Cmp'], inplace = True)
#%%
#sumamos las campañas exitosas o aceptadas por el cliente incluyendo Response que fue la ultima campaña
df_clean['Cmp'] = df_clean[df_clean.columns[df_clean.columns.str.contains('Cmp')]].sum(axis = 1) + df_clean['Response']

#%%
# Edad del cliente
df_clean['Age'] = 2020 - df_clean.Year_Birth

#%%
# Año en el que se convirtieron en clientes
df_clean['Year_Customer'] = pd.DatetimeIndex(df_clean['Dt_Customer']).year

#%%
new_df = df_clean.drop(columns = ['ID', 'Year_Birth', 'Kidhome', 'Teenhome', 'MntWines', 
                                  'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                                  'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                                  'NumCatalogPurchases', 'NumStorePurchases', 'AcceptedCmp1',
                                  'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
                                  'Response', 'Dt_Customer'])
#%%
#Matriz de correlacion
corrs = new_df.corr('pearson')

#%%
#Graficamos matriz de correlaciones
sns.heatmap(corrs, cmap = 'coolwarm')

#%%
#Grafica que muestra que entre mayor sea el ingreso del cliente, mayor es el gasto realizado
sns.lmplot('Income', 'Mnt', data = new_df[new_df['Income']<300000])

#%%
sns.boxplot('Dependents', 'Mnt', data = new_df)
#%%
#Numero de campañas aceptadas aumenta a medida que el 
sns.boxplot('Cmp', 'Income', data = new_df[new_df['Income']<300000], orient = 'v')

#%%
new_df['Teenhome'] = df['Teenhome']
new_df['Kidhome'] = df['Kidhome']
#%%
print(new_df.Education.value_counts()[0:])
#%%
new_df.Education.value_counts().plot(kind = 'bar')

#%%
sns.barplot(x = 'Education', y = 'Mnt', data = new_df)

#%%
sns.barplot(x = 'Education', y = 'Income', data = new_df)

#%%
new_df['Country'].value_counts().plot(kind = 'bar')
#%%
#Este grafico nos muestra que no existe una diferencia clara entre los ingresos de distintos niveles
#educativos ni de edad
sns.scatterplot(x = 'Age', y ='Income', hue = "Education", data = new_df[new_df['Income']<200000])
#%%
# Dinero gastado de acuerdo a su estado civil (casdo, divorciado, etc)
sns.barplot(x = 'Marital_Status', y = 'Mnt', data = new_df)
#%%
new_df.drop(new_df[new_df['Marital_Status'] == 'Alone'].index, inplace = True)
#%%
new_df.drop(new_df[new_df['Marital_Status'] == 'Absurd'].index, inplace = True)
new_df.drop(new_df[new_df['Marital_Status'] == 'YOLO'].index, inplace = True)
#%%
# Compras realizadas de acuerdo a su estado civil
sns.barplot(x = 'Marital_Status', y = 'Purchase', data = new_df)

#%%
new_df.groupby('Country')['Purchase'].sum().plot(kind = 'bar')

#%%
#Aquí podemos observar que a medida que las campañas publicitarias son aceptadas, el gasto promedio
#del cliente es mayor 
new_df.groupby('Cmp')['Mnt'].mean().plot(kind = 'bar')

#%%
df_clean.drop(columns='Cmp', inplace = True)

#%%
#Creamos la variable para determinar la tasa de aceptación de campaña
succ_rate = df_clean[df_clean.columns[df_clean.columns.str.contains('Cmp')]].sum()
#%%
#grafica sobre tasa de exito de campaña
sns.barplot(x = succ_rate.index , y = (succ_rate/len(new_df.Mnt))*100, palette='Blues')
plt.xticks(rotation = 20)
plt.ylabel('% Success rate')

#%% 
print(new_df.Marital_Status.value_counts())

#%%
#Grafico de la tasa de exito de cada campaña
sns.set_palette('Blues_r')
plt.pie(new_df.Marital_Status.value_counts(), labels = new_df.Marital_Status.value_counts().index,
        explode = (0.1,0,0,0,0), shadow = True, autopct='%1.1f')

#%%
#Creamos un nuevo df que contenga las distintas formas de compras
purchases = df_clean[df_clean.columns[df_clean.columns.str.contains('Purchases')]]

#%%
# añadimos las regiones para estimar que tipo de compras dominan en cada región
purchases['Country'] = df_clean.Country

#%%
#gráfico de los distintos tipos de compras
sns.barplot(x = ['Deals', 'Web', 'Catalog', 'Store'],
            y = purchases[purchases.columns[purchases.columns.str.contains('Purchases')]].sum())
plt.ylabel('No. of Purchases')

#%%
#gráfico de los distintos tipos de compras por región
sns.set_palette(palette = 'Blues')
purchases.groupby('Country').sum().plot(kind = 'bar')
plt.legend(['Deals', 'Web', 'Catalog', 'Store'])

#podemos observar que en cada una de las regiones observadas la forma de compra que encabeza la lista
#es por medio de la tienda, le sigue a través de la web, seguido del catalogo y finalmente por deals
#%%





