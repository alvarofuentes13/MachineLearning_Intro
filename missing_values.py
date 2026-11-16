import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/housing_prices_train_data.csv')

y = df['SalePrice'].dropna(axis=0)
X = df.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object', 'str'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

def mae_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(prediction, y_valid)

# Cuando tenemos un dataset con valores incompletos, tenemos 3 maneras principales de lidiar con ello
# Borrar Columnas con valores incompletos, reemplazar estos valores(Imputation) y añadir una columna-
# -especificando que ese valor estaba incompleto(Extension to Imputation)

# 1: Borrar columnas con valores incompletos
cols_missing_values = [col for col in X_train.columns
                        if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_missing_values, axis=1)
reduced_X_valid = X_valid.drop(cols_missing_values, axis=1)

print("MAE tras borrar columnas: ", mae_score(reduced_X_train, reduced_X_valid, y_train, y_valid))

# 2: Imputation: reemplazamos el valores que falta por uno por defecto que indiquemos

my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

#Al imputar los valores se borran los nombres de las columnas, debemos volver a asignarlos
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE tras imputar columnas: ", mae_score(imputed_X_train, imputed_X_valid, y_train, y_valid))

# 3: Imputar y añadir columna indicando la imputación
# Creamos una copia de nuestra muestra
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

#Añadimos la columna
for col in cols_missing_values:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

#Al imputar los valores se borran los nombres de las columnas, debemos volver a asignarlos
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE tras imputar columnas e indicar imputación: ", mae_score(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

# Los resultados indican que el 1º Método ha sido el mejor, esto se debe a que en las columnas con valores incompletos,
# estos representan una gran parte de la columna, por tanto al borrar la columna NO descartamos mucha información importante
