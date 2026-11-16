import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv('./data/housing_prices_train_data.csv')

y = df['SalePrice'].dropna(axis=0)
X = df.drop(['SalePrice'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

def mae_score(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    return mean_absolute_error(prediction, y_valid)

# Valores en formato string como categorías normalmente dan error al intentar trabajar con ellos con modelos de predicción
# Hay varías maneras de lidiar con esto, borrar la columna correspondiente, asignar valor a un entero(Ordinal Encoding) y
# Crear una 'tabla' con los valores y marcar la presencia o no presencia de estos(One-hot Encoding)

# Antes de empezar, vamos a encontrar aquellas columnas con valores de tipo categórico

object_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
print(object_cols)

# 1: borrar columnas, borramos las columnas donde los valores son tipo string u objeto

reduced_X_train = X_train.select_dtypes(exclude=['object', 'str'])
reduced_X_valid = X_valid.select_dtypes(exclude=['object', 'str'])

print("MAE tras borrar columnas con valores categoricos: ", mae_score(reduced_X_train, reduced_X_valid, y_train, y_valid))

# 2: Ordinal Encoding: creamos una nueva columna
# Al dividir los datos en entrenamiento/validación, es posible que ciertos valores no aparezcan en el entrenamiento y si en la validación
# Por ello debemos encontrar aquellos campos que se repitan en ambas colecciones y trabajar solo con ellos

# Columnas donde TODAS las categorías de validación están en entrenamiento
good_label_cols = [col for col in object_cols
                   if set(X_valid[col]).issubset(set(X_train[col]))]

# El resto las eliminamos
bad_label_cols = list(set(object_cols) - set(good_label_cols))

label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

ordinal_encoder = OrdinalEncoder()
label_X_train[good_label_cols] = ordinal_encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = ordinal_encoder.transform(X_valid[good_label_cols])

print("MAE tras borrar columnas con valores categoricos: ", mae_score(label_X_train, label_X_valid, y_train, y_valid))

# 3: One-hot Encoding: vamos a codificar las columnas con valores categoricos y luego borraremos la columna base
# Asi nos quedamos solo con columnas tipo numérico

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

#OH Encoding remueve el index, hay que poner de vuelta
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

numeric_X_train = X_train.drop(object_cols, axis=1)
numeric_X_valid = X_valid.drop(object_cols, axis=1)

OH_X_train = pd.concat([numeric_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([numeric_X_valid, OH_cols_valid], axis=1)

OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE tras borrar columnas con valores categoricos: ", mae_score(OH_X_train, OH_X_valid, y_train, y_valid))
