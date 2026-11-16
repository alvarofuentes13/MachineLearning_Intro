import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

df = pd.read_csv('data/housing_prices_train_data.csv')

#Cross-Validation: cuando separamos los datos entre lote de entrenamiento y de validación, en ocasiones nuestro modelo-
# puede funcionar correctamente para predecir dentro del lote de validación dado, pero puede hacerlo muy mal dado un lote distinto.
# Con cross-validation lo que se pretende es dividir todos los datos en un número de pliegues, y luego separar datos de entrenamiento/validación-
# correspondientes a cada lote, de esta manera podemos realizar pruebas de validación con el conjunto de datos al completo
# Por ejemplo, si hacemos 5 pliegues, en el primer entrenamiento usamos como validación el primer 20%(0-20%),
# en el segundo usamos el segundo 20%(20-40%) y así sucesivamente hasta pasar por todos los pliegues, luego hacemos la media de los 5 y listo

numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
X = df[numerical_cols].drop('SalePrice', axis=1)
y = df['SalePrice']

my_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

# Usamos cross_val_score para predecir el mae respecto a la media de los pligues(cv=5)
# Multiplicamos por -1 ya que el metodo nos devuelve un mae negativo
scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')

print('MAE para 5 pliegues: ', scores) # Se puede observar como cada pliegue obtiene un MAE bastante distinto
print('Media de MAE: ', scores.mean())