import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/housing_prices_train_data.csv')

numerical_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]
X = df[numerical_cols].drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# Gradient Boosting se refiere a la predicción mediante el cálculo del gradiente a través de la función de coste
# Se calcula una prediccion, se calcula la función de coste, que se usará para confeccionar el nuevo modelo y volver a hacer predicciones,
# asi hasta que la función de coste se aproxime a 0

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

prediction_1 = my_model.predict(X_valid)
print('MAE: ', mean_absolute_error(prediction_1, y_valid))

#Podemos indicar cuantas veces recorrer el bucle (n_estimators), asi como la tasa de aprendizaje para la función de calculo de gradiente(learning_rate)

my_model = XGBRegressor(n_estimators=300, learning_rate=0.05)
my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

prediction_2 = my_model.predict(X_valid)
print('MAE: ', mean_absolute_error(prediction_2, y_valid))