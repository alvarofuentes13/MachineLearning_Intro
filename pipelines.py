import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv('./data/housing_prices_train_data.csv')

y = df['SalePrice'].dropna(axis=0)
X = df.drop(['SalePrice'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
                        X_train[cname].dtype in ['object', 'str']]

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = X_train[my_cols].copy()
X_valid = X_valid[my_cols].copy()

#Pipelines: son una manera simple de organizar el codigo para el preprocesamiento de datos y el modelado en conjunto
# Conseguimos codigo más limpio, menos bugs y más facilidad para sacar a producción en terminos de escalabilidad

# 1er paso: definir los pasos de preprocesado
# Usamos ColumnTransformer para unir distintos pasos de preprocesado
# En este ejemplo uniremos dos pasos; uno para imputar valores en datos numericos incompletos y otro para imputar y aplicacar-
# one-hot encoding a datos categóricos

numerical_transformer = SimpleImputer(strategy='constant')

#Instanciamos el Pipeline, y definimos cada paso interno con (clave, valor)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Cada paso dentro del transformador será (clave, valor, columnas a las que afecta)
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ]
)

# 2do paso: definimos el model a usar

model = RandomForestRegressor(n_estimators=100, random_state=0)

# 3er paso: creamos y evaluamos el Pipeline final
# Este Pipeline incluira tanto nuestro preprocesado como el modelo definido

my_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])

my_pipeline.fit(X_train, y_train)
prediction = my_pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, prediction)
print('MAE: ', mae)

# Gracias al Pipeline, tanto los datos de entrenamiento como los de las predicciones son preprocesados de manera automática-
# sin necesidad de manejar cada lote por separado

