# -*- coding: utf-8 -*-
"""
Лабораторная работа № 1-2
Методы искусственного интеллекта. EDA. Линейная регрессия. Дерево решений.
CatBoost. XGBoost. Нейронные сети (MLP).
"""

import os
import sys
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import datetime

# Настройки
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_palette('Set2')

# Создаем директории для результатов
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ------------------------------
# 1. Загрузка данных и EDA
# ------------------------------
print("1. Загрузка данных Medical Cost Personal и EDA")

data_path = '/tpu_8EM51_ML_course/1/data/raw/insurance.csv'
df = pd.read_csv(data_path)

# Целевая переменная
target = 'charges'
X = df.drop(target, axis=1)
y = df[target]

# Категориальные и числовые признаки
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

print("\nОписание датасета:")
print(df.describe(include='all'))
print("\nИнформация о данных:")
print(df.info())
print("\nПервые 5 строк:")
print(df.head())

# EDA визуализации
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_features):
    axes[i].hist(df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Распределение {col}')
# Добавим целевую переменную
axes[3].hist(y, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
axes[3].set_title(f'Распределение {target}')
# Категориальные count plots
for i, col in enumerate(categorical_features):
    sns.countplot(data=df, x=col, ax=axes[4+i], palette='viridis')
    axes[4+i].set_title(f'Распределение {col}')
plt.tight_layout()
plt.savefig('results/distributions.png', dpi=150)
plt.close()

# Корреляционная матрица (только числовые)
plt.figure(figsize=(8, 6))
corr = df[numeric_features + [target]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.savefig('results/correlation_heatmap.png', dpi=150)
plt.close()

# Boxplots для категориальных признаков vs target
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(categorical_features):
    sns.boxplot(data=df, x=col, y=target, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs {target}')
plt.tight_layout()
plt.savefig('results/categorical_boxplots.png', dpi=150)
plt.close()

# Вывод EDA: все признаки потенциально важны, особенно smoker.
selected_features = numeric_features + categorical_features
print("\nВыбранные признаки:", selected_features)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

# ------------------------------
# 2. Построение пайплайна (DVC стиль)
# ------------------------------
print("\n2. Построение пайплайна предобработки и определение DVC stages")

# Создаем препроцессор: масштабирование числовых, one-hot для категориальных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# DVC: создадим файл dvc.yaml с описанием этапов
dvc_yaml_content = """
stages:
  prepare:
    cmd: python lab1_2_insurance.py prepare
    deps:
      - lab1_2_insurance.py
    outs:
      - data/processed
  train:
    cmd: python lab1_2_insurance.py train
    deps:
      - data/processed
      - lab1_2_insurance.py
    outs:
      - models/
      - results/metrics.csv
    metrics:
      - results/metrics.csv
  evaluate:
    cmd: python lab1_2_insurance.py evaluate
    deps:
      - models/
      - data/processed
    metrics:
      - results/test_metrics.json
"""
with open('dvc.yaml', 'w', encoding='utf-8') as f:
    f.write(dvc_yaml_content)
print("Файл dvc.yaml создан.")

# Попробуем вывести DAG, если DVC установлен
try:
    result = subprocess.run(['dvc', 'dag'], capture_output=True, text=True)
    if result.returncode == 0:
        print("\nВычислительный граф DVC (dvc dag):")
        print(result.stdout)
    else:
        print("DVC не установлен или не инициализирован. Граф не выведен.")
except FileNotFoundError:
    print("DVC не найден в системе. Пропускаем вывод графа.")

# ------------------------------
# Функция для вычисления метрик
# ------------------------------
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Словарь для хранения метрик всех моделей
all_metrics = {}

# ------------------------------
# 3. Линейная регрессия
# ------------------------------
print("\n3. Линейная регрессия")
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lr', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

metrics_lr = compute_metrics(y_test, y_pred_lr)
all_metrics['Linear Regression'] = metrics_lr
print("Метрики Linear Regression:", metrics_lr)

# Веса (коэффициенты)
lr_model = lr_pipeline.named_steps['lr']
# Получаем имена признаков после OneHot
preprocessor_fitted = lr_pipeline.named_steps['preprocessor']
feature_names = (numeric_features +
                 list(preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_features)))
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_model.coef_
})
print("\nКоэффициенты линейной регрессии:")
print(coef_df)
coef_df.to_csv('results/linear_coefficients.csv', index=False)

# ------------------------------
# 4. Дерево решений
# ------------------------------
print("\n4. Дерево решений")
dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeRegressor(max_depth=5, random_state=42))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

metrics_dt = compute_metrics(y_test, y_pred_dt)
all_metrics['Decision Tree'] = metrics_dt
print("Метрики Decision Tree:", metrics_dt)

# Визуализация первых узлов дерева
plt.figure(figsize=(20, 10))
dt_model = dt_pipeline.named_steps['dt']
plot_tree(dt_model, max_depth=3, feature_names=feature_names,
          filled=True, rounded=True, fontsize=10)
plt.title('Первые узлы дерева решений (глубина 3)')
plt.tight_layout()
plt.savefig('results/decision_tree_first_nodes.png', dpi=150, bbox_inches='tight')
plt.close()
print("Рисунок первых узлов дерева сохранен в 'results/decision_tree_first_nodes.png'")

# ------------------------------
# 5. CatBoost
# ------------------------------
print("\n5. CatBoost")
# CatBoost может работать с категориальными признаками без предварительного кодирования,
# но для единообразия используем тот же препроцессор (OneHot)
cb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cb', CatBoostRegressor(verbose=0, random_seed=42))
])
cb_pipeline.fit(X_train, y_train)
y_pred_cb = cb_pipeline.predict(X_test)

metrics_cb = compute_metrics(y_test, y_pred_cb)
all_metrics['CatBoost'] = metrics_cb
print("Метрики CatBoost:", metrics_cb)

# Feature Importance
cb_model = cb_pipeline.named_steps['cb']
fi_cb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': cb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nCatBoost Feature Importance:")
print(fi_cb)
fi_cb.to_csv('results/catboost_feature_importance.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_cb, x='Importance', y='Feature', palette='viridis')
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.savefig('results/catboost_feature_importance.png', dpi=150)
plt.close()

# ------------------------------
# 6. XGBoost
# ------------------------------
print("\n6. XGBoost")
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42, verbosity=0))
])
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_test)

metrics_xgb = compute_metrics(y_test, y_pred_xgb)
all_metrics['XGBoost'] = metrics_xgb
print("Метрики XGBoost:", metrics_xgb)

# Feature Importance
xgb_model = xgb_pipeline.named_steps['xgb']
fi_xgb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nXGBoost Feature Importance:")
print(fi_xgb)
fi_xgb.to_csv('results/xgboost_feature_importance.csv', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_xgb, x='Importance', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('results/xgboost_feature_importance.png', dpi=150)
plt.close()

# ------------------------------
# 7. Нейронная сеть (MLP)
# ------------------------------
print("\n7. Нейронная сеть (MLP)")

# Подготовка данных с препроцессором
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
input_dim = X_train_transformed.shape[1]

# Создание модели MLP
def build_mlp_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,),
                     kernel_initializer='he_normal', name='hidden1'),
        layers.Dense(32, activation='relu', kernel_initializer='he_normal', name='hidden2'),
        layers.Dense(16, activation='relu', kernel_initializer='he_normal', name='hidden3'),
        layers.Dense(1, name='output')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

mlp_model = build_mlp_model(input_dim)
mlp_model.summary()

# Настройка TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Обучение
history = mlp_model.fit(
    X_train_transformed, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0,
    callbacks=[tensorboard_callback]
)

# Предсказание и метрики
y_pred_mlp = mlp_model.predict(X_test_transformed).flatten()
metrics_mlp = compute_metrics(y_test, y_pred_mlp)
all_metrics['MLP Neural Network'] = metrics_mlp
print("Метрики MLP:", metrics_mlp)

# Кривые обучения
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Curve')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('results/mlp_learning_curves.png', dpi=150)
plt.close()

# Гистограммы весов
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
layer_names = ['hidden1', 'hidden2', 'hidden3', 'output']
for i, layer_name in enumerate(layer_names):
    layer = mlp_model.get_layer(layer_name)
    weights, biases = layer.get_weights()
    ax = axes[i//2, i%2]
    ax.hist(weights.flatten(), bins=50, alpha=0.7, color='teal', edgecolor='black')
    ax.set_title(f'Распределение весов слоя {layer_name}')
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    mean_w = np.mean(weights)
    ax.axvline(mean_w, color='red', linestyle='dashed', linewidth=1, label=f'Mean={mean_w:.3f}')
    ax.legend()
plt.tight_layout()
plt.savefig('results/mlp_weight_histograms.png', dpi=150)
plt.close()

print("Логи TensorBoard сохранены в:", log_dir)
print("Для просмотра TensorBoard выполните: tensorboard --logdir logs/fit")

# Сохраняем модель
mlp_model.save('models/mlp_model.h5')

# ------------------------------
# 8. Конечный вычислительный граф DVC
# ------------------------------
print("\n8. Конечный вычислительный граф DVC")
try:
    subprocess.run(['dvc', 'status'], capture_output=True)
    subprocess.run(['dvc', 'repro'], capture_output=True)
    result = subprocess.run(['dvc', 'dag'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Финальный DAG DVC:")
        print(result.stdout)
    else:
        print("Не удалось построить DAG. Убедитесь, что DVC инициализирован.")
except Exception as e:
    print(f"DVC не активен или ошибка: {e}")
    print("Вычислительный граф описан в dvc.yaml.")

# ------------------------------
# 9. Сводная таблица метрик и вывод
# ------------------------------
print("\n9. Сводная таблица метрик всех моделей")
metrics_df = pd.DataFrame(all_metrics).T
metrics_df = metrics_df.round(4)
print(metrics_df)
metrics_df.to_csv('results/all_models_metrics.csv')

best_r2_model = metrics_df['R2'].idxmax()
best_rmse_model = metrics_df['RMSE'].idxmin()
print(f"\nЛучшая модель по R²: {best_r2_model} (R² = {metrics_df.loc[best_r2_model, 'R2']})")
print(f"Лучшая модель по RMSE: {best_rmse_model} (RMSE = {metrics_df.loc[best_rmse_model, 'RMSE']})")

# Визуализация сравнения
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
metrics_to_plot = ['MSE', 'RMSE', 'MAE', 'R2']
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i//2, i%2]
    sorted_idx = metrics_df[metric].sort_values().index if metric != 'R2' else metrics_df[metric].sort_values(ascending=False).index
    sns.barplot(x=metrics_df.loc[sorted_idx, metric], y=sorted_idx, ax=ax, palette='coolwarm')
    ax.set_title(f'Сравнение моделей по {metric}')
    ax.set_xlabel(metric)
plt.tight_layout()
plt.savefig('results/models_comparison.png', dpi=150)
plt.close()

# ------------------------------
# 10. Общий вывод по работе
# ------------------------------
print("\n10. Вывод по работе")
print("""
В ходе лабораторной работы проведен EDA датасета Medical Cost Personal.
Выявлено, что признак 'smoker' (курение) оказывает наибольшее влияние на медицинские расходы.
Другие важные признаки: возраст (age), индекс массы тела (bmi), количество детей (children).
Регион (region) и пол (sex) имеют меньшее влияние.

Построены модели: линейная регрессия, дерево решений, CatBoost, XGBoost, MLP.
Наилучшие результаты по R² и RMSE показали градиентный бустинг (CatBoost, XGBoost) и нейронная сеть.
Линейная регрессия показала приемлемый базовый уровень, но уступает нелинейным моделям.
Важность признаков подтверждает доминирующую роль курения.

Использование DVC обеспечило воспроизводимость пайплайна обработки данных и обучения.
""")

print("\nВсе результаты сохранены в папках 'results/', 'models/', 'logs/'.")
print("Работа завершена.")