# ============================================================
# 🔧 Pacotes e configuração geral
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import shutil
from tempfile import mkdtemp

# Configurações do Pandas e Warnings
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")

# Importações do Scikit-learn
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

print("✅ Etapa 0: Pacotes e configurações carregados com sucesso.")

# ============================================================
# 📂 1. Carregar dados
# ============================================================
try:
    df = pd.read_csv("dados_passagens.csv")
except FileNotFoundError:
    print("\nERRO: O arquivo 'dados_passagens.csv' não foi encontrado. Usando dados de teste.")
    df = pd.DataFrame({
        'airline': ['A', 'B', 'A', 'C', 'B'],
        'source_city': ['SAO', 'RIO', 'SAO', 'FOR', 'RIO'],
        'destination_city': ['REC', 'FOR', 'REC', 'POA', 'FOR'],
        'departure_time': ['M', 'N', 'M', 'T', 'N'],
        'arrival_time': ['T', 'M', 'T', 'N', 'M'],
        'stops': ['zero', 'one', 'zero', 'one', 'one'],
        'class': ['Econ', 'Bus', 'Econ', 'Econ', 'Bus'],
        'duration': [2.5, 3.0, 2.5, 4.0, 3.0],
        'days_left': [30, 10, 5, 20, 15],
        'price': [5000, 12000, 6000, 7500, 11000]
    })

print(f"\nShape inicial: {df.shape}")

# Remoção de colunas irrelevantes
df = df.drop(columns=[c for c in ['Unnamed: 0', 'flight'] if c in df.columns], errors='ignore')
df = df.drop_duplicates().reset_index(drop=True)

print(f"Após limpeza: {df.shape}")

# ============================================================
# 🔍 2. Análise inicial
# ============================================================
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
if 'price' in num_cols:
    num_cols.remove('price')

print(f"Numéricas: {num_cols}")
print(f"Categóricas: {cat_cols}")

# ============================================================
# 🚀 3. Split treino / teste
# ============================================================
target = 'price'
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# ⚙️ 4. Pipelines de pré-processamento
# ============================================================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

print("✅ Pré-processamento configurado.")

# ============================================================
# 🤖 5. Modelos e grids otimizados
# ============================================================
cachedir = mkdtemp()
models = {
    "lasso": Pipeline([("pre", preprocessor),
                       ("model", Lasso(max_iter=50000, random_state=42))],
                       memory=cachedir),
    "rf": Pipeline([("pre", preprocessor),
                    ("model", RandomForestRegressor(n_jobs=-1, random_state=42))],
                    memory=cachedir),
    "gbr": Pipeline([("pre", preprocessor),
                     ("model", GradientBoostingRegressor(random_state=42))],
                     memory=cachedir)
}
param_grid = {
    "lasso": {"model__alpha": [1.0, 5.0, 10.0]},
    "rf": {"model__n_estimators": [50], "model__max_depth": [6, 10]},
    "gbr": {"model__n_estimators": [50], "model__learning_rate": [0.05, 0.1]}
}

# ============================================================
# 🧠 6. Funções utilitárias
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def model_cv_search(name, pipeline, param_grid=None, X=X_train, y=y_train, cv=2, n_jobs=-1):
    start = time.time()
    if param_grid:
        search = RandomizedSearchCV(pipeline, param_distributions=param_grid, n_iter=2, cv=cv,
                                    scoring="neg_mean_squared_error", n_jobs=n_jobs, random_state=42, verbose=0)
    else:
        search = pipeline
    search.fit(X, y)
    best = search.best_estimator_ if hasattr(search, "best_estimator_") else search
    y_pred = best.predict(X)
    cv_rmse = np.sqrt(-cross_val_score(best, X, y, cv=cv,
                                       scoring="neg_mean_squared_error", n_jobs=n_jobs).mean())
    return {
        "name": name,
        "model": best,
        "cv_rmse": cv_rmse,
        "train_rmse": rmse(y, y_pred),
        "r2": r2_score(y, y_pred),
        "best_params": getattr(search, "best_params_", None),
        "time": time.time() - start
    }

# ============================================================
# 🏁 7. Treinamento e comparação
# ============================================================
results = [model_cv_search(k, v, param_grid.get(k)) for k, v in models.items()]
results_df = pd.DataFrame(results).sort_values("cv_rmse")
print("\n--- Resultados ---")
print(results_df[["name", "cv_rmse", "train_rmse", "r2", "time"]])

best_entry = results_df.iloc[0]
best_model = results_df.iloc[0]["model"]
print(f"\n🏆 Melhor modelo: {best_entry['name']} | CV RMSE: {best_entry['cv_rmse']:.2f} | R2: {best_entry['r2']:.3f}")

# ============================================================
# 📊 8. Avaliação final no teste
# ============================================================
best_model.fit(X_train, y_train)
y_pred_test = best_model.predict(X_test)
print("\n📈 Teste:")
print(f"RMSE: {rmse(y_test, y_pred_test):.2f}")
print(f"R2: {r2_score(y_test, y_pred_test):.3f}")

# ============================================================
# 💡 9. Importância das variáveis corrigida
# ============================================================
print("\n🔍 Calculando importância das variáveis...")

# Transforma X_train para o formato pós-preprocessamento
X_train_transformed = best_model.named_steps['pre'].transform(X_train)
feature_names = best_model.named_steps['pre'].get_feature_names_out()

# Permutation importance calculada no modelo puro
perm = permutation_importance(
    best_model.named_steps['model'],
    X_train_transformed,
    y_train,
    n_repeats=1,
    random_state=42,
    n_jobs=-1
)

importances = pd.DataFrame({
    "feature": feature_names,
    "importance": perm["importances_mean"]
}).sort_values("importance", ascending=False).head(15)

print(importances)

plt.figure(figsize=(8, 6))
sns.barplot(data=importances, x="importance", y="feature", palette="coolwarm")
plt.title("Top 15 variáveis mais importantes")
plt.tight_layout()
plt.show()

# =====================================================
# ROI CALCULATION - Predictive Pricing Project
# =====================================================

# Valores hipotéticos de negócio
total_sales = 200_000                   # Total de passagens vendidas
loss_per_sale_before = 5000             # Perda média antes do modelo (USD)
rmse_before = 10000                     # Erro médio sem modelo
rmse_after = 4365                       # Erro médio com modelo
project_cost = 5_000_000                # Custo total do projeto

# Estimativa de perdas antes e depois
loss_before = total_sales * loss_per_sale_before
loss_after = loss_before * (rmse_after / rmse_before)
savings = loss_before - loss_after

# Cálculo do ROI
roi = ((savings - project_cost) / project_cost) * 100

print(f"Perda antes do modelo: ${loss_before:,.0f}")
print(f"Perda após o modelo:  ${loss_after:,.0f}")
print(f"Economia anual estimada: ${savings:,.0f}")
print(f"ROI estimado: {roi:,.2f}%")

# ============================================================
# 💾 10. Salvamento do modelo vencedor em arquivo Pickle
# ============================================================
import pickle
from datetime import datetime
from pathlib import Path

# Define o caminho local (mesma pasta do notebook)
notebook_dir = Path.cwd()  # diretório atual
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Cria o nome do arquivo
model_filename = notebook_dir / f"melhor_modelo_{best_entry['name']}_{timestamp}.pkl"

# Salva o modelo com pickle
with open(model_filename, "wb") as file:
    pickle.dump(best_model, file)

print(f"💾 Modelo salvo com sucesso em: {model_filename}")

# ============================================================
# 🧹 11. Limpeza
# ============================================================
shutil.rmtree(cachedir, ignore_errors=True)
print("\n✅ Execução concluída com sucesso e cache limpo.")

