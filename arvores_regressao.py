#%% Instalar os pacotes necessários

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
!pip install xgboost

#%% Importar os pacotes

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
#from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#%% Importar o banco de dados

dados = pd.read_excel('dados_admissao.xlsx')
# Fonte: adaptado de https://www.kaggle.com/datasets/mohansacharya/graduate-admissions
# Mohan S Acharya, Asfia Armaan, Aneeta S Antony: A Comparison of Regression Models for Prediction of Graduate Admissions, IEEE International Conference on Computational Intelligence in Data Science 2019

#%% Limpeza dos dados

# Remover colunas que não serão utilizadas
dados.drop(columns=['Serial No.'], inplace=True)

#%% Estatísticas descritivas das variáveis

# Variáveis métricas
print(dados[['GRE', 'TOEFL', 'SOP', 'LOR', 'CGPA', 'Score']].describe())

# Variáveis categóricas
print(dados['UniversityRating'].value_counts())
print(dados['Research'].value_counts())

#%% Transformando variáveis explicativas categóricas em dummies

dados = pd.get_dummies(dados, 
                       columns=['UniversityRating'], 
                       drop_first=False,
                       dtype='int')

# Note que Research já é uma dummy!

#%% Separando as variáveis Y e X

X = dados.drop(columns=['Score'])
y = dados['Score']

#%% Separando as amostras de treino e teste

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=100)

#%%######################### Árvore de Decisão ################################
###############################################################################
#%% Gerando a árvore de decisão

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=100)
tree_reg.fit(X_train, y_train)

#%% Plotando a árvore

plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_reg,
          feature_names=X.columns.tolist(),
          filled=True,
          node_ids=True,
          precision=2)
plt.show()

#%% Analisando os resultados dos splits

tree_split = pd.DataFrame(tree_reg.cost_complexity_pruning_path(X_train, y_train))
tree_split.sort_index(ascending=False, inplace=True)

print(tree_split)

#%% Importância das variáveis preditoras

tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':tree_reg.feature_importances_})

print(tree_features)

#%% Obtendo os valores preditos pelo modelo

# Base de treinamento
tree_pred_train = tree_reg.predict(X_train)

# Base de teste
tree_pred_test = tree_reg.predict(X_test)

#%% Avaliando o modelo (base de treino)

mse_train_tree = mean_squared_error(y_train, tree_pred_train)
mae_train_tree = mean_absolute_error(y_train, tree_pred_train)
r2_train_tree = r2_score(y_train, tree_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_tree:.1f}")
print(f"RMSE: {np.sqrt(mse_train_tree):.1f}")
print(f"MAE: {mae_train_tree:.1f}")
print(f"R²: {r2_train_tree:.1%}")

#%% Avaliando o modelo (base de testes)

mse_test_tree = mean_squared_error(y_test, tree_pred_test)
mae_test_tree = mean_absolute_error(y_test, tree_pred_test)
r2_test_tree = r2_score(y_test, tree_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_tree:.1f}")
print(f"RMSE: {np.sqrt(mse_test_tree):.1f}")
print(f"MAE: {mae_test_tree:.1f}")
print(f"R²: {r2_test_tree:.1%}")

#%% Alguns hiperparâmetros do modelo

# max_depth: profundidade máxima da árvore
# min_samples_split: qtde mínima de observações para dividir o nó
# min_samples_leaf: qtde mínima de observações para ser nó folha

# Vamos aplicar um Grid Search
param_grid_tree = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 10]
}

# Identificar o algoritmo em uso
tree_grid = DecisionTreeRegressor(random_state=100)

# Treinar os modelos para o grid search
tree_grid_model = GridSearchCV(estimator = tree_grid, 
                               param_grid = param_grid_tree,
                               scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                               cv=5, verbose=2)

tree_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
tree_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
tree_best = tree_grid_model.best_estimator_

# Predict do modelo
tree_grid_pred_train = tree_best.predict(X_train)
tree_grid_pred_test = tree_best.predict(X_test)

#%% Plotando a árvore após o grid search

plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_best,
          feature_names=X.columns.tolist(),
          filled=True,
          node_ids=True)
plt.show()

#%% Avaliando o novo modelo (base de treino)

mse_train_tree_grid = mean_squared_error(y_train, tree_grid_pred_train)
mae_train_tree_grid = mean_absolute_error(y_train, tree_grid_pred_train)
r2_train_tree_grid = r2_score(y_train, tree_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_tree_grid):.1f}")
print(f"MAE: {mae_train_tree_grid:.1f}")
print(f"R²: {r2_train_tree_grid:.1%}")

#%% Avaliando o novo modelo (base de teste)

mse_test_tree_grid = mean_squared_error(y_test, tree_grid_pred_test)
mae_test_tree_grid = mean_absolute_error(y_test, tree_grid_pred_test)
r2_test_tree_grid = r2_score(y_test, tree_grid_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_tree_grid):.1f}")
print(f"MAE: {mae_test_tree_grid:.1f}")
print(f"R²: {r2_test_tree_grid:.1%}")

#%% Importância das variáveis preditoras

tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':tree_best.feature_importances_}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(tree_features)

#%% Gráfico fitted values

# Valores preditos pelo modelo para as observações da amostra de teste
graph = pd.DataFrame({'Score': y_test,
                      'pred_tree': tree_grid_pred_test})

plt.figure(dpi=600)
sns.scatterplot(graph, x='Score', y='pred_tree', color='purple')
plt.title('Analisando as Previsões', fontsize=10)
plt.xlabel('Score Observado', fontsize=10)
plt.ylabel('Score Previsto pelo Modelo', fontsize=10)
plt.axline((25, 25), (max(dados['Score']), max(dados['Score'])), linewidth=1, color='grey')
plt.show()

#%%######################### Random Forest ####################################
###############################################################################
#%% Iniciando o Grid Search

## Alguns hiperparâmetros do modelo

# n_estimators: qtde de árvores na floresta
# max_depth: profundidade máxima da árvore
# max_features: qtde de variáveis X consideradas na busca pelo melhor split
# min_samples_leaf: qtde mínima de observações para ser nó folha

# Vamos aplicar um Grid Search
param_grid_rf = {
    'n_estimators': [100, 500],
    'max_depth': [5, 10],
    'max_features': [3, 5, 7],
    'min_samples_leaf': [30, 50]
}

# Identificar o algoritmo em uso
rf_grid = RandomForestRegressor(random_state=100)

# Treinar os modelos para o grid search
rf_grid_model = GridSearchCV(estimator = rf_grid, 
                             param_grid = param_grid_rf,
                             scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                             cv=5, verbose=2)

rf_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
rf_best = rf_grid_model.best_estimator_

# Predict do modelo
rf_grid_pred_train = rf_best.predict(X_train)
rf_grid_pred_test = rf_best.predict(X_test)
   
#%% Importância das variáveis preditoras

rf_features = pd.DataFrame({'features':X.columns.tolist(),
                            'importance':np.round(rf_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(rf_features)

#%% Avaliando a RF (base de treino)

mse_train_rf_grid = mean_squared_error(y_train, rf_grid_pred_train)
mae_train_rf_grid = mean_absolute_error(y_train, rf_grid_pred_train)
r2_train_rf_grid = r2_score(y_train, rf_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_rf_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_rf_grid):.1f}")
print(f"MAE: {mae_train_rf_grid:.1f}")
print(f"R²: {r2_train_rf_grid:.1%}")

#%% Avaliando a RF (base de teste)

mse_test_rf_grid = mean_squared_error(y_test, rf_grid_pred_test)
mae_test_rf_grid = mean_absolute_error(y_test, rf_grid_pred_test)
r2_test_rf_grid = r2_score(y_test, rf_grid_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_rf_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_rf_grid):.1f}")
print(f"MAE: {mae_test_rf_grid:.1f}")
print(f"R²: {r2_test_rf_grid:.1%}")

#%%######################### XGBoost ##########################################
###############################################################################
#%% Iniciando o Grid Search

## Alguns hiperparâmetros do modelo
# n_estimators: qtde de árvores no modelo
# max_depth: profundidade máxima das árvores
# learning_rate: taxa de aprendizagem
# colsample_bytree: percentual de variáveis X subamostradas para cada árvore

# Vamos aplicar um Grid Search
param_grid_xgb = {
    'n_estimators': [100, 500, 700],
    'max_depth': [3, 5],
    'learning_rate': [0.001, 0.01, 0.1],
    'colsample_bytree': [0.5, 0.8],
}

# Identificar o algoritmo em uso
xgb_grid = XGBRegressor(random_state=100)

# Treinar os modelos para o grid search
xgb_grid_model = GridSearchCV(estimator = xgb_grid, 
                             param_grid = param_grid_xgb,
                             scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                             cv=5, verbose=2)

xgb_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
xgb_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
xgb_best = xgb_grid_model.best_estimator_

# Predict do modelo
xgb_grid_pred_train = xgb_best.predict(X_train)
xgb_grid_pred_test = xgb_best.predict(X_test)

#%% Importância das variáveis preditoras

xgb_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(xgb_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(xgb_features)

#%% Avaliando o XGB (base de treino)

mse_train_xgb_grid = mean_squared_error(y_train, xgb_grid_pred_train)
mae_train_xgb_grid = mean_absolute_error(y_train, xgb_grid_pred_train)
r2_train_xgb_grid = r2_score(y_train, xgb_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_xgb_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_xgb_grid):.1f}")
print(f"MAE: {mae_train_xgb_grid:.1f}")
print(f"R²: {r2_train_xgb_grid:.1%}")

#%% Avaliando o XGB (base de teste)

mse_test_xgb_grid = mean_squared_error(y_test, xgb_grid_pred_test)
mae_test_xgb_grid = mean_absolute_error(y_test, xgb_grid_pred_test)
r2_test_xgb_grid = r2_score(y_test, xgb_grid_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_xgb_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_xgb_grid):.1f}")
print(f"MAE: {mae_test_xgb_grid:.1f}")
print(f"R²: {r2_test_xgb_grid:.1%}")