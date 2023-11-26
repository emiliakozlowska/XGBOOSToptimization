import numpy as np
import pandas as pd
import zipfile


pip install wget


#!wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
with zipfile.ZipFile("C:/Users/admin/source/Desktop/bank+marketing.zip", 'r') as z:
    z.extract('bank-additional.zip')

with zipfile.ZipFile("bank-additional.zip", 'r') as z:
      with z.open('bank-additional/bank-additional.csv') as f:
        dane = pd.read_csv(f, delimiter=";")

dane

pip install imblearn


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pandas as pd
import numpy as np

# Zmiana cech kategorialnych na numeryczne
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
dane = pd.get_dummies(dane, columns=categorical_features)

# Zmapowanie zmiennej Y do postaci binarnej
dane['y'] = dane['y'].map({'yes': 1, 'no': 0})

# Normalizacja zmiennych numerycznych
numerical_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
scaler = StandardScaler()
dane[numerical_features] = scaler.fit_transform(dane[numerical_features])

# Podział na X i Y
X = dane.drop('y', axis=1)
y = dane['y']

# Zbalansowanie zmiennej Y
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)

# Podział zbioru na train i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Dalszy podział jeszcze na zbiór do cross-validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# model
model = xgb.XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss',
                          max_depth=2,
                          min_child_weight=10,
                          reg_alpha=10,
                          reg_lambda=10,
                          subsample=0.7,
                          learning_rate=0.01,
                          gamma=0.3,
                          colsample_bytree=0.7)

# fit
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

# pred
y_pred = model.predict(X_test)

# metryki
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")

# walidacja krzyżowa
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores):.2f}")


# **PARTICLE SWARM**

get_ipython().system('pip install pyswarms')

import pyswarms as ps
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, make_scorer

# Hyperparameter optimization function
def f_per_particle(m, alpha):
    total_model = xgb.XGBClassifier(max_depth=int(m[0]),
                                    min_child_weight=int(m[1]),
                                    reg_alpha=m[2],
                                    reg_lambda=m[3],
                                    subsample=m[4],
                                    learning_rate=m[5],
                                    gamma=m[6],
                                    use_label_encoder=False,
                                    eval_metric='logloss')
    n_splits=5
    scorer = make_scorer(accuracy_score)
    scores = cross_val_score(total_model, X_train, y_train, cv=n_splits, scoring=scorer)
    return scores.mean()

def f(x):
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], alpha=0.88) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 30, 'p':2}

# Hyperparameters to optimize
dimensions = 7

# Bounds
max_bound = np.array([10, 20, 1, 1, 1, 0.2, 0.5])
min_bound = np.array([1, 1, 0, 0, 0.5, 0.01, 0])
bounds = (min_bound, max_bound)

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options, bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=50)

# Create model with optimized parameters
opt_model = xgb.XGBClassifier(max_depth=int(pos[0]),
                              min_child_weight=int(pos[1]),
                              reg_alpha=pos[2],
                              reg_lambda=pos[3],
                              subsample=pos[4],
                              learning_rate=pos[5],
                              gamma=pos[6],
                              use_label_encoder=False,
                              eval_metric='logloss')

opt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = opt_model.predict(X_test)

# Get accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'The accuracy score of the optimized model on test set: {accuracy:.4f}')



from sklearn.metrics import log_loss

# Tworzenie i trenowanie modelu XGBoost przed optymalizacją
model_pre_opt = model
model_pre_opt.fit(X_train, y_train)

# Przewidywanie prawdopodobieństw dla danych testowych
y_pred_proba_pre_opt = model_pre_opt.predict_proba(X_test)

# Obliczanie logloss na danych testowych
logloss_pre_opt = log_loss(y_test, y_pred_proba_pre_opt)

print(f'The logloss of the model before optimization on the test set: {logloss_pre_opt}')

# W przypadku modelu po optymalizacji, prawdopodobieństwa przewidywań mogą być obliczone jako:
y_pred_proba_post_opt = opt_model.predict_proba(X_test)

# A następnie logloss można obliczyć jako:
logloss_post_opt = log_loss(y_test, y_pred_proba_post_opt)

print(f'The logloss of the model after optimization on the test set: {logloss_post_opt}')


#     Algorytm genetyczny


import copy
from sklearn.metrics import log_loss

# model
model = xgb.XGBClassifier(use_label_encoder=False,
                          eval_metric='logloss',
                          max_depth=2,
                          min_child_weight=10,
                          reg_alpha=10,
                          reg_lambda=10,
                          subsample=0.7,
                          learning_rate=0.01,
                          gamma=0.3,
                          colsample_bytree=0.7)

# fit
model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

# pred
y_pred = model.predict(X_test)

current_error = log_loss(y_test, y_pred)
   
# Dopasowanie modelu
model.fit(X_train, y_train)

# Obliczanie prognoz na zbiorze walidacyjnym
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Obliczanie AUC-ROC
auc_roc = roc_auc_score(y_val, y_pred_proba)
# Obliczanie accuracy
acc = accuracy_score(y_val, y_pred)
# Obliczanie precision
prec = precision_score(y_val, y_pred)
# Obliczanie F1-score
f1 = f1_score(y_val, y_pred)
# Obliczanie współczynnika Giniego na podstawie AUC-ROC
gini_coef = 2 * auc_roc - 1
# Wyświetlanie wyników
print("AUC-ROC:", auc_roc)
print("Accuracy:", acc)
print("Precision:", prec)
print("F1-score:", f1)
print("Współczynnik Giniego:", gini_coef)
print(current_error)
print(model.get_params())

pip install deap

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from deap import base, creator, tools, algorithms

# Tworzenie funkcji celu (minimizacja straty logarytmicznej)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Inicjalizacja narzędzi DEAP
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Definicja funkcji oceny
def evaluate(individual):
    # Przetwarzanie wartości genotypu na parametry modelu
    params = {
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'max_depth': max(0, int(individual[0] * 10) + 1),
        'min_child_weight': int(individual[1] * 10) + 1,
        'reg_alpha': 10 ** (individual[2] * 4 - 2),
        'reg_lambda': 10 ** (individual[3] * 4 - 2),
        'subsample': individual[4] if 0 <= individual[4] <= 1 else (0 if individual[4] < 0 else 1),
        'learning_rate': 10 ** (individual[5] * 3 - 4),
        'gamma': individual[6],
        'colsample_bytree': individual[7] if 0 <= individual[7] <= 1 else (0 if individual[7] < 0 else 1),
        'min_split_loss': max(0.0, individual[8])  # Poprawka dla parametru min_split_loss
    }

    # Inicjalizacja i dopasowanie modelu XGBoost
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])

    # Obliczanie prognoz
    y_pred = model.predict_proba(X_val)

    # Obliczanie straty logarytmicznej
    loss = log_loss(y_val, y_pred)

    return loss,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    # Inicjalizacja populacji
    population = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.6, 0.2, 10

    print("Start ewolucji...")

    # Ewolucja populacji
    for gen in range(NGEN):
        print(f"Generacja {gen+1}/{NGEN}")

        # Selekcja osobników do reprodukcji
        offspring = toolbox.select(population, len(population))

        # Klonowanie wybranych osobników
        offspring = list(map(toolbox.clone, offspring))

        # Krzyżowanie i mutacja osobników
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Ocena nieocenionych osobników
        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit

        # Zastąpienie populacji potomnej populacją rodzicielską
        population[:] = offspring

        # Wypisanie najlepszego wyniku w każdej generacji
        best_individual = tools.selBest(population, k=1)[0]
        best_fitness = best_individual.fitness.values[0]
        print(f"Najlepsza wartość funkcji celu: {best_fitness}")

    print("Ewolucja zakończona.")

    # Wybór najlepszego osobnika
    
    best_individual = tools.selBest(population, k=1)[0]
    best_params = {
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'max_depth': max(0, int(best_individual[0] * 10) + 1),
        'min_child_weight': max(0,int(best_individual[1] * 10) + 1),
        'reg_alpha': 10 ** (best_individual[2] * 4 - 2),
        'reg_lambda': 10 ** (best_individual[3] * 4 - 2),
        'subsample': best_individual[4] if 0 <= best_individual[4] <= 1 else (0 if best_individual[4] < 0 else 1),
        'learning_rate': 10 ** (best_individual[5] * 3 - 4),
        'gamma': best_individual[6],
        'colsample_bytree': best_individual[7] if 0 <= best_individual[7] <= 1 else (0 if best_individual[7] < 0 else 1),
        'min_split_loss': max(0.0, best_individual[8] )
    }
    
    
    # Inicjalizacja i dopasowanie modelu XGBoost z najlepszymi parametrami
    best_model = XGBClassifier(**best_params)
    best_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])
    
    # Wyświetlenie najlepszych hiperparametrów
    print("Najlepsze hiperparametry:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    #miary dopasowania
    
    # Obliczanie prognoz na zbiorze testowym
    y_pred = best_model.predict(X_test)
    
    # Dopasowanie modelu
    best_model.fit(X_train, y_train)

    # Obliczanie prognoz na zbiorze walidacyjnym
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]

    # Obliczanie AUC-ROC
    auc_roc = roc_auc_score(y_val, y_pred_proba)

    # Obliczanie accuracy
    acc = accuracy_score(y_val, y_pred)

    # Obliczanie precision
    prec = precision_score(y_val, y_pred)

    # Obliczanie F1-score
    f1 = f1_score(y_val, y_pred)

    # Obliczanie współczynnika Giniego na podstawie AUC-ROC
    gini_coef = 2 * auc_roc - 1

    # Wyświetlanie wyników
    print("AUC-ROC:", auc_roc)
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("F1-score:", f1)
    print("Współczynnik Giniego:", gini_coef)

    return y_pred

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = main()


# Najlepsza wartość funkcji celu:  0.09551559484875465
# Ewolucja zakończona.
# 
# Najlepsze hiperparametry:
# use_label_encoder: False
# eval_metric: logloss
# max_depth: 8
# min_child_weight: 0
# reg_alpha: 0.5701260879872309
# reg_lambda: 7.796682141083987
# subsample: 0.5998389372906002
# learning_rate: 0.23190378846862963
# gamma: 0.47168138152436834
# colsample_bytree: 0.9018632613650786
# min_split_loss: 0.31590236036698865
# 
# 
# AUC-ROC: 0.9943443680337073
# Accuracy: 0.9598092643051771
# Precision: 0.9617486338797814
# F1-score: 0.9597818677573279
# Współczynnik Giniego: 0.9886887360674146


#       Random Search
# Przestrzeń parametrów do przeszukania
param_space = {
    'max_depth': np.linspace(1, 10, 10),
    'learning_rate': np.linspace(0.1, 1, 100),
    'reg_alpha': np.linspace(1, 30, 30),
    'reg_lambda': np.linspace(1, 30, 30),
    'gamma': np.linspace(1, 10, 10),
    'min_child_weight' : np.linspace(0.1, 1, 10),
    'subsample': np.linspace(0.1, 1, 10),
    'colsample_bytree': np.linspace(0, 1, 10),
    'min_split_loss': np.linspace(0, 10, 100),
}

# Liczba iteracji Random Search
num_iterations = 1000

# Inicjalizacja najlepszych parametrów i najlepszego logloss
best_params None
best_loss = np.inf

# Pętla Random Search
for iteration in tqdm(range(num_iterations)):
    params = {param: np.random.choice (values) for param, values in param_space.items()}
    loss = objective (params)
    
    if loss < best_loss:
        best_loss = loss
        best_params = params
        

#       Wyzarzanie symulowane
best_model copy.deepcopy (model)
best_loss = np.inf
print (best_loss)
T = 10.0 # Początkowa temperatura
T_min = 0.01 # Minimalna temperatura
cooling_rate = 0.9 # Współczynnik schładzania
np. random.seed(88)

# Iteracja
while T > T_min:
    # Generowanie nowego zestawu hiperparametrów
    new_model= copy.deepcopy (model)
    new_model.set_params (**{
        'max_depth': np.random.randint(1, 10),
        'min_child_weight': np.random. randint (0.1, 1),
        'reg_alpha': np.random.uniform(1, 30),
        'reg_lambda': np.random.uniform(1, 30),
        'subsample': np.random.uniform(0.1, 1),
        'learning_rate': np.random.uniform(0.1, 1),
        'gamma': np.random.uniform(1, 10),
        'colsample_bytree': np.random.uniform(0, 1),
        'min_split_loss': np.random.uniform(0, 10)
    })
    
    # Trenowanie i obliczanie błędu
    new_model.fit(X_train, y_train)
    new_loss log_loss (y_test, new_model.predict_proba(X_test))
    
    # Akceptacja nowego zestawu hiperparametrów
    if new_loss < best_loss:
        best_model = new_model
        best_loss new_loss
    elif np.random.rand() < np.exp(-(new_loss - best_loss) / T):
        model = new_model
    
    # Schładzanie
    T = cooling_rate
    
    # Najlepszy model
    model = best_model
