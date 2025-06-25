import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss,
    average_precision_score
)

# === Evaluation Metric Function ===
def compute_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    logloss = log_loss(y_true, y_proba)
    map_score = average_precision_score(y_true, y_proba)
    return acc, recall, fdr, fpr, mcc, precision, logloss, map_score


# === Dynamic Hybrid Optimization Algorithm with Simulated Annealing (DHOA-SA) ===
class DHOA_SA_Optimiser:
    def __init__(self, model_class, param_space, max_iter=30, population_size=5, temperature=1.0):
        self.model_class = model_class
        self.param_space = param_space
        self.max_iter = max_iter
        self.population_size = population_size
        self.temperature = temperature

        self.best_params = None
        self.best_score = np.inf
        self.history = []
        self.best_accuracy_per_iteration = []
        self.best_cost_per_iteration = []

    # === Generate a random set of hyperparameters ===
    def _sample_individual(self):
        return {
            'learning_rate': np.random.uniform(*self.param_space['learning_rate']),
            'max_depth': np.random.randint(*self.param_space['max_depth']),
            'n_estimators': np.random.randint(*self.param_space['n_estimators']),
            'subsample': np.random.uniform(*self.param_space['subsample']),
            'colsample_bytree': np.random.uniform(*self.param_space['colsample_bytree']),
        }

    # === Simulated Annealing-style exploration around the global best ===
    def _explore_exploit_sa(self, individual, global_best):
        new_individual = {}
        for key in individual:
            low, high = self.param_space[key]
            if key in ['max_depth', 'n_estimators']:  # Integer-type param
                delta = int(0.3 * (global_best[key] - individual[key]) + np.random.randint(-3, 4))
                new_value = int(individual[key] + delta)
                new_individual[key] = int(np.clip(new_value, low, high))
            else:  # Float-type param
                delta = 0.3 * (global_best[key] - individual[key]) + np.random.normal(0, 0.05)
                new_value = individual[key] + delta
                new_individual[key] = float(np.clip(new_value, low, high))
        return new_individual

    # === Main optimization loop ===
    def fit(self, X_train, y_train, X_test, y_test):
        population = [self._sample_individual() for _ in range(self.population_size)]
        best_individual = None

        for iteration_index in range(self.max_iter):
            print(f"\n🔁 DHOA-SA Iteration {iteration_index + 1}/{self.max_iter}")
            scores = []

            for candidate_index, individual in enumerate(population):
                model = self.model_class(**individual)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(
                    y_test, y_pred, y_proba
                )

                # Custom cost function (lower is better)
                score = (1.0 / (acc + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + \
                        (0.3 * (fdr + fpr))

                print(f"[{candidate_index}] | Accuracy = {acc:.4f} | Score = {score:.6f} | MAP = {map_score:.4f}")
                scores.append((individual, acc, score))

                if score < self.best_score:
                    self.best_score = score
                    self.best_params = individual
                    best_individual = individual

                # Save to history for analysis
                self.history.append({
                    'iteration': iteration_index,
                    **individual,
                    'accuracy': acc,
                    'MAR': recall,
                    'FDR': fdr,
                    'FPR': fpr,
                    'MCC': mcc,
                    'precision': precision,
                    'logloss': logloss,
                    'MAP': map_score,
                    'Cost Function': score
                })

            # === Apply strong elitism (keep best individual unchanged) ===
            new_population = []
            for individual, _, _ in scores:
                if individual == best_individual:
                    new_population.append(individual)  # Keep the best
                else:
                    new_population.append(self._explore_exploit_sa(individual, best_individual))

            population = new_population

            # === Evaluate and track best model again ===
            model_best = self.model_class(**best_individual)
            model_best.fit(X_train, y_train)

            y_pred_best = model_best.predict(X_test)
            y_proba_best = model_best.predict_proba(X_test)[:, 1]

            acc_best, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(
                y_test, y_pred_best, y_proba_best
            )

            score_best = (1.0 / (acc_best + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + \
                         (0.3 * (fdr + fpr))

            self.best_accuracy_per_iteration.append(acc_best)
            self.best_cost_per_iteration.append(score_best)

    # === Return model trained with best parameters ===
    def get_best_model(self):
        return self.model_class(**self.best_params)
