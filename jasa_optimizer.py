import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss,
    average_precision_score
)

# === Metric computation helper function ===
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


# === Jellyfish-Inspired Optimizer Class ===
class JASAOptimizer:
    def __init__(
        self,
        model_class,
        param_space,
        max_iter=25,
        population_size=5,
        temperature=1.0,
        metric_to_optimize='logloss'
    ):
        self.model_class = model_class
        self.param_space = param_space
        self.max_iter = max_iter
        self.population_size = population_size
        self.temperature = temperature
        self.metric_to_optimize = metric_to_optimize

        self.best_params = None
        self.best_score = np.inf
        self.history = []
        self.best_accuracy_per_iteration = []
        self.best_cost_per_iteration = []

    # === Generate a random individual (parameter set) ===
    def _sample_individual(self):
        return {
            'learning_rate': np.random.uniform(*self.param_space['learning_rate']),
            'max_depth': np.random.randint(*self.param_space['max_depth']),
            'n_estimators': np.random.randint(*self.param_space['n_estimators']),
            'subsample': np.random.uniform(*self.param_space['subsample']),
            'colsample_bytree': np.random.uniform(*self.param_space['colsample_bytree']),
        }

    # === Generate initial population ===
    def _sample_population(self):
        return [self._sample_individual() for _ in range(self.population_size)]

    # === Apply jellyfish movement to mutate an individual ===
    def _jellyfish_move(self, individual):
        new_individual = {}
        for param_key in individual:
            low, high = self.param_space[param_key]
            if param_key in ['max_depth', 'n_estimators']:
                move = np.random.choice([-1, 1]) * np.random.randint(1, 3)
                new_value = int(individual[param_key] + move)
                new_individual[param_key] = int(np.clip(new_value, low, high))
            else:
                move = np.random.normal(0, 0.02)
                new_value = individual[param_key] + move
                new_individual[param_key] = float(np.clip(new_value, low, high))
        return new_individual

    # === Main optimization loop ===
    def fit(self, X_train, y_train, X_test, y_test):
        population = self._sample_population()
        best_overall_individual = None
        best_overall_score = np.inf

        for iteration_index in range(self.max_iter):
            print(f"\nIteration {iteration_index + 1}/{self.max_iter}")
            population_scores = []

            for candidate_index, candidate_params in enumerate(population):
                model = self.model_class(**candidate_params)
                model.fit(X_train, y_train)

                # Predict and evaluate
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                acc, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(
                    y_test, y_pred, y_proba
                )

                # === Custom cost function ===
                cost_score = (1.0 / (acc + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + \
                             (0.3 * (fdr + fpr))

                print(f"[Candidate {candidate_index}] | Accuracy = {acc:.4f} | Cost = {cost_score:.4f} | MAP = {map_score:.4f}")

                # Store performance
                population_scores.append((candidate_params, acc, cost_score))

                # Update best individual if this one is better
                if cost_score < best_overall_score:
                    best_overall_score = cost_score
                    best_overall_individual = candidate_params

                # Store detailed metrics for analysis
                self.history.append({
                    'iteration': iteration_index,
                    **candidate_params,
                    'accuracy': acc,
                    'MAR': recall,
                    'FDR': fdr,
                    'FPR': fpr,
                    'MCC': mcc,
                    'precision': precision,
                    'logloss': logloss,
                    'MAP': map_score,
                    'Cost Function': cost_score
                })

            # === Apply strict elitism: keep best, mutate the rest ===
            new_population = []
            for candidate_params, _, _ in population_scores:
                if candidate_params == best_overall_individual:
                    new_population.append(candidate_params)  # Keep elite
                else:
                    new_population.append(self._jellyfish_move(candidate_params))
            population = new_population

            # === Evaluate and record best model at this iteration ===
            final_model = self.model_class(**best_overall_individual)
            final_model.fit(X_train, y_train)
            y_pred_final = final_model.predict(X_test)
            y_proba_final = final_model.predict_proba(X_test)[:, 1]

            acc_final, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(
                y_test, y_pred_final, y_proba_final
            )

            cost_final = (1.0 / (acc_final + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + \
                         (0.3 * (fdr + fpr))

            self.best_accuracy_per_iteration.append(acc_final)
            self.best_cost_per_iteration.append(cost_final)

        self.best_params = best_overall_individual
        self.best_score = best_overall_score

    # === Return final trained model with best parameters ===
    def get_best_model(self):
        return self.model_class(**self.best_params)
