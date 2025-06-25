import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss,
    average_precision_score
)

# === Function to compute evaluation metrics ===
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


# === Crisscross Optimization Strategy (COS) ===
class COSOptimiser:
    def __init__(self, model_class, param_space, max_iter=30, population_size=10, temperature=1.0):
        """
        COS Optimiser to find the best hyperparameters.
        """
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

    # Generate a random individual (hyperparameter configuration)
    def _sample_individual(self):
        return {
            'learning_rate': np.random.uniform(*self.param_space['learning_rate']),
            'max_depth': np.random.randint(*self.param_space['max_depth']),
            'n_estimators': np.random.randint(*self.param_space['n_estimators']),
            'subsample': np.random.uniform(*self.param_space['subsample']),
            'colsample_bytree': np.random.uniform(*self.param_space['colsample_bytree']),
        }

    # Apply crisscross move (mutation) to create a new individual
    def _crisscross_move(self, individual):
        new_individual = {}
        for param in individual:
            low, high = self.param_space[param]
            if param in ['max_depth', 'n_estimators']:  # Integer-type parameters
                move = np.random.choice([-2, -1, 1, 2])
                new_value = int(individual[param] + move)
                new_individual[param] = int(np.clip(new_value, low, high))
            else:  # Float-type parameters
                move = np.random.normal(0, 0.05)
                new_value = individual[param] + move
                new_individual[param] = float(np.clip(new_value, low, high))
        return new_individual

    # Main optimization process
    def fit(self, X_train, y_train, X_test, y_test):
        population = [self._sample_individual() for _ in range(self.population_size)]
        best_individual = None

        for iteration_index in range(self.max_iter):
            print(f"\nCOS Iteration {iteration_index + 1}/{self.max_iter}")
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

                print(f"[{candidate_index}] | Accuracy = {acc:.4f} | Cost = {score:.6f} | MAP = {map_score:.4f}")

                scores.append((score, acc, individual))

                if score < self.best_score:
                    self.best_score = score
                    self.best_params = individual
                    best_individual = individual

                # Save metrics for history
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

            # === Elitism: Keep the best individual unchanged ===
            new_population = []
            for individual in population:
                if individual == best_individual:
                    new_population.append(individual)
                else:
                    new_population.append(self._crisscross_move(individual))
            population = new_population

            # Track best of this iteration
            best_score_iter, best_acc_iter, _ = min(scores, key=lambda x: (x[0], -x[1]))
            self.best_accuracy_per_iteration.append(best_acc_iter)
            self.best_cost_per_iteration.append(best_score_iter)

    # Return the final best model
    def get_best_model(self):
        return self.model_class(**self.best_params)
