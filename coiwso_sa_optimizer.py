# File: Optimizer/coiwso_sa_optimizer.py

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss,
    average_precision_score
)

# === Helper Function to Compute Model Evaluation Metrics ===
def compute_metrics(y_true, y_pred, y_proba):
    """
    Computes various classification metrics to evaluate model performance.
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Discovery Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    logloss = log_loss(y_true, y_proba)
    map_score = average_precision_score(y_true, y_proba)
    return acc, recall, fdr, fpr, mcc, precision, logloss, map_score


# === Hybrid COIWSO-SA Optimizer Class ===
class COIWSO_SA_Optimizer:
    def __init__(self, model_class, param_space, max_iter=30, population_size=5, temperature=1.0, metric_to_optimize='logloss'):
        """
        Initializes the optimizer with model and search parameters.
        """
        self.model_class = model_class  # ML model to be optimized
        self.param_space = param_space  # Hyperparameter ranges
        self.max_iter = max_iter        # Number of optimization iterations
        self.population_size = population_size
        self.temperature = temperature  # Used in SA-style mutation (optional)
        self.metric_to_optimize = metric_to_optimize

        # Tracking results
        self.best_params = None
        self.best_score = np.inf
        self.history = []
        self.best_accuracy_per_iteration = []
        self.best_cost_per_iteration = []

    # === Create one random candidate (set of hyperparameters) ===
    def _sample_individual(self):
        return {
            'learning_rate': np.random.uniform(*self.param_space['learning_rate']),
            'max_depth': np.random.randint(*self.param_space['max_depth']),
            'n_estimators': np.random.randint(*self.param_space['n_estimators']),
            'subsample': np.random.uniform(*self.param_space['subsample']),
            'colsample_bytree': np.random.uniform(*self.param_space['colsample_bytree']),
        }

    # === COS Operator: Combine two parents to create a new child ===
    def _cos_operator(self, parent1, parent2):
        child = {}
        # Crossover: choose value from parent1 or parent2
        for key in parent1:
            child[key] = parent1[key] if np.random.rand() < 0.5 else parent2[key]

        # Apply mutation to the child
        for key in child:
            low, high = self.param_space[key]
            if key in ['max_depth', 'n_estimators']:
                mutation = int(np.random.choice([-1, 1]) * np.random.randint(1, 3))
                child[key] = int(np.clip(child[key] + mutation, low, high))
            else:
                mutation = np.random.normal(0, 0.05)
                child[key] = float(np.clip(child[key] + mutation, low, high))
        return child

    # === IWO Operator: Mutate a single individual ===
    def _iwo_operator(self, individual):
        mutant = individual.copy()
        for key in mutant:
            low, high = self.param_space[key]
            if key in ['max_depth', 'n_estimators']:
                mutation = int(np.random.choice([-1, 1]) * np.random.randint(1, 4))
                mutant[key] = int(np.clip(mutant[key] + mutation, low, high))
            else:
                mutation = np.random.normal(0, 0.1)
                mutant[key] = float(np.clip(mutant[key] + mutation, low, high))
        return mutant

    # === Main Optimization Loop ===
    def fit(self, X_train, y_train, X_val, y_val):
        """
        Runs the optimization algorithm across multiple iterations.
        """
        # Step 1: Generate initial population
        population = [self._sample_individual() for _ in range(self.population_size)]

        for iteration_index in range(self.max_iter):
            print(f"\n🔄 COIWSO-SA Iteration {iteration_index + 1}/{self.max_iter}")
            new_population = []
            evaluated = []

            # Step 2: Evaluate all individuals
            for individual in population:
                model = self.model_class(**individual)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                # Compute evaluation metrics
                acc, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(y_val, y_pred, y_proba)

                # Define custom cost function (lower is better)
                cost = (1.0 / (acc + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + (0.3 * (fdr + fpr))

                print(f" Accuracy = {acc:.4f} | Cost Function = {cost:.6f} | MAP = {map_score:.4f}")

                # Keep record of this individual
                evaluated.append((cost, acc, individual))

                # Update global best
                if cost < self.best_score:
                    self.best_score = cost
                    self.best_params = individual

                # Save full metrics to history
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
                    'Cost Function': cost
                })

            # Step 3: Apply elitism — keep best individual
            evaluated.sort(key=lambda x: (x[0]))  # Sort by cost
            elite_individual = evaluated[0][2]
            elite_accuracy = evaluated[0][1]
            elite_cost = evaluated[0][0]

            # Step 4: Generate new individuals using COS or IWO
            for _ in range(self.population_size - 1):
                if np.random.rand() < 0.5:
                    # COS: Combine two parents
                    parent1, parent2 = np.random.choice(population, size=2, replace=False)
                    candidate = self._cos_operator(parent1, parent2)
                else:
                    # IWO: Mutate a single parent
                    parent = population[np.random.randint(0, len(population))]
                    candidate = self._iwo_operator(parent)

                new_population.append(candidate)

            # Step 5: Update population for next generation
            population = [elite_individual] + new_population

            # Track best scores per iteration
            self.best_accuracy_per_iteration.append(elite_accuracy)
            self.best_cost_per_iteration.append(elite_cost)

    # === Return Best-Trained Model ===
    def get_best_model(self):
        """
        Returns the final model trained with the best parameters.
        """
        return self.model_class(**self.best_params)
