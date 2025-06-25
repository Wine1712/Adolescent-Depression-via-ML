import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss,
    average_precision_score
)

# Helper function to calculate key classification metrics
def compute_metrics(y_true, y_pred, y_proba):
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


class IWO_SA_Optimizer:
    def __init__(self, model_class, param_space, max_iter=30, initial_pop=5, max_pop=20, temperature=1.0):
        """
        IWO-SA optimizer setup
        - model_class: the ML model to optimize
        - param_space: dictionary with parameter bounds
        - max_iter: how many iterations to run
        - initial_pop: number of initial random individuals
        - max_pop: max population size per generation
        """
        self.model_class = model_class
        self.param_space = param_space
        self.max_iter = max_iter
        self.initial_pop = initial_pop
        self.max_pop = max_pop
        self.temperature = temperature

        # Tracking best results
        self.best_params = None
        self.best_score = np.inf
        self.history = []
        self.best_accuracy_per_iteration = []
        self.best_cost_per_iteration = []

    # Create one random individual (a set of hyperparameters)
    def _sample_individual(self):
        return {
            'learning_rate': np.random.uniform(*self.param_space['learning_rate']),
            'max_depth': np.random.randint(*self.param_space['max_depth']),
            'n_estimators': np.random.randint(*self.param_space['n_estimators']),
            'subsample': np.random.uniform(*self.param_space['subsample']),
            'colsample_bytree': np.random.uniform(*self.param_space['colsample_bytree']),
        }

    # Slightly change an individual's parameters (mutation)
    def _mutate(self, individual, step_size=0.1):
        mutated = individual.copy()
        for param in mutated:
            if param in ['max_depth', 'n_estimators']:  # Integer parameters
                mutation = int(np.random.choice([-1, 1]) * np.random.randint(1, 3))
                mutated[param] = int(np.clip(mutated[param] + mutation, *self.param_space[param]))
            else:  # Float parameters
                mutation = np.random.normal(0, step_size)
                mutated[param] = float(np.clip(mutated[param] + mutation, *self.param_space[param]))
        return mutated

    def fit(self, X_train, y_train, X_val, y_val):
        # Step 1: Create initial random population
        population = [self._sample_individual() for _ in range(self.initial_pop)]
        best_individual = None

        for iteration_index in range(self.max_iter):
            print(f"\n🌱 IWO-SA Iteration {iteration_index + 1}/{self.max_iter}")
            offspring = []

            # Step 2: Evaluate each individual in the population
            for individual in population:
                model = self.model_class(**individual)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                acc, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(y_val, y_pred, y_proba)
                cost_score = (1.0 / (acc + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + (0.3 * (fdr + fpr))

                print(f"Accuracy = {acc:.4f} | Cost = {cost_score:.6f} | MAP = {map_score:.4f}")

                # Update best if this one is better
                if cost_score < self.best_score:
                    self.best_score = cost_score
                    self.best_params = individual
                    best_individual = individual

                # Save current result
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
                    'Cost Function': cost_score
                })

                # Step 3: Generate offspring from this individual
                num_offspring = np.random.randint(1, 4)
                for _ in range(num_offspring):
                    child = self._mutate(individual, step_size=0.1)
                    offspring.append(child)

            # Step 4: Combine parents and children for next generation
            all_candidates = population + offspring
            evaluated_candidates = []

            for candidate in all_candidates:
                model = self.model_class(**candidate)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]

                acc, recall, fdr, fpr, mcc, precision, logloss, map_score = compute_metrics(y_val, y_pred, y_proba)
                cost_score = (1.0 / (acc + 0.5 * map_score + 0.3 * mcc + 0.3 * precision + 1e-8)) + (0.3 * (fdr + fpr))

                evaluated_candidates.append((cost_score, acc, candidate))

            # Step 5: Keep best candidates for next generation
            evaluated_candidates.sort(key=lambda x: (x[0], -x[1]))  # prioritize low cost, then high accuracy
            population = [x[2] for x in evaluated_candidates[:self.max_pop]]

            # Save best of this generation
            best_score_gen, best_acc_gen, _ = evaluated_candidates[0]
            self.best_accuracy_per_iteration.append(best_acc_gen)
            self.best_cost_per_iteration.append(best_score_gen)

    # Return the final best-trained model
    def get_best_model(self):
        return self.model_class(**self.best_params)
