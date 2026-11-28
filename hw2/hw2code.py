import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.asarray(feature_vector, dtype=float)
    order = np.argsort(feature_vector, kind="stable")
    values = feature_vector[order]
    cuts = np.flatnonzero(values[1:] > values[:-1]) + 1
    if not cuts.size:
        return np.empty(0), np.empty(0), None, None

    lower, upper = values[cuts - 1], values[cuts]
    thresholds = lower / 2 + upper / 2
    # If the midpoint rounds down, the upper endpoint preserves the split.
    thresholds = np.where(thresholds > lower, thresholds, upper)
    positives = np.cumsum(np.asarray(target_vector)[order] == 1)
    n_left, n_right = cuts, len(values) - cuts
    p_left = positives[cuts - 1]
    p_right = positives[-1] - p_left
    ginis = -2 * (p_left * (n_left - p_left) / n_left
                  + p_right * (n_right - p_right) / n_right) / len(values)
    threshold_best, gini_best = _best_threshold(thresholds, ginis)
    return thresholds, ginis, threshold_best, gini_best


def _best_threshold(thresholds, ginis):
    if not thresholds.size:
        return None, None
    # Equal scores can differ by a floating-point rounding error.
    ties = np.isclose(ginis, ginis.max(), rtol=4 * np.finfo(float).eps, atol=0)
    best = np.flatnonzero(ties)[0]
    return thresholds[best], ginis[best]


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if (len(sub_y) < max(self._min_samples_split, 2 * self._min_samples_leaf)
                or (self.max_depth is not None and depth >= self.max_depth)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self.feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float, copy=False)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                sorted_categories = sorted(counts, key=lambda key: (clicks[key] / counts[key], key))
                categories_map = {category: rank for rank, category in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)
            if self._min_samples_leaf > 1 and threshold is not None:
                values = np.sort(feature_vector)
                valid = ((thresholds > values[self._min_samples_leaf - 1])
                         & (thresholds <= values[-self._min_samples_leaf]))
                threshold, gini = _best_threshold(thresholds[valid], ginis[valid])
            if threshold is None:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [
                        category for category, rank in categories_map.items() if rank < threshold
                    ]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self.feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self.feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]
        if self.feature_types[feature] == "real":
            go_left = float(x[feature]) < node["threshold"]
        else:
            go_left = x[feature] in node["categories_split"]
        child = node["left_child"] if go_left else node["right_child"]
        return self._predict_node(x, child)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self._min_samples_split = self.min_samples_split or 2
        self._min_samples_leaf = self.min_samples_leaf or 1
        if isinstance(self._min_samples_split, (float, np.floating)):
            self._min_samples_split = max(2, int(np.ceil(self._min_samples_split * len(y))))
        if isinstance(self._min_samples_leaf, (float, np.floating)):
            self._min_samples_leaf = int(np.ceil(self._min_samples_leaf * len(y)))
        self.classes_ = np.unique(y)
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
