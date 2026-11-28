from collections import Counter
from math import ceil
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _split_scores(feature_vector, target_vector, min_samples_leaf):
    order = np.argsort(feature_vector, kind="stable")
    values = np.asarray(feature_vector[order], dtype=float)
    cut_positions = np.flatnonzero(values[1:] > values[:-1]) + 1
    n_samples = len(values)
    cut_positions = cut_positions[
        (cut_positions >= min_samples_leaf)
        & (n_samples - cut_positions >= min_samples_leaf)
    ]
    if not cut_positions.size:
        return np.empty(0), np.empty(0)

    lowers = values[cut_positions - 1]
    uppers = values[cut_positions]
    thresholds = lowers / 2 + uppers / 2
    # If the midpoint rounds down, the upper endpoint gives the same split.
    thresholds = np.where(thresholds > lowers, thresholds, uppers)

    positives = np.cumsum(target_vector[order], dtype=np.int64)
    left_size = cut_positions
    right_size = n_samples - cut_positions
    left_pos = positives[cut_positions - 1]
    right_pos = positives[-1] - left_pos
    left_impurity = left_pos * (left_size - left_pos) / left_size
    right_impurity = right_pos * (right_size - right_pos) / right_size
    ginis = -2 * (left_impurity + right_impurity) / n_samples
    return thresholds, ginis


def _best_split_index(ginis):
    # Equivalent splits can differ by a rounding error in the Gini calculation.
    best_gini = np.max(ginis)
    ties = np.isclose(ginis, best_gini, rtol=4 * np.finfo(float).eps, atol=0)
    return np.flatnonzero(ties)[0]


def find_best_split(feature_vector, target_vector):
    """Return sorted thresholds, their negative weighted Gini, and the best pair.

    Each threshold separates the objects using ``feature < threshold``.
    Ties are resolved in favor of the smallest threshold. When there are no
    valid splits, the arrays are empty and the best threshold and score are None.
    """
    feature_vector = np.asarray(feature_vector)
    target_vector = np.asarray(target_vector)
    thresholds, ginis = _split_scores(feature_vector, target_vector == 1, 1)
    if not thresholds.size:
        return thresholds, ginis, None, None
    best = _best_split_index(ginis)
    return thresholds, ginis, thresholds[best], ginis[best]


def _sample_limit(value, n_samples, minimum, name):
    if value is None:
        return minimum
    if isinstance(value, Integral):
        if value >= minimum:
            return value
    elif isinstance(value, Real) and 0 < value <= 1:
        if value < 1 or name == "min_samples_split":
            return max(minimum, ceil(value * n_samples))
    raise ValueError(f"Invalid {name}: {value!r}")


class DecisionTree(ClassifierMixin, BaseEstimator):
    def __init__(
        self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None
    ):
        if any(kind not in {"real", "categorical"} for kind in feature_types):
            raise ValueError("There is an unknown feature type")
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if len(sub_y) < max(self._min_samples_split, 2 * self._min_samples_leaf) or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        positive = sub_y == self.classes_[-1]
        gini_best = -np.inf
        best_rule = None
        for feature, kind in enumerate(self.feature_types):
            values = sub_X[:, feature]
            if kind == "real":
                vector = values.astype(float, copy=False)
            else:
                categories, codes = np.unique(values, return_inverse=True)
                counts = np.bincount(codes)
                positive_rates = np.bincount(codes, weights=positive) / counts
                category_order = np.argsort(positive_rates, kind="stable")
                ranks = np.empty_like(category_order)
                ranks[category_order] = np.arange(len(categories))
                vector = ranks[codes]

            thresholds, ginis = _split_scores(vector, positive, self._min_samples_leaf)
            if not thresholds.size:
                continue
            best = _best_split_index(ginis)
            if ginis[best] > gini_best:
                gini_best = ginis[best]
                threshold = thresholds[best]
                split = vector < threshold
                best_rule = {"feature_split": feature}
                if kind == "real":
                    best_rule["threshold"] = threshold
                else:
                    best_rule["categories_split"] = frozenset(categories[ranks < threshold])

        if best_rule is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node.update(type="nonterminal", left_child={}, right_child={}, **best_rule)
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        while node["type"] != "terminal":
            feature = node["feature_split"]
            if self.feature_types[feature] == "real":
                go_left = float(x[feature]) < node["threshold"]
            else:
                go_left = x[feature] in node["categories_split"]
            node = node["left_child"] if go_left else node["right_child"]
        return node["class"]

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        if X.shape[1] != len(self.feature_types):
            raise ValueError("feature_types must have one entry per feature")
        if self.max_depth is not None and (
            not isinstance(self.max_depth, Integral) or self.max_depth < 1
        ):
            raise ValueError("max_depth must be an integer >= 1 or None")
        classes = np.unique(y)
        if len(classes) > 2:
            raise ValueError("DecisionTree supports only binary classification")

        self._min_samples_split = _sample_limit(
            self.min_samples_split, len(y), 2, "min_samples_split"
        )
        self._min_samples_leaf = _sample_limit(
            self.min_samples_leaf, len(y), 1, "min_samples_leaf"
        )
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self._tree = {}
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X, dtype=None, ensure_min_samples=0)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return np.array(
            [self._predict_node(x, self._tree) for x in X], dtype=self.classes_.dtype
        )
