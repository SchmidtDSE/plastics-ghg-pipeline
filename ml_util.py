"""Utilities to help with machine learning tasks.

License: BSD
"""
import csv
import functools
import json
import typing

import luigi  # type: ignore
import sklearn.ensemble  # type: ignore
import sklearn.linear_model  # type: ignore
import sklearn.metrics  # type: ignore
import sklearn.pipeline  # type: ignore
import sklearn.preprocessing  # type: ignore
import sklearn.svm  # type: ignore
import sklearn.tree  # type: ignore

import const
import data_struct
import prepare
import preprocess

OPT_FLOAT = typing.Optional[float]
OPT_INT = typing.Optional[int]
OPT_STR = typing.Optional[str]
SPLIT_DATASETS = typing.Dict[str, typing.List[data_struct.Change]]


class TrainedModel:
    """Structure a model which has already been trained."""

    def __init__(self, model, train_mae: OPT_FLOAT, validation_mae: OPT_FLOAT, test_mae: OPT_FLOAT):
        """Create a new record of a trained model.

        Args:
            model: The trained scikit learn model.
            mae: The mean absolute error if one was calculated or None otherwise.
        """
        self._model = model
        self._train_mae = train_mae
        self._validation_mae = validation_mae
        self._test_mae = test_mae

    def get_model(self):
        """Get the model which was trained.

        Returns:
            The trained scikit learn model.
        """
        return self._model

    def get_train_mae(self):
        """Get the training mean absolute error associated with the model.

        Returns:
            The training mean absolute error if one was calculated or None otherwise.
        """
        return self._train_mae

    def get_validation_mae(self):
        """Get the validation mean absolute error associated with the model.

        Returns:
            The validation mean absolute error if one was calculated or None otherwise.
        """
        return self._validation_mae

    def get_test_mae(self):
        """Get the test mean absolute error associated with the model.

        Returns:
            The test mean absolute error if one was calculated or None otherwise.
        """
        return self._test_mae


class ModelDefinition:
    """Structure describing how a model should be built."""

    def __init__(self, algorithm: str, depth: OPT_INT = None, estimators: OPT_INT = None,
        features: OPT_STR = None, alpha: OPT_FLOAT = None, kernel: OPT_STR = None,
        degree: OPT_INT = None):
        """Create a new definition of a model to be built.

        Args:
            algorithm: The name of the algorithm like "random forest" or similar.
            depth: The maximum allowed depth or None if no max / not applicable.
            estimators: The number of estimators to ensemble or None if not applicable.
            features: Maximum features strategy or None if not applicable.
            alpha: Regularization parameter or None if not applicable.
            kernel: The name of the kernel like "rbf" or None if not applicable.
            degree: The degree of the function to be fit or None if not applicable.
        """
        self._algorithm = algorithm
        self._depth = depth
        self._estimators = estimators
        self._features = features
        self._alpha = alpha
        self._kernel = kernel
        self._degree = degree

    def get_algorithm(self) -> str:
        """Get the name of the algorithm to use for the regressor.

        Returns:
            The name of the algorithm like "random forest" or similar.
        """
        return self._algorithm

    def get_depth(self) -> OPT_INT:
        """Get maximum allowed regressor depth.

        Returns:
            The maximum allowed depth or None if no max / not applicable.
        """
        return self._depth

    def get_estimators(self) -> OPT_INT:
        """Get the number of regressors to ensemble.

        Returns:
            The number of estimators to ensemble or None if not applicable.
        """
        return self._estimators

    def get_features(self) -> OPT_STR:
        """Get the feature limit strategy.

        Returns:
            Maximum features strategy or None if not applicable.
        """
        return self._features

    def get_alpha(self) -> OPT_FLOAT:
        """Get the regularization strategy.

        Returns:
            Regularization parameter or None if not applicable.
        """
        return self._alpha

    def get_kernel(self) -> OPT_STR:
        """Get the kernel to apply.

        Returns:
            The name of the kernel like "rbf" or None if not applicable.
        """
        return self._kernel

    def get_degree(self) -> OPT_INT:
        """Get the polynomial degree to fit.

        Returns:
            The degree of the function to be fit or None if not applicable.
        """
        return self._degree

    def is_valid(self) -> bool:
        """Determine if the pipeline can fulfill this definition's request / has enough info.

        Returns:
            True if the definition is valid / complete. False otherwise.
        """
        assert self._algorithm in const.SUPPORTED_ALGORITHMS

        strategy = {
            'linear': check_linear_config,
            'svr': check_svr_config,
            'tree': check_tree_config,
            'random forest': check_random_forest_config,
            'adaboost': check_adaboost_config
        }[self._algorithm]

        return strategy(self)

    def to_dict(self) -> typing.Dict:
        """Serialize this definition to a dictionary containing only primitives.

        Returns:
            Serialization of this definition.
        """
        return {
            'algorithm': self.get_algorithm(),
            'depth': self.get_depth(),
            'estimators': self.get_estimators(),
            'features': self.get_features(),
            'alpha': self.get_alpha(),
            'kernel': self.get_kernel(),
            'degree': self.get_degree()
        }

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'ModelDefinition':
        """Deserialize this model from a dictionary.

        Returns:
            Deserialized model.
        """
        get_int_maybe = lambda x: None if x is None else int(x)
        get_float_maybe = lambda x: None if x is None else float(x)
        get_str_maybe = lambda x: None if x is None else str(x)

        return ModelDefinition(
            target['algorithm'],
            get_int_maybe(target.get('depth', None)),
            get_int_maybe(target.get('estimators', None)),
            get_str_maybe(target.get('features', None)),
            get_float_maybe(target.get('alpha', None)),
            get_str_maybe(target.get('kernel', None)),
            get_int_maybe(target.get('degree', None))
        )


def check_linear_config(target: ModelDefinition) -> bool:
    """Check if enough information is avialable in a defintion to make a linear model.

    Returns:
        True if sufficient information and false otherwise.
    """
    return target.get_alpha() is not None


def check_svr_config(target: ModelDefinition) -> bool:
    """Check if enough information is avialable in a defintion to make a SVR model.

    Returns:
        True if sufficient information and false otherwise.
    """
    has_kernel = target.get_kernel() is not None
    has_degree = target.get_degree() is not None
    has_alpha = target.get_alpha() is not None
    return has_kernel and has_degree and has_alpha


def check_tree_config(target: ModelDefinition) -> bool:
    """Check if enough information is avialable in a defintion to make a regression tree model.

    Returns:
        True if sufficient information and false otherwise.
    """
    return target.get_depth() is not None


def check_random_forest_config(target: ModelDefinition) -> bool:
    """Check if enough information is avialable in a defintion to make a random forest model.

    Returns:
        True if sufficient information and false otherwise.
    """
    has_depth = target.get_depth() is not None
    has_estimators = target.get_estimators() is not None
    has_features = target.get_features() is not None
    return has_depth and has_estimators and has_features


def check_adaboost_config(target: ModelDefinition) -> bool:
    """Check if enough information is avialable in a defintion to make an adaboost model.

    Returns:
        True if sufficient information and false otherwise.
    """
    has_depth = target.get_depth() is not None
    has_estimators = target.get_estimators() is not None
    return has_depth and has_estimators


def build_model(target: ModelDefinition):
    """Build a scikit model fitting the requirements of the given definition.

    Args:
        target: The definition / request to fulfill.

    Returns:
        Model fitting the input definitions.
    """
    assert target.is_valid()

    strategy = {
        'linear': build_linear,
        'svr': build_svr,
        'tree': build_tree,
        'random forest': build_random_forest,
        'adaboost': build_adaboost
    }[target.get_algorithm()]

    return strategy(target)


def build_linear(target: ModelDefinition):
    """Make a linear model not yet fit to data.

    Args:
        target: Paramters with which to make the model.

    Returns:
        Newly constructed but untrained model.
    """
    return sklearn.linear_model.Ridge(alpha=target.get_alpha())


def build_svr(target: ModelDefinition):
    """Make a SVR model not yet fit to data.

    Args:
        target: Paramters with which to make the model.

    Returns:
        Newly constructed but untrained model.
    """
    alpha = target.get_alpha()
    assert alpha is not None  # This is a noop, for mypy to know it is not None
    regularization_constant = 1 - alpha

    model = sklearn.svm.SVR(
        kernel=target.get_kernel(),
        degree=target.get_degree(),
        C=regularization_constant
    )
    pipeline = sklearn.pipeline.Pipeline([
        ('scale', sklearn.preprocessing.StandardScaler()),
        ('svr', model)
    ])
    return pipeline


def build_tree(target: ModelDefinition):
    """Make a regression tree model not yet fit to data.

    Args:
        target: Paramters with which to make the model.

    Returns:
        Newly constructed but untrained model.
    """
    return sklearn.tree.DecisionTreeRegressor(max_depth=target.get_depth())


def build_random_forest(target: ModelDefinition):
    """Make a random forest model not yet fit to data.

    Args:
        target: Paramters with which to make the model.

    Returns:
        Newly constructed but untrained model.
    """
    return sklearn.ensemble.RandomForestRegressor(
        n_estimators=target.get_estimators(),
        max_depth=target.get_depth(),
        max_features=target.get_features()
    )


def build_adaboost(target: ModelDefinition):
    """Make a adaboost model not yet fit to data.

    Args:
        target: Paramters with which to make the model.

    Returns:
        Newly constructed but untrained model.
    """
    base_estimator = sklearn.tree.DecisionTreeRegressor(max_depth=target.get_depth())
    model = sklearn.ensemble.AdaBoostRegressor(
        n_estimators=target.get_estimators(),
        estimator=base_estimator
    )
    return model


class ModelTrainTask(luigi.Task):
    """Abstract base class / template class for model training step."""

    def requires(self):
        """Require data.

        Returns:
            Dictionary with named prerequisites.
        """
        return {
            'data': preprocess.PreprocessDataTask()
        }

    def _train_model(self, model, data: typing.Optional[SPLIT_DATASETS] = None) -> TrainedModel:
        """Train a model and evaluate its performance against a hidden set.

        Args:
            model: Scikit compatible model to train.
            data: Data to use or None if data should be loaded and assigned. Defaults to
                None.

        Returns:
            Model with evaulation information.
        """
        if data is None:
            data = self._load_data()

        train_data = data['train']
        train_inputs = [x.to_vector() for x in train_data]
        train_outputs = [x.get_response() for x in train_data]
        model.fit(train_inputs, train_outputs)

        def get_error(set_name: str) -> OPT_FLOAT:
            test_data = data[set_name]
            error: typing.Optional[float] = None
            if len(test_data) > 0:
                test_inputs = [x.to_vector() for x in test_data]
                test_outputs = [x.get_response() for x in test_data]
                predicted_outputs = model.predict(test_inputs)
                error = sklearn.metrics.mean_absolute_error(test_outputs, predicted_outputs)
                return error
            else:
                return None

        return TrainedModel(model, get_error('train'), get_error('valid'), get_error('test'))

    def _load_data(self) -> SPLIT_DATASETS:
        """Load preprocessed data as Change objects.

        Returns:
            Changes indexed by set assignment.
        """
        with self.input()['data'].open() as f:
            reader = csv.DictReader(f)
            changes = map(lambda x: data_struct.Change.from_dict(x), reader)

            # Cannot use records with unknown ratios for training or eval
            changes_allowed = filter(lambda x: x.has_response(), changes)

            assigned = map(lambda x: {self._choose_set(x): [x]}, changes_allowed)
            grouped = functools.reduce(lambda a, b: {
                'train': a.get('train', []) + b.get('train', []),
                'valid': a.get('valid', []) + b.get('valid', []),
                'test': a.get('test', []) + b.get('test', [])
            }, assigned)

        return grouped

    def _choose_set(self, target: data_struct.Change) -> str:
        """Determine which set an instance should be part of like training or test."""
        raise NotImplementedError('Use implementor.')


class PrechosenModelTrainTask(ModelTrainTask):
    """Abstract base class / template class for pre-choosen model training step.

    Derivative ModelTrainTask usable for situations where there is already a model architecture
    selected (outside the informational sweep).
    """

    def requires(self):
        """Require data and config."""
        return {
            'data': preprocess.PreprocessDataTask(),
            'config': prepare.CheckConfigFileTask()
        }

    def _train_model(self, model=None,
        data: typing.Optional[SPLIT_DATASETS] = None) -> TrainedModel:
        if model is None:
            model_config = self._load_config()
            model = build_model(model_config)

        return super()._train_model(model)

    def _load_config(self) -> ModelDefinition:
        """Load the ModelDefinition requested of the pipeline."""
        with self.input()['config'].open() as f:
            content = json.load(f)
            definition = ModelDefinition.from_dict(content['model'])

        return definition
