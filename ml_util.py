import typing

import sklearn.ensemble
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

import const

OPT_FLOAT = typing.Optional[float]
OPT_INT = typing.Optional[int]
OPT_STR = typing.Optional[str]


class TrainedModel:

    def __init__(self, model, mae: OPT_FLOAT):
        self._model = model
        self._mae = mae

    def get_model(self):
        return self._model

    def get_mae(self):
        return self._mae


class ModelDefinition:

    def __init__(self, algorithm: str, depth: OPT_INT, estimators: OPT_INT, features: OPT_STR,
        alpha: OPT_FLOAT, kernel: OPT_STR, degree: OPT_INT):
        self._algorithm = algorithm
        self._depth = depth
        self._estimators = estimators
        self._features = features
        self._alpha = alpha
        self._kernel = kernel
        self._degree = degree
    
    def get_algorithm(self) -> str:
        return self._algorithm
    
    def get_depth(self) -> OPT_INT:
        return self._depth
    
    def get_estimators(self) -> OPT_INT:
        return self._estimators
    
    def get_features(self) -> OPT_STR:
        return self._features
    
    def get_alpha(self) -> OPT_FLOAT:
        return self._alpha
    
    def get_kernel(self) -> OPT_STR:
        return self._kernel
    
    def get_degree(self) -> OPT_INT:
        return self._degree

    def is_valid(self) -> bool:
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
        return {
            'depth': self.get_depth(),
            'estimators': self.get_estimators(),
            'features': self.get_features(),
            'alpha': self.get_alpha(),
            'kernel': self.get_kernel(),
            'degree': self.get_degree()
        }

    @classmethod
    def from_dict(cls, target: typing.Dict) -> 'ModelDefinition':
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
    return target.get_alpha() is not None

def check_svr_config(target: ModelDefinition) -> bool:
    has_kernel = target.get_kernel() is not None
    has_degree = target.get_degree() is not None
    has_alpha = target.get_alpha() is not None
    return has_kernel and has_degree and has_alpha

def check_tree_config(target: ModelDefinition) -> bool:
    return target.get_depth() is not None

def check_random_forest_config(target: ModelDefinition) -> bool:
    has_depth = target.get_depth() is not None
    has_estimators = target.get_estimators() is not None
    has_features = target.get_features() is not None
    return has_depth and has_estimators and has_features

def check_adaboost_config(target: ModelDefinition) -> bool:
    has_depth = target.get_depth() is not None
    has_estimators = target.get_estimators() is not None
    return has_depth and has_estimators

def build_model(target: ModelDefinition):
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
    return sklearn.linear_model.Ridge(alpha=target.get_alpha())

def build_svr(target: ModelDefinition):
    model = sklearn.svm.SVR(
        kernel=target.get_kernel(),
        degree=target.get_degree(),
        C=1-target.get_alpha()
    )
    pipeline = sklearn.pipeline.Pipeline([
        ('scale', sklearn.preprocessing.StandardScaler()),
        ('svr', model)
    ])
    return pipeline

def build_tree(target: ModelDefinition):
    return sklearn.tree.DecisionTreeRegressor(max_depth=target.get_depth())

def build_random_forest(target: ModelDefinition):
    return sklearn.ensemble.RandomForestRegressor(
        n_estimators=target.get_estimators(),
        max_depth=target.get_depth(),
        max_features=target.get_features()
    )

def build_adaboost(target: ModelDefinition):
    base_estimator = sklearn.tree.DecisionTreeRegressor(max_depth=target.get_depth())
    model = sklearn.ensemble.AdaBoostRegressor(
        n_estimators=target.get_estimators(),
        estimator=base_estimator
    )
    return model
