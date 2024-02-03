import csv
import itertools
import random
import typing

import data_struct
import ml_util


class SweepTask:

    def __init__(self, definition: ml_util.ModelDefinition, model = None,
        trained_model: typing.Optional[ml_util.TrainedModel] = None):
        self._definition = definition
        self._model = model
        self._trained_model = trained_model

    def get_definition(self) -> ml_util.ModelDefinition:
        return self._definition

    def get_model(self):
        return self._model

    def get_trained_model(self) -> typing.Optional[ml_util.TrainedModel]:
        return self._trained_model

    def get_with_model(self, target) -> 'SweepTask':
        return SweepTask(self._definition, target, self._trained_model)

    def get_with_trained_model(self, target: ml_util.TrainedModel) -> 'SweepTask':
        return SweepTask(self._definition, self._model, target)

    def get_sweep_dict(self) -> typing.Dict:
        definition_dict = self._definition.to_dict()

        if self._trained_model is None:
            definition_dict['trainMae'] = None
            definition_dict['validationMae'] = None
            definition_dict['testMae'] = None
        else:
            definition_dict['trainMae'] = self._trained_model.get_train_mae()
            definition_dict['validationMae'] = self._trained_model.get_validation_mae()
            definition_dict['testMae'] = self._trained_model.get_test_mae()

        return definition_dict


class ModelSweepTask(ml_util.ModelTrainTask):
    """Task which performs a sweep for a set of models, writing diagnostics to a CSV file."""

    def run(self):
        """Perform a simple 60 / 20 / 20 train and validation split to evaluate values"""
        data = self._load_data()

        definitions = self._get_model_definitions()
        tasks = map(
            lambda x: SweepTask(x),
            definitions
        )
        models = map(
            lambda x: x.get_with_model(ml_util.build_model(x.get_definition())),
            tasks
        )
        trained = map(
            lambda x: x.get_with_trained_model(self._train_model(x.get_model(), data)),
            models
        )
        outputs = map(
            lambda x: x.get_sweep_dict(),
            trained
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f)
            writer.writeheader()
            writer.writerows(outputs)

    def _choose_set(self, target: data_struct.Change) -> str:
        """Determine which set an instance should be part of like validation, trainingm or test."""
        return random.choice(['train', 'train', 'train', 'valid', 'test'])

    def _get_model_definitions(self) -> typing.Iterable[ml_util.ModelDefinition]:
        """Get the models to include in the sweep."""
        # Make linear models
        linear_alphas = map(lambda x: x / 10.0, range(0, 12, 2))
        linear_objs = map(lambda x: ml_util.ModelDefinition('linear', alpha=x), linear_alphas)

        # Make SVR models
        svr_alphas = map(lambda x: x / 10.0, range(0, 10, 2))
        svr_kernels = ['linear', 'poly', 'rbf']
        svr_degrees = range(1, 5)
        svr_naive_tuples = itertools.product(svr_alphas, svr_kernels, svr_degrees)
        svr_naive_objs = map(
            lambda x: ml_util.ModelDefinition('svr', alpha=x[0], kernel=x[1], degree=x[2]),
            svr_naive_tuples
        )

        # Prevent duplicate SVR work (degree 1 and degree 2 are same for all but poly)
        svr_objs = filter(
            lambda x: x.get_kernel() == 'poly' or x.get_degree() == 1,
            svr_naive_objs
        )

        # Make tree models
        tree_depths = range(2, 21)
        tree_objs = map(lambda x: ml_util.ModelDefinition('tree', depth=x), tree_depths)

        # Make random forest models
        rf_depths = range(2, 21)
        rf_estimators = range(5, 35, 5)
        rf_features = [1, 'sqrt', 'log2']
        rf_tuples = itertools.product(rf_depths, rf_estimators, rf_features)
        rf_objs = map(
            lambda x: ml_util.ModelDefinition(
                'random forest',
                depth=x[0],
                estimators=x[1],
                features=x[2]  # type: ignore
            ),
            rf_tuples
        )

        # Make adaboost models
        adaboost_depths = range(2, 21)
        adaboost_estimators = range(5, 35, 5)
        adaboost_tuples = itertools.product(adaboost_depths, adaboost_estimators)
        adaboost_objs = map(
            lambda x: ml_util.ModelDefinition(
                'adaboost',
                depth=x[0],
                estimators=x[1]
            ),
            adaboost_tuples
        )

        # Combine and return
        return itertools.chain(linear_objs, svr_objs, tree_objs, rf_objs, adaboost_objs)
