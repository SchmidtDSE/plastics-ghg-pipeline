"""Logic for running an informational model sweep to monitor changes as data evolve.

License: BSD
"""
import csv
import itertools
import random
import typing

import data_struct
import ml_util


class SweepTask:
    """Record of a single task to be performed or that was performed in a model sweep."""

    def __init__(self, definition: ml_util.ModelDefinition, model=None,
        trained_model: typing.Optional[ml_util.TrainedModel] = None):
        """Create a new task record.

        Args:
            definition: The definition of the model to try.
            model: The Scikit-Learn compatible model tried which may or may not have been trained.
                Pass None if not yet built.
            trained_model: The model with errors / diagnostics after training. Pass None if not yet
                trained.
        """
        self._definition = definition
        self._model = model
        self._trained_model = trained_model

    def get_definition(self) -> ml_util.ModelDefinition:
        """Get the definition of the model to be tried.

        Returns:
            The definition of the model to try.
        """
        return self._definition

    def get_model(self):
        """Get the model built for this task.

        Returns:
            The Scikit-Learn compatible model tried which may or may not have been trained. Returns
            None if not yet built.
        """
        return self._model

    def get_trained_model(self) -> typing.Optional[ml_util.TrainedModel]:
        """Get the model from this task with evaluation information.

        Returns:
            The model with errors / diagnostics after training. Returns None if not yet trained.
        """
        return self._trained_model

    def get_with_model(self, target) -> 'SweepTask':
        """Make a copy of this task record but with a new value for model.

        Args:
            target: The Scikit-Learn compatible model tried which may or may not have been trained.
                Pass None if not yet built.

        Returns:
            Copy of this record with the new information.
        """
        return SweepTask(self._definition, target, self._trained_model)

    def get_with_trained_model(self, target: ml_util.TrainedModel) -> 'SweepTask':
        """Make a copy of this task record but with a new value for trained model.

        Args:
            target: The model with errors / diagnostics after training. Pass None if not yet
                trained.

        Returns:
            Copy of this record with the new information.
        """
        return SweepTask(self._definition, self._model, target)

    def get_sweep_dict(self) -> typing.Dict:
        """Build a dictionary describing this record and evaluation information if available.

        Returns:
            Dictionary describing this record made up of only primitives.
        """
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

        outputs_str = map(lambda x: self._force_str(x), outputs)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f)
            writer.writeheader()
            writer.writerows(outputs_str)

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

    def _force_str(self, target: typing.Dict) -> typing.Dict[str, str]:
        """Force a dictionary's values to be only strings."""
        items = target.items()
        items_str = map(lambda x: (x[0], str(x[1])), items)
        return dict(items_str)
