"""Logic for longitudinally projecting trade ratios.

License: BSD
"""
import os

import luigi  # type: ignore
import onnxruntime  # type: ignore

import const
import data_struct
import decorator_util
import normalization_util
import projection_util
import tasks_ml_prod
import tasks_prepare


class ProjectionTask(decorator_util.DecoratedIndexedObservationsTask):
    """Project data without normalization for debugging."""

    def requires(self):
        """Require data and model to project."""
        return {
            'data': tasks_prepare.GetTradeDataFileTask(),
            'model': tasks_ml_prod.TrainProdModelTask()
        }

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'projected.csv'))

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        """Add the inferring decorator."""
        inner_model = onnxruntime.InferenceSession(
            self.input()['model'].path,
            providers=['CPUExecutionProvider']
        )
        model = projection_util.OnnxPredictor(inner_model)
        inferring_index = projection_util.InferringIndexedObservationsDecorator(index, model)
        return inferring_index

    def _get_require_response(self) -> bool:
        """Operate on all records."""
        return False


class NormalizationTask(decorator_util.DecoratedIndexedObservationsTask):
    """Normalize projected data for debugging."""

    def requires(self):
        """Require data and to normalize."""
        return {
            'data': ProjectionTask()
        }

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'normalized.csv'))

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        """Apply a normalizing decorator."""
        return normalization_util.NormalizingIndexedObservationsDecorator(index)

    def _get_require_response(self) -> bool:
        """Only operate on records with previously inferred or actual trade ratios."""
        return True


class MakeProdProjectionDataTask(decorator_util.DecoratedIndexedObservationsTask):
    """Production trask which both predicts unknown trade ratios and normalizes them if enabled.

    Production trask which both predicts unknown trade ratios and normalizes them before writing the
    updated data to disk where normalization is controlled by const.ENABLE_NORMALIZATION.
    """

    def requires(self):
        """Require data and model to project."""
        return {
            'data': tasks_prepare.GetTradeDataFileTask(),
            'model': tasks_ml_prod.TrainProdModelTask()
        }

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'projected_and_normalized.csv'))

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        """Stack the inferring and normalizing decorators."""
        inner_model = onnxruntime.InferenceSession(
            self.input()['model'].path,
            providers=['CPUExecutionProvider']
        )
        model = projection_util.OnnxPredictor(inner_model)
        inferring_index = projection_util.InferringIndexedObservationsDecorator(index, model)

        if const.ENABLE_NORMALIZATION:
            normalizing_index = normalization_util.NormalizingIndexedObservationsDecorator(
                inferring_index
            )
            return normalizing_index
        else:
            return inferring_index

    def _get_require_response(self) -> bool:
        """Operate on all records."""
        return False
