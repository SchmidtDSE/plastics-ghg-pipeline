"""Logic for longitudinally projecting goods trade ratios.

License: BSD
"""
import os

import luigi  # type: ignore
import onnxruntime  # type: ignore

import const
import data_struct
import decorator_util
import goods_ml_prod
import normalization_util
import prepare
import projection_util


class GoodsProjectionTask(decorator_util.DecoratedIndexedObservationsTask):
    """Project data without normalization for debugging."""

    def requires(self):
        """Require data and model to project."""
        return {
            'data': prepare.CheckTradeDataFileTask(),
            'model': goods_ml_prod.TrainProdModelTask()
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


class GoodsNormalizationTask(decorator_util.DecoratedIndexedObservationsTask):
    """Normalize projected data for debugging."""

    def requires(self):
        """Require data and to normalize."""
        return {
            'data': GoodsProjectionTask()
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


class GoodsProjectAndNormalizeTask(decorator_util.DecoratedIndexedObservationsTask):
    """Production trask which both predicts unknown trade ratios and normalizes them.

    Production trask which both predicts unknown trade ratios and normalizes them before writing the
    updated data to disk.
    """

    def requires(self):
        """Require data and model to project."""
        return {
            'data': prepare.CheckTradeDataFileTask(),
            'model': goods_ml_prod.TrainProdModelTask()
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
        normalizing_index = normalization_util.NormalizingIndexedObservationsDecorator(
            inferring_index
        )
        return normalizing_index

    def _get_require_response(self) -> bool:
        """Operate on all records."""
        return False
