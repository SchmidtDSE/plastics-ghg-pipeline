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
    """Project data without normalization."""

    def requires(self):
        """Require data to preprocess."""
        return {
            'data': prepare.CheckTradeDataFileTask(),
            'model': goods_ml_prod.TrainProdModelTask()
        }

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'projected.csv'))

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        inner_model = onnxruntime.InferenceSession(
            self.input()['model'].path,
            providers=['CPUExecutionProvider']
        )
        model = projection_util.OnnxPredictor(inner_model)
        inferring_index = projection_util.InferringIndexedObservationsDecorator(index, model)
        return inferring_index

    def _get_require_response(self) -> bool:
        return False


class GoodsNormalizationTask(decorator_util.DecoratedIndexedObservationsTask):
    """Normalize projected data."""

    def requires(self):
        """Require data to preprocess."""
        return {
            'data': GoodsProjectionTask()
        }

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'normalized.csv'))

    def _add_decorator(self,
        index: data_struct.IndexedObservations) -> data_struct.IndexedObservations:
        return normalization_util.NormalizingIndexedObservationsDecorator(index)

    def _get_require_response(self) -> bool:
        return True
