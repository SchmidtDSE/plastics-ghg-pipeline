"""Logic for longitudinally projecting goods trade ratios.

License: BSD
"""
import csv

import luigi

import data_struct
import goods_ml_prod
import onnxruntime
import projection_util


class ProjectionTask(luigi.Task):
    """Project data without normalization."""

    def requires(self):
        """Require data to preprocess."""
        return {
            "data": prepare.CheckTradeDataFileTask(),
            "model": goods_ml_prod.TrainProdModelTask()
        }

    def run(self):
        """Project data."""
        inner_model = onnxruntime.InferenceSession(
            self.input()['model'].path,
            providers=['CPUExecutionProvider']
        )
        model = projection_util.OnnxPredictor(inner_model)
        index = self._build_index()
        inferring_index = projection_util.PredictionObservationIndexDecorator(index, model)

        #with self.output().open('w') as f:
        #    writer = csv.DictWriter(f)
        #    writer.writeheader()
        #    writer.writerows(output_dict)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'projected.csv'))

    def _build_index(self) -> data_struct.ObservationIndexable:
        """Create an index over the raw data file."""
        return data_struct.build_index_from_file(self.input()['data'].path)
