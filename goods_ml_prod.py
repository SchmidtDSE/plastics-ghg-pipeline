"""Tasks for validating and training the machine learning model in production.

License: BSD
"""
import os
import random

import luigi  # type: ignore
import numpy
import skl2onnx  # type: ignore

import const
import data_struct
import ml_util
import prepare
import preprocess


class TraditionalValidateModelTask(ml_util.PrechosenModelTrainTask):
    """Perform a simple 80 / 20 train and test split to confirm performance is sane."""

    def run(self):
        """Perform split and evaluate."""
        trained_model = self._train_model()

        observed_error = trained_model.get_test_mae()
        if observed_error > const.ALLOWED_TEST_ERROR:
            params = (observed_error, const.ALLOWED_TEST_ERROR)
            raise RuntimeError('Found test error of %.4f. Over limit of %.4f.' % params)

        with self.output().open('w') as f:
            f.write('Test error: %.4f' % observed_error)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'traditional_test.json'))

    def _choose_set(self, target: data_struct.Change) -> str:
        """Perform simple random split."""
        return random.choice(['train', 'train', 'train', 'train', 'test'])


class TemporalValidateModelTask(ml_util.PrechosenModelTrainTask):
    """Perform a temporal out of sample split to confirm performance is sane."""

    def run(self):
        """Perform split and evaluate."""
        trained_model = self._train_model()

        observed_error = trained_model.get_test_mae()
        if observed_error > const.ALLOWED_OUT_SAMPLE_ERROR:
            params = (observed_error, const.ALLOWED_OUT_SAMPLE_ERROR)
            raise RuntimeError('Found out sample error of %.4f. Over limit of %.4f.' % params)

        with self.output().open('w') as f:
            f.write('Out sample error: %.4f' % observed_error)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'out_sample_test.csv'))

    def _choose_set(self, target: data_struct.Change) -> str:
        """Perform split by time."""
        start_year = target.get_year()
        end_year = target.get_year() + target.get_years()
        if start_year < 2019 and end_year < 2019:
            return 'train'
        elif end_year == 2019:
            return 'test'
        else:
            return 'other'


class TrainProdModelTask(ml_util.PrechosenModelTrainTask):
    """Actually train the production model."""

    def requires(self):
        """Require validation in addition to data and config."""
        return {
            'data': preprocess.PreprocessDataTask(),
            'config': prepare.CheckConfigFileTask(),
            'traditionalValidation': TraditionalValidateModelTask(),
            'temporalValidation': TemporalValidateModelTask()
        }

    def run(self):
        """Train and serialize model."""
        trained_model = self._train_model()
        training_data = self._load_data()['train']

        with open(self.output().path, 'wb') as f:
            train_row = training_data[0].to_vector()

            # Set onnx type
            train_array = numpy.array(train_row).astype(numpy.float32).reshape(1, len(train_row))
            
            model_onnx = skl2onnx.to_onnx(trained_model.get_model(), train_array)
            f.write(model_onnx.SerializeToString())

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'model.onnx'))

    def _choose_set(self, target: data_struct.Change) -> str:
        """Put everything in training."""
        return 'train'
