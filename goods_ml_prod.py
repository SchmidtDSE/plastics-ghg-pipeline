"""Tasks for validating and training the machine learning model in production.

License: BSD
"""
import csv
import functools
import json
import os
import random
import typing

import luigi  # type: ignore
import numpy
import skl2onnx  # type: ignore
import sklearn.metrics  # type: ignore

import const
import data_struct
import ml_util
import prepare
import preprocess


class ModelTrainTask(luigi.Task):
    """Abstract base class / template class for single model training step."""

    def requires(self):
        """Require data and config."""
        return {
            'data': preprocess.PreprocessDataTask(),
            'config': prepare.CheckConfigFileTask()
        }

    def _train_model(self) -> ml_util.TrainedModel:
        """Train a model and evaluate its performance against a hidden set."""
        split_datasets = self._load_data()
        model_config = self._load_config()
        model = ml_util.build_model(model_config)

        train_data = split_datasets['train']
        train_inputs = [x.to_vector() for x in train_data]
        train_outputs = [x.get_response() for x in train_data]
        model.fit(train_inputs, train_outputs)

        test_data = split_datasets['test']
        error: typing.Optional[float] = None
        if len(test_data) > 0:
            test_inputs = [x.to_vector() for x in test_data]
            test_outputs = [x.get_response() for x in test_data]
            predicted_outputs = model.predict(test_inputs)
            error = sklearn.metrics.mean_absolute_error(test_outputs, predicted_outputs)

        return ml_util.TrainedModel(model, error)

    def _load_data(self) -> typing.Dict[str, typing.List[data_struct.Change]]:
        """Load preprocessed data as Change objects."""
        with self.input()['data'].open() as f:
            reader = csv.DictReader(f)
            changes = map(lambda x: data_struct.Change.from_dict(x), reader)
            assigned = map(lambda x: {self._choose_set(x): [x]}, changes)
            grouped = functools.reduce(lambda a, b: {
                'train': a.get('train', []) + b.get('train', []),
                'test': a.get('test', []) + b.get('test', [])
            }, assigned)

        return grouped

    def _load_config(self) -> ml_util.ModelDefinition:
        """Load the ModelDefinition requested of the pipeline."""
        with self.input()['config'].open() as f:
            content = json.load(f)
            definition = ml_util.ModelDefinition.from_dict(content['model'])

        return definition

    def _choose_set(self, target: data_struct.Change) -> str:
        """Determine which set an instance should be part of like training or test."""
        raise NotImplementedError('Use implementor.')


class TraditionalValidateModelTask(ModelTrainTask):
    """Perform a simple 80 / 20 train and test split to confirm performance is sane."""

    def run(self):
        """Perform split and evaluate."""
        trained_model = self._train_model()

        observed_error = trained_model.get_mae()
        if observed_error > const.ALLOWED_VALIDATION_ERROR:
            params = (observed_error, const.ALLOWED_VALIDATION_ERROR)
            raise RuntimeError('Found validation error of %.4f. Over limit of %.4f.' % params)

        with self.output().open('w') as f:
            f.write('Validation error: %.4f' % observed_error)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'traditional_validation.csv'))

    def _choose_set(self, target: data_struct.Change) -> str:
        """Perform simple random split."""
        return random.choice(['train', 'train', 'train', 'train', 'test'])


class TemporalValidateModelTask(ModelTrainTask):
    """Perform a temporal out of sample split to confirm performance is sane."""

    def run(self):
        """Perform split and evaluate."""
        trained_model = self._train_model()

        observed_error = trained_model.get_mae()
        if observed_error > const.ALLOWED_OUT_SAMPLE_ERROR:
            params = (observed_error, const.ALLOWED_OUT_SAMPLE_ERROR)
            raise RuntimeError('Found out sample error of %.4f. Over limit of %.4f.' % params)

        with self.output().open('w') as f:
            f.write('Out sample error: %.4f' % observed_error)

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'out_sample_validation.csv'))

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


class TrainProdModelTask(ModelTrainTask):
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
            train_row_numpy = numpy.array(train_row)  # Allows onnx type sniffing
            model_onnx = skl2onnx.to_onnx(trained_model.get_model(), train_row_numpy)
            f.write(model_onnx.SerializeToString())

    def output(self):
        """Output preprocessed data."""
        return luigi.LocalTarget(os.path.join(const.DEPLOY_DIR, 'model.onnx'))

    def _choose_set(self, target: data_struct.Change) -> str:
        """Put everything in training."""
        return 'train'
