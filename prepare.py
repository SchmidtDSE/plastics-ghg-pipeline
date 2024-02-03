"""Luigi tasks for checking / preparing the environment.

License: BSD
"""
import os

import luigi  # type: ignore

import const


class CheckFileTask(luigi.Task):
    """Template task for checking that a file exists."""

    def run(self):
        """Simply check that a file exists."""
        if not os.path.exists(self.get_path()):
            raise RuntimeError('Could not find %s.' % self.get_path())

    def output(self):
        """The input file is the output."""
        return luigi.LocalTarget(self.get_path())

    def get_path(self) -> str:
        """Get the path of the file to check.

        Returns:
            File path
        """
        raise NotImplementedError('Use implementor.')


class CheckConfigFileTask(CheckFileTask):
    """Check that the job JSON file is present."""

    def get_path(self) -> str:
        return os.path.join(const.TASK_DIR, const.CONFIG_NAME)


class CheckTradeDataFileTask(CheckFileTask):
    """Check that the raw data file is present."""

    def get_path(self) -> str:
        return os.path.join(const.DEPLOY_DIR, const.TRADE_FRAME_NAME)
