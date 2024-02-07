"""Main entrypoint into the GHG preparation pipeline.

License: BSD
"""
import os

import luigi  # type: ignore

import const
import tasks_ml_sweep
import tasks_project


class SweepAndProjectTask(luigi.Task):
    """Task requiring a sweep and projected normalized values."""

    def requires(self):
        """Require both sweep and projections."""
        return {
            'sweep': tasks_ml_sweep.ModelSweepTask(),
            'projections': tasks_project.ProjectAndNormalizeTask()
        }

    def run(self):
        """Incidate that the pipeline has concluded."""
        with self.output().open('w') as f:
            f.write('Done')

    def output(self) -> str:
        """Write to the deploy directory a file confirming completion."""
        full_path = os.path.join(const.DEPLOY_DIR, 'confirm.txt')
        return luigi.LocalTarget(full_path)
