import mlflow
from mlflow.tracking import MlflowClient

from monte_carlo_demo.MonteCarlo import MonteCarlo
from monte_carlo_demo.common import Job


class MonteCarloJob(Job):

    def launch(self):
        self.logger.info("Launching Evaluation Job")

        mlflow.set_experiment(self.conf['experiment_path'])
        experiment_id = MlflowClient().get_experiment_by_name(self.conf['experiment_path']).experiment_id
        with mlflow.start_run(run_name='MC_EVALUATE') as parent_run:
            monte_carlo = MonteCarlo(self.conf, self.spark, experiment_id)
            monte_carlo.evaluate()
            mlflow.end_run()
        self.logger.info("Evaluation Job finished!")


if __name__ == "__main__":
    job = MonteCarloJob()
    job.launch()
