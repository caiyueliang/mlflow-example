import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.store.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository


def test_runs_artifact_repo_init():
    artifact_location = "s3://blah_bucket/"
    experiment_id = mlflow.create_experiment("expr_abc", artifact_location)
    with mlflow.start_run(experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
    runs_uri = "runs:/%s/path/to/model" % run_id
    runs_repo = RunsArtifactRepository(runs_uri)

    assert runs_repo.artifact_uri == runs_uri
    assert isinstance(runs_repo.repo, S3ArtifactRepository)
    expected_absolute_uri = "%s%s/artifacts/path/to/model" % (artifact_location, run_id)
    assert runs_repo.repo.artifact_uri == expected_absolute_uri


# 使用Tracking Service API管理实验和运行
# MLflow提供了更详细的跟踪服务API，用于管理实验并直接运行，可通过mlflow.tracking模块中的客户端SDK获得。
# 这样就可以查询有关过去运行的数据，记录有关它们的其他信息，创建实验，为运行添加标记等。
def manage_and_run():
    client = MlflowClient()
    experiments = client.list_experiments()                     # returns a list of mlflow.entities.Experiment
    for experiment in experiments:
        print('[manage_and_run] experiment: ', experiment.experiment_id, experiment.name)

        experiment_id = experiment.experiment_id
        run_infos = client.list_run_infos(experiment_id)
        print('[manage_and_run] run_infos: ', run_infos)

    run = client.create_run(experiments[0].experiment_id)       # returns mlflow.entities.Run
    client.log_param(run.info.run_id, "hello", "world")
    client.set_terminated(run.info.run_id)


if __name__ == '__main__':
    manage_and_run()
