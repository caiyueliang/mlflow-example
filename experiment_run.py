import mlflow


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