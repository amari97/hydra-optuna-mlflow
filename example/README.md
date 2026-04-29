# Example

Run a simple quadratic minimization with Hydra multirun and the MLflow Optuna sweeper:

```bash
python example/quadratic.py -m 'x=interval(-5.0, 5.0)' 'y=interval(0.0, 10.0)'
```

To enable MLflow study run creation, set these values in `example/conf/config.yaml`:

- `use_mlflow: true`
- `trainer.logger.tracking_uri: <your-mlflow-uri>`
- `trainer.logger.experiment_name: <your-experiment-name>`
