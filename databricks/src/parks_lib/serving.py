"""MLflow wrapper so the registered model takes the build_frame columns and returns clipped scores."""

import mlflow.pyfunc

from parks_lib.forecast import predict


class CrowdForecastModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict(self, context, model_input, params=None):
        # park_id arrives as an integer (MLflow schemas have no category type); LightGBM maps the
        # categories back to the ones it was trained on
        frame = model_input.copy()
        frame["park_id"] = frame["park_id"].astype(int).astype("category")
        return predict(self.model, frame, self.features)
