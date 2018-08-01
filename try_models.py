import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def try_regression(
        X,
        y,
        pipeline=None,
        regressor=None,
        predictions=False,
        cross_val=True,
        regressor_params=None,
        cross_val_params=None
):
    if not cross_val_params:
        cross_val_params = {
            'scoring': 'neg_mean_squared_error',
            'cv': 10
        }

    if not pipeline:
        transformator = None
    else:
        transformator = ("preparation", pipeline)

    if not regressor:
        predictor = None
    else:
        predictor = (
            "regressor_name",
            regressor(**regressor_params)
        )

    full_pipeline_with_predictor = Pipeline([
        transformator,
        regressor
    ])

    if predictions:
        model = full_pipeline_with_predictor.fit(X, y)
        y_predictions = full_pipeline_with_predictor.predict(X)

        mse = mean_squared_error(y, y_predictions)
        rmse = np.sqrt(mse)
        print("""
Predictions:   {}
RMSE:          {}
              """.format(y_predictions, rmse))

    if cross_val:
        scores = cross_val_score(
            full_pipeline_with_predictor,
            X=X,
            y=y,
            **cross_val_params
        )
        scores = np.sqrt(-scores)
        print(
            """
            CROSS_VAL_SCORES:
            
            SUM:     {}
            Mean:    {}
            STD:     {}
            """.format(
            scores,
            scores.mean(),
            scores.std()
        ))
