import logging
import sys
import pandas as pd

from sklearn.preprocessing import PowerTransformer

from log_code import setup_logging

logger = setup_logging("yeo_timing")


def variable_transformation_outliers(
    X_train,
    X_test
):

    try:

        transformer = PowerTransformer(
            method="yeo-johnson"
        )

        transformer.fit(
            X_train
        )

        X_train_transformed = transformer.transform(
            X_train
        )

        X_test_transformed = transformer.transform(
            X_test
        )

        X_train_transformed = pd.DataFrame(
            X_train_transformed,
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_transformed = pd.DataFrame(
            X_test_transformed,
            columns=X_test.columns,
            index=X_test.index
        )

        logger.info(
            "Yeo-Johnson transformation completed"
        )

        return (
            X_train_transformed,
            X_test_transformed
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.error(
            f"Error in line no : {er_line.tb_lineno} "
            f": due to : {er_type} "
            f": reason : {er_msg}"
        )

        return X_train, X_test