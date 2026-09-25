import logging
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd

from log_code import setup_logging

logger = setup_logging("fs")


def select_best_columns(X_train, X_test, y_train, y_test):

    try:

        selector = SelectKBest(
            score_func=f_classif,
            k="all"
        )

        selector.fit(
            X_train,
            y_train
        )

        X_train_selected = selector.transform(
            X_train
        )

        X_test_selected = selector.transform(
            X_test
        )

        selected_columns = X_train.columns[
            selector.get_support()
        ]

        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=selected_columns,
            index=X_train.index
        )

        X_test_selected = pd.DataFrame(
            X_test_selected,
            columns=selected_columns,
            index=X_test.index
        )

        logger.info(
            f"Selected Features : "
            f"{list(selected_columns)}"
        )

        return (
            X_train_selected,
            X_test_selected
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.error(
            f"Error in line no : {er_line.tb_lineno} "
            f": due to : {er_type} "
            f": reason : {er_msg}"
        )

        return X_train, X_test