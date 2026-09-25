'''
In this file we are going to load the data and pass the data
to required functions for training the ML Algorithm.
'''

import sys
import warnings
import pickle
import pandas as pd

warnings.filterwarnings("ignore")

from log_code import setup_logging

logger = setup_logging("main")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from yeo_timing import variable_transformation_outliers
from fs import select_best_columns

from imblearn.over_sampling import SMOTE

from all_models import common


class HEART_DISEASE_PROJECT:

    def __init__(self, path):

        try:

            # Load dataset
            self.path = path

            self.df = pd.read_csv(
                self.path
            )

            logger.info(
                f"The Number of rows and columns was : "
                f"{self.df.shape}"
            )

            # Check null values
            logger.info(
                f"Null values in the dataset : "
                f"{self.df.isnull().sum()}"
            )

            # Independent variables
            self.X = self.df.iloc[:, :-1]

            # Dependent variable
            self.y = self.df.iloc[:, -1]

            logger.info(
                f"Independent columns : "
                f"{self.X.columns.tolist()}"
            )

            logger.info(
                f"Dependent column : "
                f"{self.y.name}"
            )

            # Train Test Split
            (
                self.X_train,
                self.X_test,
                self.y_train,
                self.y_test
            ) = train_test_split(

                self.X,
                self.y,

                test_size=0.20,

                random_state=42,

                stratify=self.y
            )

            logger.info(
                f"Training dataset size : "
                f"{self.X_train.shape} => "
                f"{self.y_train.shape}"
            )

            logger.info(
                f"Testing dataset size : "
                f"{self.X_test.shape} => "
                f"{self.y_test.shape}"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # SPLIT NUMERICAL DATA
    # --------------------------------------------------

    def spliting_data_clearlly(self):

        try:

            # All columns are numerical
            self.X_train_numerical = self.X_train.copy()

            self.X_test_numerical = self.X_test.copy()

            logger.info(
                f"Train Numerical columns : "
                f"{self.X_train_numerical.shape} : "
                f"{self.X_train_numerical.columns.tolist()}"
            )

            logger.info(
                f"Test Numerical columns : "
                f"{self.X_test_numerical.shape} : "
                f"{self.X_test_numerical.columns.tolist()}"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # YEO-JOHNSON TRANSFORMATION
    # --------------------------------------------------

    def vt_outliers(self):

        try:

            (
                self.X_train_numerical,
                self.X_test_numerical
            ) = variable_transformation_outliers(

                self.X_train_numerical,

                self.X_test_numerical
            )

            logger.info(
                "Yeo-Johnson transformation completed"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # FEATURE SELECTION
    # --------------------------------------------------

    def feature_selection(self):

        try:

            (
                self.X_train_numerical,
                self.X_test_numerical
            ) = select_best_columns(

                self.X_train_numerical,

                self.X_test_numerical,

                self.y_train,

                self.y_test
            )

            logger.info(
                f"Feature selection completed : "
                f"{self.X_train_numerical.shape}"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # CATEGORICAL TO NUMERICAL
    # --------------------------------------------------

    def cat_to_numerical(self):

        try:

            # All columns are already numerical.
            # Therefore encoding is not required.

            self.final_training_data = \
                self.X_train_numerical.copy()

            self.final_testing_data = \
                self.X_test_numerical.copy()

            logger.info(
                f"Final Training Data : "
                f"{self.final_training_data.shape}"
            )

            logger.info(
                f"Final Testing Data : "
                f"{self.final_testing_data.shape}"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # DATA BALANCING + STANDARD SCALING
    # --------------------------------------------------

    def data_balancing(self):

        try:

            logger.info(
                f"Total Final Training data shape : "
                f"{self.y_train.shape}"
            )

            logger.info(
                f"Number of Rows for Target : 0 : "
                f"Class : {sum(self.y_train == 0)}"
            )

            logger.info(
                f"Number of Rows for Target : 1 : "
                f"Class : {sum(self.y_train == 1)}"
            )

            # SMOTE
            sm_obj = SMOTE(
                random_state=42
            )

            (
                self.X_train_bal,
                self.y_train_bal
            ) = sm_obj.fit_resample(

                self.final_training_data,

                self.y_train
            )

            logger.info(
                f"After Balancing Training data shape : "
                f"{self.y_train_bal.shape}"
            )

            logger.info(
                f"Number of Rows for Target : 0 : "
                f"Class : {sum(self.y_train_bal == 0)}"
            )

            logger.info(
                f"Number of Rows for Target : 1 : "
                f"Class : {sum(self.y_train_bal == 1)}"
            )

            # Standard Scaling
            sc = StandardScaler()

            sc.fit(
                self.X_train_bal
            )

            # Save scaler
            with open(
                "scaled_model.pkl",
                "wb"
            ) as f:

                pickle.dump(
                    sc,
                    f
                )

            logger.info(
                "Scaler saved as scaled_model.pkl"
            )

            # Transform training data
            self.X_train_bal_scaled = \
                sc.transform(
                    self.X_train_bal
                )

            # Transform testing data
            self.X_test_scaled = \
                sc.transform(
                    self.final_testing_data
                )

            logger.info(
                "Standard scaling completed"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in line no : {er_line.tb_lineno} "
                f": due to : {er_type} "
                f": reason : {er_msg}"
            )


    # --------------------------------------------------
    # TRAIN ALL MODELS
    # --------------------------------------------------

    def train_all_models(self):

        try:

            common(

                self.X_train_bal_scaled,

                self.y_train_bal,

                self.X_test_scaled,

                self.y_test
            )

            logger.info(
                "All models training completed"
            )

        except Exception as e:

            er_type, er_msg, er_line = sys.exc_info()

            logger.error(
                f"Error in model training : "
                f"{er_line.tb_lineno} "
                f": due to : {e}"
            )


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    try:

        obj = HEART_DISEASE_PROJECT(
            "heart.csv"
        )

        obj.spliting_data_clearlly()

        obj.vt_outliers()

        obj.feature_selection()

        obj.cat_to_numerical()

        obj.data_balancing()

        obj.train_all_models()

        logger.info(
            "Heart Disease ML Pipeline completed successfully"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.error(
            f"Error in line no : {er_line.tb_lineno} "
            f": due to : {er_type} "
            f": reason : {er_msg}"
        )