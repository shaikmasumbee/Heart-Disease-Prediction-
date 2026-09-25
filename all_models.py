import pickle
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

from log_code import setup_logging


logger = setup_logging("all_models")


def common(
    X_train,
    y_train,
    X_test,
    y_test
):

    try:

        models = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=1000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    random_state=42
                ),

            "KNN":
                KNeighborsClassifier(),

            "SVM":
                SVC(),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                )
        }

        # Store ROC values
        roc_data = {}

        # Train and evaluate all models
        for name, model in models.items():

            logger.info(
                "=" * 50
            )

            logger.info(
                f"Model : {name}"
            )

            logger.info(
                "=" * 50
            )

            # Training
            model.fit(
                X_train,
                y_train
            )

            logger.info(
                f"{name} training completed"
            )

            # Prediction
            y_pred = model.predict(
                X_test
            )

            # Accuracy
            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            # Precision
            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            # Recall
            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            # F1 Score
            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            # Confusion Matrix
            cm = confusion_matrix(
                y_test,
                y_pred
            )

            # ROC-AUC score
            if isinstance(model, SVC):

                y_score = model.decision_function(
                    X_test
                )

            else:

                y_score = model.predict_proba(
                    X_test
                )[:, 1]

            auc = roc_auc_score(
                y_test,
                y_score
            )

            # ROC curve values
            fpr, tpr, _ = roc_curve(
                y_test,
                y_score
            )

            roc_data[name] = (
                fpr,
                tpr,
                auc
            )

            # Log results
            logger.info(
                f"Accuracy : {accuracy}"
            )

            logger.info(
                f"Precision : {precision}"
            )

            logger.info(
                f"Recall : {recall}"
            )

            logger.info(
                f"F1 Score : {f1}"
            )

            logger.info(
                f"ROC-AUC : {auc}"
            )

            logger.info(
                f"Confusion Matrix :\n{cm}"
            )

        # ------------------------------------------------
        # ROC CURVE
        # ------------------------------------------------

        plt.figure(
            figsize=(10, 7)
        )

        for name, values in roc_data.items():

            fpr = values[0]
            tpr = values[1]
            auc = values[2]

            plt.plot(
                fpr,
                tpr,
                label=f"{name} (AUC = {auc:.3f})"
            )

        # Random classifier line
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            "ROC Curve - Heart Disease Classification"
        )

        plt.legend()

        plt.grid()

        # Save ROC figure
        plt.savefig(
            "roc_curve.png"
        )

        logger.info(
            "ROC curve saved as roc_curve.png"
        )

        # Show ROC figure
        plt.show()

        # ------------------------------------------------
        # FINAL MODEL
        # ------------------------------------------------

        logger.info(
            "=" * 50
        )

        logger.info(
            "FINAL MODEL"
        )

        logger.info(
            "=" * 50
        )

        # Logistic Regression is selected
        # based on the ROC-AUC comparison
        final_model = LogisticRegression(
            max_iter=1000
        )

        final_model.fit(
            X_train,
            y_train
        )

        logger.info(
            "Final model training completed"
        )

        # Save final model
        with open(
            "Heart_Disease_Model.pkl",
            "wb"
        ) as f:

            pickle.dump(
                final_model,
                f
            )

        logger.info(
            "Final Model : Logistic Regression"
        )

        logger.info(
            "Model saved as : Heart_Disease_Model.pkl"
        )

    except Exception as e:

        er_type, er_msg, er_line = sys.exc_info()

        logger.error(
            f"Error in line no : {er_line.tb_lineno} "
            f": due to : {er_type} "
            f": reason : {er_msg}"
        )