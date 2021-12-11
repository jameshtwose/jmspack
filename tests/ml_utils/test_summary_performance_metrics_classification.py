import pytest
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from jmspack.ml_utils import summary_performance_metrics_classification


@pytest.fixture
def df_iris():
    """Example data set: iris."""
    df = sns.load_dataset("iris")
    df = df[df["species"].isin(["virginica", "versicolor"])].assign(
        species=lambda d: d["species"].astype("category").cat.codes
    )
    return df


class TestPerformanceMetricClassification:
    """Test for the function summary_performance_metrics_classification"""

    def test_returns_expected_values(self, df_iris):
        X = df_iris.drop("species", axis=1)
        y = df_iris["species"]

        clf = LogisticRegression().fit(X, y)

        summary_df = summary_performance_metrics_classification(
            model=clf, X_test=X, y_true=y, bootstraps=100, fold_size=1000
        )

        assert summary_df["TN"].iloc[0] == pytest.approx(47)
        assert summary_df["FP"].iloc[0] == pytest.approx(3)
        assert summary_df["FN"].iloc[0] == pytest.approx(1)
        assert summary_df["TP"].iloc[0] == pytest.approx(49)
        assert summary_df["Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Balanced Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Prevalence"].iloc[0] == pytest.approx(0.5)
        assert summary_df["Sensitivity"].iloc[0] == pytest.approx(0.98)
        assert summary_df["Specificity"].iloc[0] == pytest.approx(0.94)
        assert summary_df["PPV"].iloc[0] == pytest.approx(0.942)
        assert summary_df["NPV"].iloc[0] == pytest.approx(0.979)
        assert summary_df["auc"].iloc[0] == pytest.approx(0.995)
        assert (
            summary_df["Mean AUC (CI 5%-95%)"].iloc[0] == "0.997 (95% CI 0.997-0.997)"
        )
        assert summary_df["F1"].iloc[0] == pytest.approx(0.961)

    def test_returns_expected_values_SVC_probability_False(self, df_iris):
        X = df_iris.drop("species", axis=1)
        y = df_iris["species"]

        clf = SVC(probability=False).fit(X, y)

        summary_df = summary_performance_metrics_classification(
            model=clf, X_test=X, y_true=y, bootstraps=100, fold_size=1000
        )

        assert summary_df["TN"].iloc[0] == pytest.approx(48)
        assert summary_df["FP"].iloc[0] == pytest.approx(2)
        assert summary_df["FN"].iloc[0] == pytest.approx(2)
        assert summary_df["TP"].iloc[0] == pytest.approx(48)
        assert summary_df["Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Balanced Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Prevalence"].iloc[0] == pytest.approx(0.5)
        assert summary_df["Sensitivity"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Specificity"].iloc[0] == pytest.approx(0.96)
        assert summary_df["PPV"].iloc[0] == pytest.approx(0.96)
        assert summary_df["NPV"].iloc[0] == pytest.approx(0.96)
        assert summary_df["auc"].iloc[0] == pytest.approx(0.96)
        assert (
            summary_df["Mean AUC (CI 5%-95%)"].iloc[0] == "0.970 (95% CI 0.970-0.970)"
        )
        assert summary_df["F1"].iloc[0] == pytest.approx(0.96)

    def test_returns_expected_values_SVC_probability_True(self, df_iris):
        X = df_iris.drop("species", axis=1)
        y = df_iris["species"]

        clf = SVC(probability=True).fit(X, y)

        summary_df = summary_performance_metrics_classification(
            model=clf, X_test=X, y_true=y, bootstraps=100, fold_size=1000
        )

        assert summary_df["TN"].iloc[0] == pytest.approx(48)
        assert summary_df["FP"].iloc[0] == pytest.approx(2)
        assert summary_df["FN"].iloc[0] == pytest.approx(2)
        assert summary_df["TP"].iloc[0] == pytest.approx(48)
        assert summary_df["Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Balanced Accuracy"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Prevalence"].iloc[0] == pytest.approx(0.5)
        assert summary_df["Sensitivity"].iloc[0] == pytest.approx(0.96)
        assert summary_df["Specificity"].iloc[0] == pytest.approx(0.96)
        assert summary_df["PPV"].iloc[0] == pytest.approx(0.96)
        assert summary_df["NPV"].iloc[0] == pytest.approx(0.96)
        assert summary_df["auc"].iloc[0] == pytest.approx(0.995)
        assert (
            summary_df["Mean AUC (CI 5%-95%)"].iloc[0] == "0.997 (95% CI 0.997-0.997)"
        )
        assert summary_df["F1"].iloc[0] == pytest.approx(0.96)

    def test_warning_predict_proba(self, df_iris):
        X = df_iris.drop("species", axis=1)
        y = df_iris["species"]

        clf = SVC(probability=False).fit(X, y)

        with pytest.warns(UserWarning):
            _ = summary_performance_metrics_classification(
                model=clf, X_test=X, y_true=y, bootstraps=100, fold_size=1000
            )
