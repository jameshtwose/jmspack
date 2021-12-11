# import matplotlib
# import matplotlib.pyplot as plt
# import pytest
# import seaborn as sns
#
# from jmspack.ml_utils import plot_cv_indices
#
#
# @pytest.fixture
# def df_test():
#     """test dataset from seaborn"""
#     return sns.load_dataset("iris")
#
#
# class TestPlotCvIndices:
#     """Testing class to test the plot_decision_boundary function."""
#
#     def test_if_dataframe_not_affected(self, df_test):
#         """Check if the function leaves the data frame the same."""
#         feature_list = df_test.columns.tolist()[0:2]
#         target = "species"
#         X = df_test[feature_list]
#         y = df_test[target]
#         X_original = X.copy()
#         y_original = y.copy()
#         _, _ = plot_decision_boundary(X=X, y=y)
#         assert X_original.equals(X)
#         assert y_original.equals(y)
#
#     def test_returns_expected_objects(self, df_test):
#         """Check if the function returns the expected output objects."""
#         feature_list = df_test.columns.tolist()[0:2]
#         target = "species"
#         X = df_test[feature_list]
#         y = df_test[target]
#         fig, ax = plot_decision_boundary(X=X, y=y)
#
#         assert isinstance(fig, matplotlib.figure.Figure)
#         assert isinstance(ax, matplotlib.axes.Axes)
#
#     def test_ax_attributes(self, df_test):
#         """Check if the function outputs the expected axis object."""
#         feature_list = df_test.columns.tolist()[0:2]
#         target = "species"
#         X = df_test[feature_list]
#         y = df_test[target]
#         fig, ax = plot_decision_boundary(X=X, y=y,
#                                          clf = LogisticRegression(),
#                                          title = 'Decision Boundary Logistic Regression',
#                                          legend_title = 'Legend',
#                                          h = 0.05,
#                                          figsize = (11.7, 8.27))
#         _ = plt.show(block=False)
#
#         test_y_ticklabels = ["Text(0, 1.0, '1.0')", "Text(0, 1.5, '1.5')", "Text(0, 2.0, '2.0')"]
#         test_x_ticklabels = ["Text(3.0, 0, '3')", "Text(4.0, 0, '4')", "Text(5.0, 0, '5')"]
#         ax_y_ticklabels = [str(x) for x in list(ax.get_yticklabels()[:3])]
#         ax_x_ticklabels = [str(x) for x in list(ax.get_xticklabels()[:3])]
#
#         assert ax_x_ticklabels == test_x_ticklabels
#         assert ax_y_ticklabels == test_y_ticklabels
#
#         _ = plt.pause(1)
#         _ = plt.close(fig=fig)
#
#     # for this test to run you will need to make sure that
#     # pytest-mpl is installed and that you have the baseline images
#     # by running `pytest --mpl-generate-path=baseline` in the same
#     # directory as the test files
#     @pytest.mark.mpl_image_compare
#     def test_plot_decision_boundary(self, df_test):
#         """Check if the function returns the expected figure (compared to a baseline plot)."""
#         feature_list = df_test.columns.tolist()[0:2]
#         target = "species"
#         X = df_test[feature_list]
#         y = df_test[target]
#         fig, ax = plot_decision_boundary(X=X, y=y,
#                                          clf=LogisticRegression(),
#                                          title='Decision Boundary Logistic Regression',
#                                          legend_title='Legend',
#                                          h=0.05,
#                                          figsize=(11.7, 8.27))
#         return fig
#
