from sklearn import metrics
import numpy as np

# * Function for display performance report of regression.
def regressionPerformanceReport(cereals_feature_test, cereals_rating_prediction_model, cereals_rating_test , cereals_predicted_rating):
    # Show performance vectors.
    # root mean squared error
    # absolute error
    # relative error
    # relative error lenient
    # relative error strict

    print("+"*8, "Performance Vector", "+"*8)
    print()
    print("Root mean squared error : ", np.sqrt(metrics.mean_squared_error(cereals_rating_test , cereals_predicted_rating)));
    print("Mean absolute error : ", metrics.mean_absolute_error(cereals_rating_test , cereals_predicted_rating));
    print("R2-Score : ",cereals_rating_prediction_model.score(cereals_feature_test , cereals_rating_test));



# Guide : https://datatofish.com/statsmodels-linear-regression/
