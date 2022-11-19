from sklearn import metrics
import numpy as np
import statsmodels.api as sm

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

# * Function for show regression equation of cereals rating prediction model.
def showRegressionEquation(cereals_rating_predicted_model, cereals_feature_labels):
    
    # Get intercept value of prediction model.

    # Get array of coefficent value of prediction model.

    # Zip model coefficient with cereals feature names.
    
    # Iteration for creating regression equation as string.
    
    # Print regression equation.
    return None


# Guide : https://datatofish.com/statsmodels-linear-regression/
# * Function for show regression statistics
def showRegressionStats(cereals_feature_train, cereals_feature_test, cereals_rating_train, cereals_rating_test):
    # Passing dataframes.

    # Set independenct variable(s).
    
    # Set dependent variable. (`rating`)

    # Add constant 
    cereals_feature_train = sm.add_constant(cereals_feature_train)
    cereals_feature_test = sm.add_constant(cereals_feature_test)

    # Create regression model with OLS class from statmodel.api
    model = sm.OLS(cereals_rating_train, cereals_feature_train).fit()
    
    # Predicted model
    prediction = model.predict(cereals_feature_test)

    # Display model Summary
    print(model.summary())
    pass