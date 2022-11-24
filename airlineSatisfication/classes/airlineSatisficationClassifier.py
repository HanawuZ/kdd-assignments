import pandas as pd

class airlineSatificationClassifier:
    def __init__(self, airline_satification_train_df, airline_satisfication_test_df):
        self.airline_satification_df = pd.concat([airline_satification_train_df, airline_satisfication_test_df])

    def getAirlineSatificationInfo(self):
        print(self.airline_satification_df.info())

    def dataPreparation(self):
        # Exclude unique attributes e.g. `id`
        # Exclude missing columns.
        # Exclude attributes `Age` and `Gender`.
        # Reset index.
        pass

    def dataPreprocessing(self):
        # Remove records that have missing value of `Arrival Delay in Minute`.
        # Encode string attributes to polynomial values.
        pass

    def splitValidation(self):
        # Split dataframe into train & test sets.
        pass
    
    def naiveBayesInit(self):
        # Initialize Naive Bayes model.
        pass

    def decisionTreeInit(self):
        # Initialize Decision Tree model.
        pass

    def trainModel(self):
        # Train model.
        pass

    def predictedTestSet(self):
        # Predict test dataframe.
        pass

    
    