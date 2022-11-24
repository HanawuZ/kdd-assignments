from classes.airlineSatisficationClassifier import airlineSatificationClassifier
import pandas as pd
import pathlib
# Define train and test dataset path.
AIRLINE_SATIFICATION_TRAIN_DATA_PATH = pathlib.Path(__file__).parent / "datasets" / "train.csv"
AIRLINE_SATIFICATION_TEST_DATA_PATH = pathlib.Path(__file__).parent / "datasets" / "test.csv"

# Read csv.
airlineSatisficationTrain = pd.read_csv(AIRLINE_SATIFICATION_TRAIN_DATA_PATH)
airlineSatisficationTest = pd.read_csv(AIRLINE_SATIFICATION_TEST_DATA_PATH)

# Initialize airline satisfication classifier object.
airlineSatisficationClf = airlineSatificationClassifier(airlineSatisficationTrain, airlineSatisficationTest)

# Show dataframe's information.
airlineSatisficationClf.getAirlineSatificationInfo()