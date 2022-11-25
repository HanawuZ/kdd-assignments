from classes.airlineSatisfactionClassifier import airlineSatisfactionClassifier
import pandas as pd
import pathlib
# Define train and test dataset path.
AIRLINE_SATIFICATION_TRAIN_DATA_PATH = pathlib.Path(__file__).parent / "datasets" / "train.csv"
AIRLINE_SATIFICATION_TEST_DATA_PATH = pathlib.Path(__file__).parent / "datasets" / "test.csv"

# Read csv.
airlineSatisficationTrain = pd.read_csv(AIRLINE_SATIFICATION_TRAIN_DATA_PATH, index_col=[0])
airlineSatisficationTest = pd.read_csv(AIRLINE_SATIFICATION_TEST_DATA_PATH, index_col=[0])

# Initialize airline satisfication classifier object.
airlineSatisfactionClf = airlineSatisfactionClassifier(airlineSatisficationTrain, airlineSatisficationTest)

# Execute data preparation.
airlineSatisfactionClf.dataPreparation()

# Execute data preprocessing.
airlineSatisfactionClf.dataPreprocessing() 

# Show dataframe's information.
# airlineSatisficationClf.getAirlineSatificationInfo()

# Sample prepared dataframe.
# airlineSatisficationClf.sampleDatas(10)

airlineSatisfactionClf.splitValidation()

airlineSatisfactionClf.modelInit()

airlineSatisfactionClf.trainModel()

airlineSatisfactionClf.predictedTestSet()