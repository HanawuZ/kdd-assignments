import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
from modules.shoppingCartDataPreprocessing import *

# Random Shopping Cart file path.
SHOPPING_CART_PATH = pathlib.Path(__file__).parent / "datasets" / "Random_shopping_cart.csv"

shoppingCartData = pd.read_csv(SHOPPING_CART_PATH, header=None)

def dataPreprocessing(shoppingCartData):
    """
    ### Function for prepare datas
    - Remove missing value, which is 0 and NaN.
    - Transform all records into list of transaction.
    """
    transaction = []
    # Iterate rows.
    for row in range(shoppingCartData.shape[0]):
        transaction.append([str(shoppingCartData.values[row,col]) for col in range(shoppingCartData.shape[1])])
    print(transaction[3])

#* Function for applying association rule.
def applyAssociationRule(transaction):
    """
    Function for applying association rule by passing transaction list to appiori algorithm.
    """
    pass
dataPreprocessing(shoppingCartData)
