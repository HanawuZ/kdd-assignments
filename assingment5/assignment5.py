import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
from modules.shoppingCartDataPreprocessing import *
from modules.applyAssociationRule import *
# Define Random shopping cart file's path.
SHOPPING_CART_PATH = pathlib.Path(__file__).parent / "datasets" / "Random_shopping_cart.csv"

shoppingCartData = pd.read_csv(SHOPPING_CART_PATH, header=None)

shopping_transaction = encodeTransaction(dataPreprocessing(shoppingCartData))
applyAssociationRule(shopping_transaction)