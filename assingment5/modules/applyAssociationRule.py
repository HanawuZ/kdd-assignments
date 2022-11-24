from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

#* Function for applying association rule.
def applyAssociationRule(shopping_transaction):
    """
    Function for applying association rule by passing transaction list to appiori algorithm.
    """

    # Iteration for testing association rules from interval values of mininum support
    # and minimum confidence in range 0.0-1.0

    """
    #?############### Pseudo Code ###############
    Iterate support value in range(0 to 1] incresed by 0.1
        Get frequent itemsets dataframe with current minimum support
        Iterate confidence value in range(0 to 1)
            Get association rule with current min confidence
            Show association rule 
    #?###########################################  
    """
    sup=0.1
    while sup <= 1.0:
        
        # Apply an apriori algorithm.
        frequent_itemsets = apriori(shopping_transaction, min_support=sup, use_colnames=True)
        
        # Continue the iteration if there're no freqently itemset
        if (len(frequent_itemsets.index) == 0):
            print("Dataframe is empty!")
            sup+=0.1
            break
        
        # The length column has been added to increase ease of filtering.
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

        # Create association rules.
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

        # Sort association rules by confidence descending
        rules = rules.sort_values("confidence",ascending=False)

        # Define minimum confidence.
        conf = 0.1

        # Iterate confidence value in range 0.0-1.0 incresed by 0.1
        while conf <= 1.0:
            print("Minimum Support = {} | Minimum Confidence = {}".format(sup,conf))
            print(rules[(rules['confidence'] > conf) & (rules['lift'] > 1) ])
            conf+=0.1        
            print("\n")
        sup+=0.1

        