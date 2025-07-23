import pandas as pd
import numpy as np

def effective(input):

    input1 = input.RESULTS
    input2 = input.MOVE_LIST
    list_days = {}
    list_days_sep = {}
    day = 1

    for results_key_1 in input1:
        list = []
        list_sep = {'OPEN': [], 'POWER': [], 'TRIPOD': [], 'KEY': [], 'PINCH': [], 'INDEX': []}
        move_td = input2[day]
        if type(results_key_1) == int:
            VAL = input1[results_key_1]
            item = 0
            for results_key_2 in input1[results_key_1]:
                if type(results_key_2) == int:
                    results_key = VAL[results_key_2]['CLASSIFICATION_VALUES']

                    counts = pd.Series(results_key).value_counts()
                    get_counts = counts.get(move_td[item])
                    if get_counts is None:
                        get_counts = 0

                    # Finding the number of rest and wrist movement in the data, taking them out because we want the effective movement of the hand movement

                    get_rest = counts.get("REST")
                    if get_rest is None:
                        get_rest = 0

                    get_pd = counts.get("PALM DOWN")
                    if get_pd is None:
                        get_pd = 0

                    get_pu = counts.get("PALM UP")
                    if get_pu is None:
                        get_pu = 0

                    total_redundant = get_rest+get_pd+get_pu
                    total_counts = len(results_key)
                    total_effective = total_counts-total_redundant
                    print(total_effective)
                    effective = get_counts/total_effective

                    list.append(effective)
                    list_sep[move_td[item]].append(effective)
                    item +=1

            list_days[day] = list
            list_days_sep[day] = list_sep
            day += 1

    # Calculating average
    list_avgs = {}
    list_avgs_sep = {}

    for things in list_days:
        list_avgs[things] = np.mean(list_days[things])
    for things in list_days_sep:
        list_sep = {'OPEN': 0, 'POWER': 0, 'TRIPOD': 0, 'KEY': 0, 'PINCH': 0, 'INDEX': 0}
        for move in list_days_sep[things]:
            if len(list_days_sep[things][move]) == 0:
                list_sep[move] = 0
            else:
                list_sep[move] = np.mean(list_days_sep[things][move])
        list_avgs_sep[things] = list_sep
    return list_avgs, list_avgs_sep
