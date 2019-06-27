from pandas import read_csv
from nltk import edit_distance
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys


def main():
    test_file = sys.argv[1]
    output_file = sys.argv[2]
    THRESHOLD = 20.7782

    # Testing data
    df_test = read_csv(test_file)
    df_test['title1_zh'] = df_test['title1_zh'].fillna('')
    df_test['title2_zh'] = df_test['title2_zh'].fillna('')

    X_1_test = df_test['title1_zh'].tolist()
    X_2_test = df_test['title2_zh'].tolist()
    id_test = df_test['id'].tolist()

    # Predict
    y_pred = []
    for i in range(len(X_1_test)):
        dist = edit_distance(X_1_test[i], X_2_test[i])
        if dist > THRESHOLD:
            y_pred.append('unrelated')
        else:
            y_pred.append('agreed')

    # Export
    with open(output_file, 'w') as file:
        file.write('Id,Category\n')
        for i in range(len(y_pred)):
            file.write(f'{id_test[i]},{y_pred[i]}\n')


if __name__ == "__main__":
    main()
