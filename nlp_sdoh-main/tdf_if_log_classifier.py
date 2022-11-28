import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_data_split(input_df, num_val, num_test):
    random.seed(0)
    uniq_ids = {}
    for index, row in input_df.iterrows():
        outcome, uniq_id = row["outcome"], row["uniq_id"]
        uniq_ids[uniq_id] = outcome
    true_uniq_ids = [uniq_id for uniq_id in uniq_ids if uniq_ids[uniq_id]]
    false_uniq_ids = [uniq_id for uniq_id in uniq_ids if not uniq_ids[uniq_id]]
    num_each_label = int((num_val + num_test) / 2)
    val_test_true_uniq_ids = random.sample(true_uniq_ids, num_each_label)
    random.shuffle(val_test_true_uniq_ids)
    val_test_false_uniq_ids = random.sample(false_uniq_ids, num_each_label)
    random.shuffle(val_test_false_uniq_ids)
    val_test_split_index = int(num_val / 2)
    val_uniq_ids = val_test_true_uniq_ids[:val_test_split_index] + val_test_false_uniq_ids[:val_test_split_index]
    test_uniq_ids = val_test_true_uniq_ids[val_test_split_index:] + val_test_false_uniq_ids[val_test_split_index:]
    val_test_uniq_ids = set(val_uniq_ids + test_uniq_ids)
    train_uniq_ids = [uniq_id for uniq_id in uniq_ids if uniq_id not in val_test_uniq_ids]
    return (train_uniq_ids, [uniq_ids[uniq_id] for uniq_id in train_uniq_ids]), \
           (val_uniq_ids, [uniq_ids[uniq_id] for uniq_id in val_uniq_ids]),\
           (test_uniq_ids, [uniq_ids[uniq_id] for uniq_id in test_uniq_ids]),


def get_uniq_id_docs(input_df, uniq_ids):
    docs = list(input_df[input_df["uniq_id"].isin(uniq_ids)]["text"])
    return docs


def get_vectoriser_and_trian_vectors(input_df, train_ids):
    train_docs = get_uniq_id_docs(input_df, train_ids)
    vectoriser = TfidfVectorizer()
    train_vectors = vectoriser.fit_transform(train_docs)
    return vectoriser, train_vectors


def get_tdf_if_vectors(vectoriser, docs):
    tdf_if_vectors = vectoriser.transform(docs)
    return tdf_if_vectors


data_df = pd.read_csv("combined_data_v3.csv", usecols=["uniq_id", "outcome", "text"])
(train_ids, train_y), (val_ids, val_y), (test_ids, test_y) = get_data_split(data_df, 40, 20)
tdf_if_vectoriser, train_X = get_vectoriser_and_trian_vectors(data_df, train_ids)
tdf_if_clf = LogisticRegression().fit(train_X, train_y)
    
val_docs = get_uniq_id_docs(data_df, val_ids)
val_X = get_tdf_if_vectors(tdf_if_vectoriser, val_docs)
val_pred_y = tdf_if_clf.predict(val_X)
val_pred_proba_y = tdf_if_clf.predict_proba(val_X)

un, up, tp, fp, tn, fn = 0, 0, 0, 0, 0, 0
for val_id, y, pred_y, pred_proba_y in zip(val_ids, val_y, val_pred_y, val_pred_proba_y):
    print(f"{y}\t{pred_y}\t{pred_proba_y}")
    if max(pred_proba_y) < 0.7:
        if y:
            up += 1
        else:
            un += 1
    elif y:
        if pred_y:
            tp += 1
        else:
            fn += 1
    else:
        if pred_y:
            fp += 1
        else:
            tn += 1
print(f"up: {up}")
print(f"tp: {tp}")
print(f"fp: {fp}")
print(f"tn: {tn}")
print(f"fn: {fn}")

print(f"positive undetermined rate = {up/ (tp + fn + up)}")
print(f"negative undetermined rate = {un/ (tn + fp + un)}")
print(f"undetermined rate = {(un + up)/ (un + up + tp + fp + tn + fn)}")
print(f"positive predictive value = {tp/ (tp + fp)}")
print(f"negative predictive value = {tn/ (tn + fn)}")
print(f"sensitivity = {tp/ (tp + fn)}")
print(f"specificity = {tn/ (fp + tn)}")
