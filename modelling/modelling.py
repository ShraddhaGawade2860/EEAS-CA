import os
from model.randomforest import RandomForest
from embeddings import get_tfidf_embd
from Config import Config

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_type2(df):
    print("\n[LEVEL 1] Training Type 2...")
    X = get_tfidf_embd(df)
    data = Data(X, df, Config.TYPE2)
    if data.should_skip():
        print("Skipping Type2 due to low class count")
        return None

    model = RandomForest("Type2", *data.get_all_data())
    model.train_data(*data.get_train_data())
    preds = model.predict_data(data.X)

    model.print_results(data.y)
    df.loc[data.X_train.shape[0]*[True] + data.X_test.shape[0]*[False], "PRED_TYPE2"] = preds[:data.X_train.shape[0]]
    df.loc[data.X_train.shape[0]*[False] + data.X_test.shape[0]*[True], "PRED_TYPE2"] = preds[data.X_train.shape[0]:]

    # Save model & predictions
    model.save_model(f"{MODEL_DIR}/rf_type2.pkl")
    df.to_csv("type2_predictions.csv", index=False)

    return df


def train_type3(df):
    for type2_class in df["PRED_TYPE2"].dropna().unique():
        subset = df[df["PRED_TYPE2"] == type2_class].copy()
        print(f"\n[LEVEL 2] Training Type 3 for Type2 = '{type2_class}' (samples={len(subset)})")
        X = get_tfidf_embd(subset)
        data = Data(X, subset, Config.TYPE3)
        if data.should_skip():
            print(f"Skipping T3 for Type2={type2_class}")
            continue

        model = RandomForest(f"T3_{type2_class}", *data.get_all_data())
        model.train_data(*data.get_train_data())
        preds = model.predict_data(data.X)

        model.print_results(data.y)
        subset.loc[:, "PRED_TYPE3"] = preds

        model.save_model(f"{MODEL_DIR}/rf_type3_{type2_class}.pkl")
        subset.to_csv(f"type3_predictions_{type2_class}.csv", index=False)

        yield type2_class, subset


def train_type4(df_by_t2_class, t2_class):
    for t3_class in df_by_t2_class["PRED_TYPE3"].dropna().unique():
        subset = df_by_t2_class[df_by_t2_class["PRED_TYPE3"] == t3_class].copy()
        print(f"\n[LEVEL 3] Training Type 4 for T2 = '{t2_class}', T3 = '{t3_class}' (samples={len(subset)})")
        X = get_tfidf_embd(subset)
        data = Data(X, subset, Config.TYPE4)
        if data.should_skip():
            print(f"Skipping T4 for T2={t2_class}, T3={t3_class}")
            continue

        model = RandomForest(f"T4_{t2_class}_{t3_class}", *data.get_all_data())
        model.train_data(*data.get_train_data())
        preds = model.predict_data(data.X)

        model.print_results(data.y)
        subset.loc[:, "PRED_TYPE4"] = preds

        model.save_model(f"{MODEL_DIR}/rf_type4_{t2_class}_{t3_class}.pkl")
        subset.to_csv(f"type4_predictions_{t2_class}_{t3_class}.csv", index=False)