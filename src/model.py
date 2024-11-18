from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model("model.sbm")
