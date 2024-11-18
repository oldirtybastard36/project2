from catboost import CatBoostClassifier, Pool
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from clearml import Task
from src.utils import clean_text

task = Task.init(project_name="toxic_model", task_name="Training", tags=["catboost"])

dataset = load_dataset(
    "AlexSham/Toxic_Russian_Comments",
)


def prepare_function(example):
    example["text"] = clean_text(example["text"])
    return example


train_dataset = dataset["train"].map(prepare_function)
test_dataset = dataset["test"].map(prepare_function)

train_pool = Pool(
    train_dataset["text"], train_dataset["label"], text_features=[0]
)  # Указываем текстовую колонку
test_pool = Pool(test_dataset["text"], test_dataset["label"], text_features=[0])

model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.01,
    depth=6,
    verbose=50,
    task_type="CPU",
    random_seed=63,
    loss_function="Logloss",
)


model.fit(train_pool, eval_set=test_pool, use_best_model=True)


y_pred = model.predict(test_pool)

print("Accuracy:", accuracy_score(test_dataset["label"], y_pred))
print("Classification Report:")
print(classification_report(test_dataset["label"], y_pred))

model.save_model("model.sbm")
