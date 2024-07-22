import json
import random

all_dev_data_path = "./dev_data/dev.json"
with open(all_dev_data_path, "r", encoding="UTF-8") as f:
    all_dev_data = json.load(f)

all_formula_1_data = []
for data in all_dev_data:
    if data["db_id"] != "formula_1":
        continue
    all_formula_1_data.append(data)

formula_1_test_data_path = "./formula_1_data/test.json"
with open(formula_1_test_data_path, "r", encoding="UTF-8") as f:
    formula_1_test_data = json.load(f)

test_question_ids = [data["question_id"] for data in formula_1_test_data]

# if any item in all_dev_data has the same question_id, delete that item from all_dev_data
formula_1_train_val_data = [data for data in all_formula_1_data if data["question_id"] not in test_question_ids]

# randomly sample 50% of the data for training and 50% for validation
random.seed(42)
random.shuffle(formula_1_train_val_data)
num_train_val_data = len(formula_1_train_val_data) // 2
formula_1_train_data = formula_1_train_val_data[:num_train_val_data]  # 50%
formula_1_val_data = formula_1_train_val_data[num_train_val_data:]  # 50%
print("num_train_val_data:", num_train_val_data)
print("num_train_data:", len(formula_1_train_data))
print("num_val_data:", len(formula_1_val_data))

# save the data to json files
formula_1_train_data_path = "./formula_1_data/train.json"
with open(formula_1_train_data_path, "w", encoding="UTF-8") as f:
    json.dump(formula_1_train_data, f, indent=4)
formula_1_val_data_path = "./formula_1_data/val.json"
with open(formula_1_val_data_path, "w", encoding="UTF-8") as f:
    json.dump(formula_1_val_data, f, indent=4)
