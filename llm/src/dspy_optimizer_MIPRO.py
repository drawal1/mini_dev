import json
import os
import sys
import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2

from dspy_sig_and_model import SqlErrorCorrectionSig, Text2SqlSig, Text2SqlModel
from evaluation_ex import calculate_ex
from evaluation_f1 import calculate_f1_score
from evaluation_utils import exec_sql
from gpt_request import generate_sql_file, post_process_response
from table_schema import generate_schema_prompt


with open("./openai_api_key.env", "r", encoding="utf-8") as f:
    OPENAI_API_KEY = f.read().strip()
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=512)
teacher_lm = dspy.OpenAI(model="gpt-4-0125-preview", api_key=OPENAI_API_KEY, model_type='chat')

def metric(  # pylint: disable=too-many-locals, too-many-branches
    example: Text2SqlSig,
    prediction: Text2SqlSig,
    trace=None,
) -> bool:
    """Metric for optimization"""

    ground_truth_res, gt_error = exec_sql(example.SQL, db_path, sql_dialect)
    assert not gt_error, f"Error in ground truth SQL: {gt_error}"
    predicted_res, _ = exec_sql(prediction.sqlite, db_path, sql_dialect)

    if trace is None:  # if we're doing evaluation or optimization, return evaluation_f1 score
        return calculate_f1_score(predicted_res, ground_truth_res)

    # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step,
    # return evaluation_ex
    return calculate_ex(predicted_res, ground_truth_res)

def compile_model(model_path: str,
                  train_set: list[dspy.Example],
                  validation_set: list[dspy.Example]) -> "Text2SqlModel":
    """Compile the module"""

    optimizer = MIPROv2(
        metric=metric,
        prompt_model=teacher_lm,
        task_model=lm
    )

    # kwargs = dict(num_threads=3, display_progress=True, display_table=0)

    compiled_module = optimizer.compile(
        student=Text2SqlModel().activate_assertions(),
        trainset=train_set,
        valset=validation_set,
        # max_bootstrapped_demos=3,
        # max_labeled_demos=5,
        # eval_kwargs=kwargs,
    )

    return compiled_module

def compile_and_save(model_path: str, train_set: list[dspy.Example],
                     validation_set: list[dspy.Example]):
    """train, compile and save. return the compiled model"""

    compiled_model = compile_model(model_path, train_set, validation_set)
    compiled_model.save(model_path)

def load_compiled_model(model_path: str) -> tuple["Text2SqlModel", bool]:
    """load compiled model"""
    compiled_model = Text2SqlModel().activate_assertions()

    is_compiled = False
    if os.path.exists(model_path):
        compiled_model.load(model_path)
        is_compiled = True

    # DO NOT ACTIVATE ASSERTIONS AFTER LOADING A COMPILED MODEL
    return compiled_model, is_compiled

# initialize the parameters
exec_result = []
db_path = './formula_1_data/formula_1/formula_1.sqlite'

sql_dialect = 'SQLite'
diff_json_path = "./formula_1_data/dev.json"

train_json_path = "./formula_1_data/train.json"
validation_json_path = "./formula_1_data/val.json"


def run_gpt():
    # delete error log file if it exists
    try:
        os.remove("error_log.txt")
    except Exception:
        pass

    # read the db schema, training and validation data
    db_schema = generate_schema_prompt(sql_dialect, db_path)

    # define the prompt BEFORE using the signature
    Text2SqlSig.__doc__ = (
        f"###DB SCHEMA:###\n"
        f"{db_schema}\n\n"
        f"###SYSTEM INSTRUCTIONS:###\n"
        "You are an expert sqlite developer. "
        "Using valid SQLite and understanding the evidence, answer the following question for the db schema provided above."
        f"\n\n"
    )

    SqlErrorCorrectionSig.__doc__ = (
        f"###DB SCHEMA:###\n"
        f"{db_schema}\n\n"
        f"###SYSTEM INSTRUCTIONS:###\n"
        "You are an expert sqlite developer well versed in debugging sqlite queries.\n"
        "You have been asked to fix the errors in a draft sqlite query based on the provided database schema and the errorinfo returned from the draft sqlite execution.\n"
        "Use the relevant db schema to find the correct table and join sequence to fix 'column does not exist' errors.\n"
        "IMPORTANT! Use table aliases for ALL tables in draft sql query to fix 'ambiguous column' errors. For e.g. Instead of 'SELECT colname FROM table_name', use 'SELECT alias.colname FROM table_name alias'."
        # f" \n"
        # "Notes: Always use table aliases to avoid ambiguity in column names. Review schema to ensure all column names are valid."
        f"\n\n"
    )

    # read the train data from the file
    with open(train_json_path, "r", encoding="UTF-8") as f:
        train_data = json.load(f)
    train_set = [dspy.Example(**d).with_inputs("question", "evidence") for d in train_data]

    # read the validation data from the file
    with open(validation_json_path, "r", encoding="UTF-8") as f:
        val_data = json.load(f)
    val_set = [dspy.Example(**d).with_inputs("question", "evidence") for d in val_data]

    model_path = "./dspy_models/MIPROv2.json"

    # compile and save
    compile_and_save(model_path, train_set, val_set)


    # delete error log file if it exists
    try:
        os.remove("error_log.txt")
    except Exception:
        pass

    # load the compiled model
    compiled_model, _ = load_compiled_model(model_path)

    # read the test data from the file
    with open(diff_json_path, "r", encoding="UTF-8") as f:
        test_data = json.load(f)
    test_examples = [dspy.Example(**d).with_inputs("question", "evidence") for d in test_data]

    # run the model on the test data
    responses = []
    for i, example in enumerate(test_examples):
        print(f"Processing example {i+1} of {len(test_examples)}", end='\r')
        sql = compiled_model(example.question, example.evidence).sqlite
        response = post_process_response(sql, db_path)
        responses.append((response, i))
    print("Done!")

    output_folder = "./exp_result/formula_1_dspy_miprov2/"
    engine='gpt-3.5-turbo'
    mode='mini_dev'

    output_path = (
        output_folder
        + "predict_"
        + mode
        + "_"
        + engine
        + "_cot"
        + "_"
        + sql_dialect
        + ".json"
    )
    generate_sql_file(sql_lst=responses, output_path=output_path)

    use_knowledge='True'
    chain_of_thought='True'
    print(
        "successfully collect results from {} for {} evaluation; SQL dialect {} Use knowledge: {}; Use COT: {}".format(
            engine,
            mode,
            sql_dialect,
            use_knowledge,
            chain_of_thought,
        )
    )


if __name__ == "__main__":
    run_gpt()
