from contextlib import suppress
import json
import os
import re
import shutil
import dspy

from mixture_of_experts import MixtureOfExperts, MixtureOfExpertsProgram

from dspy_sig_and_model import SqlErrorCorrectionSig, Text2SqlSig, Text2SqlModel
from evaluation_ex import calculate_ex
from evaluation_f1 import calculate_f1_score
from evaluation_utils import exec_sql
from gpt_request import generate_sql_file, post_process_response
from table_schema import generate_schema_prompt


with open("./openai_api_key.env", "r", encoding="utf-8") as f:
    OPENAI_API_KEY = f.read().strip()
teacher_lm = dspy.OpenAI(model="gpt-4-0125-preview", api_key=OPENAI_API_KEY, max_tokens=512)

def metric(  # pylint: disable=too-many-locals, too-many-branches
    example: Text2SqlSig,
    prediction: Text2SqlSig,
    trace=None,
) -> bool:
    """Metric for optimization"""

    ground_truth_res, gt_error = exec_sql(example.SQL, db_path, sql_dialect)
    assert not gt_error, f"Error in ground truth SQL: {gt_error}"
    predicted_res, pred_error = exec_sql(prediction.sqlite, db_path, sql_dialect)

    if trace is None:  # if we're doing evaluation or optimization, return evaluation_f1 score
        return calculate_f1_score(predicted_res, ground_truth_res)

    # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step,
    # return evaluation_ex
    return calculate_ex(predicted_res, ground_truth_res)


def complexity_based_cluster_func(examples: list[dspy.Example], num_of_clusters) -> list[list[dspy.Example]]:
    """complexisty based cluster function."""
    assert num_of_clusters == 3, "This function is designed to cluster into 3 groups"

    challenging_sql_keywords = [
        "WITH", "IIF",
        "CASE WHEN",
        "BETWEEN", "HAVING", "UNION", "ALL", "EXCEPT", "PARTITION BY", "OVER",
        "ABS", "LENGTH", "STRFTIME", "JULIANDAY", "NOW", "CAST", "SUBSTR", "INSTR"
    ]
    moderate_complexity_sql_keywords = [
        "NOT", "IN", "EXISTS",
        "LEFT JOIN",
        "LIKE", "LIMIT", "GROUP BY"
    ]

    challenging_questions = []
    moderate_complexity_questions = []
    simple_questions = []
    for example in examples:
        # we are using regular expressions because we want to match whole words only
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', example.SQL) for keyword in challenging_sql_keywords):
            challenging_questions.append(example)
        elif any(re.search(r'\b' + re.escape(keyword) + r'\b', example.SQL) for keyword in moderate_complexity_sql_keywords):
            moderate_complexity_questions.append(example)
        else:
            simple_questions.append(example)

    return [simple_questions, moderate_complexity_questions, challenging_questions]

def compile_model(train_set: list[dspy.Example],
                  validation_set: list[dspy.Example]) -> "Text2SqlModel":
    """Compile the module"""

    compiled_module = MixtureOfExperts(
        metric_func=metric,
        # teacher_settings={"lm": teacher_lm},
        cluster_func=complexity_based_cluster_func,
        bootstrapfewshotwithrandomsearch_num_candidate_programs = 10,
        bootstrapfewshotwithrandomsearch_num_threads = 3,
    ).compile(
        Text2SqlModel().activate_assertions(),
        trainset=train_set,
        valset=validation_set,
        number_of_experts=3,
    )

    return compiled_module

def compile_and_save(model_path: str, train_set: list[dspy.Example],
                     validation_set: list[dspy.Example]):
    """train, compile and save. return the compiled model"""

    compiled_model = compile_model(train_set, validation_set)

    with suppress(FileNotFoundError):
        if os.path.isfile(model_path):
            os.remove(model_path)
        else:
            shutil.rmtree(model_path)

    compiled_model.save_folder(model_path)

def load_compiled_model(model_path: str) -> tuple["Text2SqlModel", bool]:
    """load compiled model"""

    compiled_model = Text2SqlModel().activate_assertions()

    is_compiled = False
    if os.path.exists(model_path):
        compiled_model = MixtureOfExpertsProgram()
        compiled_model.load_folder(model_path, Text2SqlModel, activate_assertions=True)

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

    model_path = "./dspy_models/mixture_of_experts"

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
        example_input = {key: value for key, value in example.items() if key in example._input_keys}
        sql = compiled_model(**example_input).sqlite
        response = post_process_response(sql, db_path)
        responses.append((response, i))
    print("Done!")

    output_folder = "./exp_result/formula_1_dspy_mixtureofexperts/"
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
