import os
import re
import json

import dspy

from evaluation_utils import exec_sql
from gpt_request import generate_sql_file, post_process_response
from table_schema import generate_schema_prompt

class QueryPlannerSignature(dspy.Signature):
    """DYNAMICALLY SET"""

    question = dspy.InputField()
    evidence = dspy.InputField(desc="Use this external knowledge to guide the sql generation plan.")
    expertguidance = dspy.OutputField()

class QueryBreakdownSignature(dspy.Signature):
    """If the original sentence contains more than 3 key pieces of information, generate 1-3 simpler sentences containing only the key information."""

    originalsentence = dspy.InputField()
    sentencebreakdown = dspy.OutputField(desc="sentence1, sentence2, ...")

class Text2SqlSig(dspy.Signature):
    """DYNAMICALLY SET"""

    question = dspy.InputField()
    evidence = dspy.InputField(desc="Use this external knowledge to guide the sql generation.")
    # expertguidance = dspy.InputField()
    # keyquestions = dspy.InputField()
    sqlite = dspy.OutputField(desc="Formatted as: ```sql\n<SQL>\n``` where <SQL> is the generated SQL query.")

    @staticmethod
    def extract_sql(s: str) -> str:
        """Extracts the sql"""
        return (
            s
            .replace("```", "")
            .replace("Correctedsqlite:", "").replace("correctedsqlite:", "")
            .replace("Sqlite:", "").replace("sqlite:", "")
            .replace("sql", "")
        )

    @staticmethod
    def get_last_sql_statement(sql_text: str) -> str:
        """Extracts the last sql statement from a block of sql text with potentially multiple sql statements"""
        # Split the input text by ';' to get all statements
        statements = sql_text.split(';')

        # Filter out any empty statements and strip leading/trailing whitespace
        statements = [stmt.strip() for stmt in statements if stmt.strip()]

        # Return the last statement if there are any statements, else return None
        assert(statements)
        return statements[-1]

class SqlErrorCorrectionSig(dspy.Signature):
    """DYNAMICALLY SET"""

    question = dspy.InputField()
    evidence = dspy.InputField(desc="Use this external knowledge to guide the sql generation.")
    # expertguidance = dspy.InputField()
    # keyquestions = dspy.InputField()
    draftsqlite = dspy.InputField()
    errorinfo = dspy.InputField()
    relevant_schema = dspy.InputField(desc="Relevant portion of the database schema to help correct the SQL query.")
    correctedsqlite = dspy.OutputField(desc="Formatted as: ```sql\n<SQL>\n``` where <SQL> is a generated SQL query.")

    @staticmethod
    def get_table_schema_with_column(schema, column_name):
        # Split the schema into individual CREATE TABLE statements
        table_schemas = schema.split('CREATE TABLE ')
        # remove empty string from the list
        table_schemas = [schema for schema in table_schemas if schema]

        # Initialize a list to hold table schemas that contain the specified column
        table_schemas_with_column = ""

        # Check each table snippet for the specified column
        for schema in table_schemas:
            # search for column_name in the schema snippet. it should be in the format 'column_name '
            if f'{column_name} ' in schema:
                table_schemas_with_column += f'CREATE TABLE {schema}'

        return table_schemas_with_column

    @staticmethod
    def get_schema_of_FROM_table(sql_query, schema):
        # Define a regular expression pattern to match the 'FROM' clause
        from_pattern = re.compile(r'\bFROM\s+([^\s,]+)', re.IGNORECASE)

        # Search for the pattern in the query
        match = from_pattern.search(sql_query)

        # If a match is found, return the table name (ignoring any alias)
        if match:
            table_name = match.group(1)
            table_name = table_name.split()[0]  # Return the table name without alias

            # Split the schema into individual CREATE TABLE statements
            table_schemas = schema.split('CREATE TABLE ')
            # remove empty string from the list
            table_schemas = [schema for schema in table_schemas if schema]
            # append 'CREATE TABLE ' to each schema
            table_schemas = [f'CREATE TABLE {schema}' for schema in table_schemas]

            for schema in table_schemas:
                if f'CREATE TABLE {table_name}' in schema:
                    return schema

        return None


with open("./openai_api_key.env", "r", encoding="utf-8") as f:
    OPENAI_API_KEY = f.read().strip()
lm = dspy.OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=512)
# lm = DSPyUtils.get_lm("mistral/mistral.mixtral-8x7b-instruct-v0:1", max_tokens=512)
# lm = DSPyUtils.get_lm("meta/meta.llama3-70b-instruct-v1:0", max_tokens=512)

class Text2SqlModel(dspy.Module):
    def __init__(self):
        super().__init__()  # Call the __init__ method of the base class

        # self.queryplanner_func = dspy.ChainOfThought(QueryPlannerSignature)
        # self.querybreakdown_func = dspy.ChainOfThought(QueryBreakdownSignature)

        self.text2sql_func = dspy.ChainOfThought(Text2SqlSig)

        # self.text2sql_func = dspy.ChainOfThought(Text2SqlSig)
        self.sqlerrorcorrection_func = dspy.ChainOfThought(SqlErrorCorrectionSig)

    def forward(self, question: str, evidence: str) -> str:
        assert(question)
        if not evidence:
            evidence = "na"

        # with dspy.context(lm=lm):
        #     expertguidance = self.queryplanner_func(
        #         question=question,
        #         evidence=evidence
        #     ).expertguidance

        # with dspy.context(lm=lm):
        #     sentencebreakdown = self.querybreakdown_func(
        #         originalsentence=question,
        #     ).sentencebreakdown

        with dspy.context(lm=lm):
            prediction = self.text2sql_func(
                question=question,
                evidence=evidence,
                # expertguidance=expertguidance,
                # keyquestions=sentencebreakdown,
            )

        # lm.inspect_history(n=1)

        prediction.sqlite = Text2SqlSig.get_last_sql_statement(
            Text2SqlSig.extract_sql(prediction.sqlite).strip()
            )

        _, pred_error = exec_sql(prediction.sqlite, db_path, sql_dialect)

        if pred_error:
            # extract column name from the error message
            # For example, if the error message is "Error: no such column: races.laps"
            # then the column name is "laps"
            # For example, if the error message is "Error: no such column: driverId"
            # then the column name is "driverId"
            relevant_schema = "na"
            if 'no such column' in pred_error:
                match = re.search(r'(\.|\: )([^.]+)$', pred_error)
                column_name = match.group(2)
                table_schemas_with_column = SqlErrorCorrectionSig.get_table_schema_with_column(db_schema, column_name)
                from_table_schema = SqlErrorCorrectionSig.get_schema_of_FROM_table(prediction.sqlite, db_schema)
                if from_table_schema and table_schemas_with_column:
                    # build relevant schema by combining the FROM table schema and the schema with the column
                    relevant_schema = from_table_schema + table_schemas_with_column
                else:
                    relevant_schema = f"The column '{column_name}' does not exist in any table in the db schema"

            # dspy.Suggest(result=False, msg=pred_error)
            with dspy.context(lm=lm):
                prediction = self.sqlerrorcorrection_func(
                    question=question,
                    evidence=evidence,
                    # expertguidance = expertguidance,
                    # keyquestions=sentencebreakdown,
                    draftsqlite=prediction.sqlite,
                    errorinfo=pred_error,
                    relevant_schema=relevant_schema,
                )

            prediction.sqlite = Text2SqlSig.get_last_sql_statement(
                Text2SqlSig.extract_sql(prediction.correctedsqlite).strip()
                )

            _, pred_error = exec_sql(prediction.sqlite, db_path, sql_dialect)

            if pred_error:
                # create or append pred_error to an error log file
                with open("error_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"Que  :\n{question}\n")
                    f.write(f"Pred :\n{prediction.sqlite}\n")
                    f.write(f"Error: {pred_error}\n\n")

        return prediction

sql_dialect='SQLite'
db_path = "./formula_1_data/formula_1/formula_1.sqlite"
db_schema = generate_schema_prompt(sql_dialect, db_path)

# read the db schema from the schema.yaml file in the './formula_1_data/' folder
with open("./formula_1_data/schema.yaml", "r", encoding="UTF-8") as f:
    yaml_db_schema = f.read()

def run_gpt():
    # delete error log file if it exists
    try:
        os.remove("error_log.txt")
    except Exception:
        pass

    # define the prompt BEFORE using the signature
    # QueryPlannerSignature.__doc__ = (
    #     f"###DB SCHEMA:###\n"
    #     f"{db_schema}\n\n"
    #     f"###SYSTEM INSTRUCTIONS:###\n"
    #     "You are an expert sql developer. "
    #     "Your task is to write easy to read, step-by-step instructions for a junior developer to help her in writing the SQL query.\n"
    #     "The instructions should identify the relevant tables and columns, and the join conditions using the question, evidence, and db schema above."
    #     f"\n\n"
    # )

    # define the prompt BEFORE using the signature
    Text2SqlSig.__doc__ = (
        f"###DB SCHEMA:###\n"
        f"{yaml_db_schema}\n\n"
        f"###SYSTEM INSTRUCTIONS:###\n"
        "You are an expert sqlite developer. "
        "Using valid SQLite and understanding the evidence, answer the question using the db schema provided above."
        # f" \n"
        # "Notes: Always use table aliases to avoid ambiguity in column names. Review schema to ensure all column names are valid."
        f"\n\n"
    )

    SqlErrorCorrectionSig.__doc__ = (
        f"###DB SCHEMA:###\n"
        f"{yaml_db_schema}\n\n"
        f"###SYSTEM INSTRUCTIONS:###\n"
        "You are an expert sqlite developer well versed in debugging sqlite queries.\n"
        "You have been asked to fix the errors in a draft sqlite query based on the provided database schema and the errorinfo returned from the draft sqlite execution.\n"
        "Use the relevant db schema to find the correct table and join sequence to fix 'column does not exist' errors.\n"
        "IMPORTANT! Use table aliases for ALL tables in draft sql query to fix 'ambiguous column' errors. For e.g. Instead of 'SELECT colname FROM table_name', use 'SELECT alias.colname FROM table_name alias'."
        # f" \n"
        # "Notes: Always use table aliases to avoid ambiguity in column names. Review schema to ensure all column names are valid."
        f"\n\n"
    )

    model = Text2SqlModel().activate_assertions()

    eval_path = "./formula_1_data/dev.json"
    # read the test data from the file
    with open(eval_path, "r", encoding="UTF-8") as f:
        test_data = json.load(f)
    test_examples = [dspy.Example(**d).with_inputs("question", "evidence") for d in test_data]

    responses = []
    for i, example in enumerate(test_examples):
        print(f"Processing example {i+1} of {len(test_examples)}", end='\r')
        sql = model(example.question, example.evidence).sqlite
        response = post_process_response(sql, db_path)
        responses.append((response, i))
    print("\nDone!")

    output_folder = "./exp_result/formula_1_dspy_zeroshot/"
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