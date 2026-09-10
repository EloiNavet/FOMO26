import os
from dotenv import load_dotenv


def var_is_set(var):
    return var in os.environ.keys()


def get_environment_variable(var):
    load_dotenv()

    if not var_is_set(var):
        raise ValueError(f"Missing required environment variable {var}.")

    path = os.environ[var]
    os.makedirs(path, exist_ok=True)
    return path


def get_data_path():
    return get_environment_variable("ASPARAGUS_DATA")


def get_raw_labels_path():
    return get_environment_variable("ASPARAGUS_RAW_LABELS")


def get_source_path():
    return get_environment_variable("ASPARAGUS_SOURCE")
