from .detect import (
    find_and_add_test_splits as find_and_add_test_splits,
    find_and_add_train_splits as find_and_add_train_splits,
    find_processed_dataset as find_processed_dataset,
)
from .loading import load_json as load_json
from .splitting import (
    BIDSsplit_40_10_50,
    MCSAsplit_40_10_50,
    PatientIDsplit_40_10_50,
    dynamic_grouped_split as dynamic_grouped_split,
    dynamic_split as dynamic_split,
    split as split,
    split_40_10_50,
)

all_split_fn = [
    split_40_10_50.__name__,
    BIDSsplit_40_10_50.__name__,
    PatientIDsplit_40_10_50.__name__,
    MCSAsplit_40_10_50.__name__,
]
