"""
Feb 16 2019
Model selection and hyper-parameter search
"""
import copy
import itertools
import os
from typing import Dict, List, Union, Iterable
import model_builder


def gen_hparam_set(
    src_dict: Dict[str, Union[List[object], object]]
) -> List[Dict[str, object]]:
    """
    Generate a collection of hyperparameters for hyparam searching.
    NOTE: in this version, a parameter configuration object is a 
        dictionary with string as keys. When a parameter config
        is fed into a model training session, we use 'globals().update(param_config)'
        to read the config.
    Args:
        src_dict:
            A dictionary with string-keys exactly the same as the sample
            input config below.
            NOTE: for parameters that one wish to search over certain set
                of potential choices, put a LIST of feasible values at the 
                corresponding value in src_dict.
            Example:
                to search over learning_rate parameter,
                set src_dict["learning_rate"] = [0.1, 0.03, 0.01] etc.
    Returns:
        A list (iterable) with all combination of candidates in
            flexiable (to be searched) parameters.
    """
    gen = list()
    detected_list_keys = list()
    detected_list_vals = list()

    for k, v in src_dict.items():
        if isinstance(v, list):
            detected_list_keys.append(k)
            detected_list_vals.append(v)

    cartesian_prod = list(itertools.product(*detected_list_vals))

    for coor in cartesian_prod:
        new_para = copy.deepcopy(src_dict)
        hparam_str = "-".join(
            f"{k}={v}" for k, v in zip(detected_list_keys, coor))
        for i, key in enumerate(detected_list_keys):
            new_para[key] = coor[i]
        gen.append(new_para)

    print(f"Total number of parameter sets generated: {len(gen)}")
    return gen


def grid_search(
    param_set: Iterable[dict],
    X_train, X_test,
    y_train, y_test
)-> List[dict]:
    candidates = list()
    total = len(param_set)
    for i, param in enumerate(param_set):
        c = model_builder.Classifier(param, verbose=False)

        c.fit(X_train, y_train)
        auc = c.evaluate_auc(X_test, y_test)
        accuracy = c.model.evaluate(X_test, y_test, verbose=0)[1]
        record = {
            "param": param,
            "test_auc": auc,
            "test_acc": accuracy
        }
        candidates.append(record)
        best_auc = max(x["test_auc"] for x in candidates)
        best_acc = max(x["test_acc"] for x in candidates)
        print(f"HP Grid Searching: [{i+1}/{total}].")
        print(f"[Test Set]Best AUC: {best_auc}, best ACC: {best_acc}")
    candidates.sort(key=lambda x: x["test_auc"])
    return candidates
