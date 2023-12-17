import inspect
import re

from torch import optim


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        s = s.split(",")
        method = s[0]
        optim_params = {}
        for x in s[1:]:
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = optim.Adam
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception("Unknown optimization method: '%s'" % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception("Unexpected parameters: expected '%s', got '%s'" % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params
