import pytest

from doc_ufcn import model


@pytest.mark.parametrize(
    ("no_of_classes", "expected_no_of_params"),
    [
        # 1 class
        (1, 4092897),
        # 2 classes
        (2, 4093474),
        # Wrong number of classes
        (2.5, None),
        # Wrong number of classes
        (-1, None),
        # 5 classes
        (5, 4095205),
    ],
)
def test_DocUFCNModel(no_of_classes, expected_no_of_params):
    """
    Test of the DocUFCNModel init function.
    Check that the model is correct.
    """
    if isinstance(no_of_classes, float):
        with pytest.raises(
            ValueError, match="out_channels must be divisible by groups"
        ):
            net = model.DocUFCNModel(no_of_classes)
    elif no_of_classes < 0:
        with pytest.raises(RuntimeError):
            net = model.DocUFCNModel(no_of_classes)
    else:
        net = model.DocUFCNModel(no_of_classes)
        total_params = sum(parameter.numel() for parameter in net.parameters())
        assert total_params == expected_no_of_params
