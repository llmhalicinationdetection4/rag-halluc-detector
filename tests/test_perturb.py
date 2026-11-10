from halludetect.inject.perturb import (
    perturb_contradiction,
    perturb_date_swap,
    perturb_negation,
    perturb_number_swap,
)


def test_perturb_numbers():
    text = "There are 12 steps in the pipeline."
    perturbed = perturb_number_swap(text)
    assert perturbed != text
    assert "12" not in perturbed


def test_perturb_dates():
    text = "The foundation was built in 1985."
    perturbed = perturb_date_swap(text)
    assert perturbed != text
    assert "1985" not in perturbed


def test_perturb_negation():
    text = "The system is reliable."
    perturbed = perturb_negation(text)
    assert "is not" in perturbed or "false" in perturbed


def test_perturb_contradiction():
    text = "This was the largest ocean on Earth."
    perturbed = perturb_contradiction(text)
    assert "smallest" in perturbed