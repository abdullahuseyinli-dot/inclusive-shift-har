from __future__ import annotations

from inclusive_shift_har.models.legacy_specs import (
    bilstm_parameter_count,
    cnn1d_parameter_count,
    corrected_legacy_architectures,
)


def test_corrected_legacy_specs_reconstruct_exact_primary_six_graphs() -> None:
    specs = {spec.architecture_id: spec for spec in corrected_legacy_architectures()}

    assert specs["legacy_bilstm_h192"].parameter_count() == 1_198_086
    assert specs["legacy_bilstm_h192_temporal"].parameter_count() == 1_201_158
    assert specs["legacy_bilstm_h256_temporal"].parameter_count() == 2_125_830
    assert specs["legacy_cnn1d_h128"].parameter_count() == 568_198
    assert specs["legacy_joint_bilstm256_cnn128"].parameter_count() == 2_694_028
    joint = specs["legacy_joint_bilstm256_cnn128"].to_dict()
    assert joint["training_semantics"] == "joint_end_to_end_single_objective"
    assert joint["aggregation"] == "arithmetic_mean_of_branch_logits"


def test_parameter_count_helpers_reject_invalid_dimensions() -> None:
    try:
        bilstm_parameter_count(input_size=0, hidden=192, layers=2, classes=6, temporal_head=False)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid BiLSTM dimensions were accepted")

    try:
        cnn1d_parameter_count(input_size=6, hidden=0, classes=6)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid CNN dimensions were accepted")
