import numpy as np

from moogp.nuclear_data_experiment import (
    DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
    load_nuclear_dataset,
    make_block_specs,
    train_test_split_indices,
)


def test_output_index_parser_clamps_to_available_columns():
    dataset = load_nuclear_dataset()
    families = dataset["output_index"]

    assert [family.name for family in families] == [
        "dNch_deta",
        "dET_deta",
        "dN_dy_pion",
        "dN_dy_kaon",
        "dN_dy_proton",
        "mean_pT_pion",
        "mean_pT_kaon",
        "mean_pT_proton",
        "pT_fluct",
        "v22",
    ]
    assert families[-1].start == 90
    assert families[-1].end == 98


def test_block_specs_match_requested_ranges_and_q_fraction():
    specs = make_block_specs(
        98,
        blocks=DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
        q_fraction=0.25,
    )

    assert [(spec["start"], spec["end"], spec["q"]) for spec in specs] == [
        (0, 30, 8),
        (30, 54, 6),
        (54, 78, 6),
        (78, 98, 5),
    ]
    assert [spec["column_range"] for spec in specs] == ["1-30", "31-54", "55-78", "79-98"]
    assert [spec["q_rule"] for spec in specs] == [
        "ceil(0.25 * p_block)",
        "ceil(0.25 * p_block)",
        "ceil(0.25 * p_block)",
        "ceil(0.25 * p_block)",
    ]


def test_block_specs_support_fixed_q_override():
    specs = make_block_specs(
        98,
        blocks=DEFAULT_NUCLEAR_OUTPUT_BLOCKS,
        q_fraction=0.25,
        fixed_q=4,
    )

    assert [spec["q"] for spec in specs] == [4, 4, 4, 4]
    assert [spec["q_rule"] for spec in specs] == [
        "fixed_q=4",
        "fixed_q=4",
        "fixed_q=4",
        "fixed_q=4",
    ]


def test_train_test_split_is_reproducible_and_disjoint():
    train_idx, test_idx = train_test_split_indices(541, train_fraction=0.8, seed=42)

    assert train_idx.size == 432
    assert test_idx.size == 109
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert np.array_equal(np.sort(np.concatenate([train_idx, test_idx])), np.arange(541))
