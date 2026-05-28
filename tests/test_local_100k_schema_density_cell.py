from neuroloc.simulations.memory.local_100k_schema_density_cell import (
    SchemaDensityNeuronCell,
    build_summary,
    capacities,
    generate_random_facts,
    generate_schema_facts,
    score_reads,
    split_schema_sets,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_schema_density_generation_is_deterministic() -> None:
    assert generate_schema_facts(31, 32) == generate_schema_facts(31, 32)
    assert generate_schema_facts(31, 32) != generate_schema_facts(32, 32)


def test_schema_density_train_test_keys_do_not_overlap() -> None:
    train, test = split_schema_sets(37, 24, 512)
    train_keys = {row["key_index"] for row in train}
    test_keys = {row["key_index"] for row in test}
    assert not train_keys.intersection(test_keys)


def test_schema_density_cell_generalizes_without_per_fact_rows() -> None:
    caps = capacities()
    train, test = split_schema_sets(41, 24, 512)
    cell = SchemaDensityNeuronCell(caps)
    cell.fit(train)
    reads = [cell.read(tuple(row["key"])) for row in test]
    assert all(row["exact_success"] == 1.0 for row in score_reads(test, reads))
    assert not hasattr(cell, "values")
    assert not hasattr(cell, "provenance")
    assert not hasattr(cell, "commit")


def test_schema_density_controls_collapse() -> None:
    summary = build_summary("smoke", seed=419)
    assert summary["local_100k_schema_density_cell_exact_retrieval_success"] == 1.0
    assert summary["local_100k_schema_density_cell_controls_collapse"] == 1.0
    assert summary["local_100k_schema_density_cell_no_memory_success"] == 0.0
    assert summary["local_100k_schema_density_cell_schema_disabled_success"] == 0.0
    assert summary["local_100k_schema_density_cell_shuffled_schema_success"] == 0.0
    assert summary["local_100k_schema_density_cell_decoder_disabled_success"] == 0.0
    assert summary["local_100k_schema_density_cell_write_disabled_success"] == 0.0


def test_schema_density_random_entropy_control_does_not_compress() -> None:
    caps = capacities()
    train = generate_random_facts(47, 24, 0)
    test = generate_random_facts(47, 512, 24)
    cell = SchemaDensityNeuronCell(caps)
    cell.fit(train[: cell.width])
    reads = [cell.read(tuple(row["key"])) for row in test]
    assert sum(row["exact_success"] for row in score_reads(test, reads)) == 0
    summary = build_summary("smoke", seed=419)
    assert summary["local_100k_schema_density_cell_random_entropy_control_success"] == 0.0
    assert summary["local_100k_schema_density_cell_independent_random_600x_pass"] == 0.0


def test_schema_density_summary_rejects_structured_target_as_knowledge_compression() -> None:
    summary = build_summary("smoke", seed=419)
    assert summary["local_100k_schema_density_cell_engineering_pass"] == 1.0
    assert summary["local_100k_schema_density_cell_structured_strict_600x_pass"] == 1.0
    assert summary["local_100k_schema_density_cell_strict_density"] >= 1500.0
    assert summary["local_100k_schema_density_cell_strict_with_supervision_density"] >= 1500.0
    assert summary["local_100k_schema_density_cell_structured_strict_compression_authorized"] == 0.0
    assert summary["local_100k_schema_density_cell_general_independent_fact_breakthrough_authorized"] == 0.0
    assert summary["local_100k_schema_density_cell_claim_limited_to_structured_facts"] == 1.0
    assert summary["local_100k_schema_density_cell_target_rejected_by_user"] == 1.0
    assert summary["local_100k_schema_density_cell_target_valid_for_high_density_knowledge"] == 0.0


def test_schema_density_registry_entry() -> None:
    assert "local_100k_schema_density_cell" in SIMULATION_SPECS
    assert "local_100k_schema_density_cell" in SUITES["compression_mirror"]
    assert "local_100k_schema_density_cell" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_schema_density_cell"]
    assert spec.metrics_filename == "local_100k_schema_density_cell_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_schema_density_cell_structured_boundary_result"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_schema_density_cell_structured_strict_compression_authorized"] == 0.0
    assert dict(spec.maximum_summary_values)["local_100k_schema_density_cell_general_independent_fact_breakthrough_authorized"] == 0.0
