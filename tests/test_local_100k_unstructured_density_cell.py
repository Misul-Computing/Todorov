from neuroloc.simulations.memory.local_100k_unstructured_density_cell import (
    UnstructuredSketchCell,
    build_summary,
    capacities,
    generate_unstructured_facts,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_unstructured_density_generation_is_deterministic_without_formula_labels() -> None:
    first = generate_unstructured_facts(61, 64)
    second = generate_unstructured_facts(61, 64)
    third = generate_unstructured_facts(62, 64)
    assert first == second
    assert first != third
    ordered_values = [row["key_index"] % (2**16) for row in first]
    assert [row["value"] for row in first] != ordered_values


def test_unstructured_density_key_split_has_no_overlap() -> None:
    train = generate_unstructured_facts(67, 128, 0)
    test = generate_unstructured_facts(67, 512, 128)
    train_keys = {row["key_index"] for row in train}
    test_keys = {row["key_index"] for row in test}
    assert not train_keys.intersection(test_keys)


def test_unstructured_sketch_does_not_store_per_fact_rows() -> None:
    facts = generate_unstructured_facts(71, 256)
    cell = UnstructuredSketchCell(bin_count=4, payload_bits=32, checksum_bits=16)
    cell.write(facts, provenance_bits=16)
    reads = [cell.read(tuple(row["key"]), provenance_bits=16) for row in facts]
    assert sum(row["exact_success"] for row in score_reads(facts, reads)) < len(facts) * 0.25
    assert not hasattr(cell, "values")
    assert not hasattr(cell, "provenance")
    assert not hasattr(cell, "commit")


def test_unstructured_summary_rejects_600x_exact_storage() -> None:
    summary = build_summary("smoke", seed=521)
    assert summary["local_100k_unstructured_density_cell_engineering_pass"] == 1.0
    assert summary["local_100k_unstructured_density_cell_information_theoretic_600x_possible"] == 0.0
    assert summary["local_100k_unstructured_density_cell_strict_600x_pass"] == 0.0
    assert summary["local_100k_unstructured_density_cell_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_unstructured_density_cell_general_independent_fact_breakthrough_authorized"] == 0.0
    assert summary["local_100k_unstructured_density_cell_exact_gate_pass"] == 0.0
    assert summary["local_100k_unstructured_density_cell_entropy_budget_gap_bits"] > 0.0
    assert summary["local_100k_unstructured_density_cell_formula_or_schema_labels_present"] == 0.0
    assert summary["local_100k_unstructured_density_cell_seed_oracle_authorized"] == 0.0


def test_unstructured_controls_collapse() -> None:
    summary = build_summary("smoke", seed=521)
    assert summary["local_100k_unstructured_density_cell_controls_collapse"] == 1.0
    assert summary["local_100k_unstructured_density_cell_no_memory_success"] == 0.0
    assert summary["local_100k_unstructured_density_cell_write_disabled_success"] == 0.0
    assert summary["local_100k_unstructured_density_cell_read_disabled_success"] == 0.0
    assert summary["local_100k_unstructured_density_cell_decoder_disabled_success"] == 0.0


def test_unstructured_registry_entry() -> None:
    assert "local_100k_unstructured_density_cell" in SIMULATION_SPECS
    assert "local_100k_unstructured_density_cell" in SUITES["compression_mirror"]
    assert "local_100k_unstructured_density_cell" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_unstructured_density_cell"]
    assert spec.metrics_filename == "local_100k_unstructured_density_cell_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_unstructured_density_cell_useful_negative_result"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_unstructured_density_cell_strict_breakthrough_authorized"] == 0.0
