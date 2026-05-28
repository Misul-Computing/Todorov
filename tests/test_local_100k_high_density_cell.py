from neuroloc.simulations.memory.local_100k_high_density_cell import (
    HighDensityNeuronCell,
    build_summary,
    capacities,
    generate_facts,
    key_to_index,
    score_reads,
    split_fact_sets,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_high_density_fact_generation_is_deterministic() -> None:
    assert generate_facts(11, 16) == generate_facts(11, 16)
    assert generate_facts(11, 16) != generate_facts(12, 16)


def test_high_density_train_test_keys_do_not_overlap() -> None:
    train, test = split_fact_sets(17, 64, 256)
    train_keys = {row["key_index"] for row in train}
    test_keys = {row["key_index"] for row in test}
    assert not train_keys.intersection(test_keys)


def test_high_density_labels_are_not_recency_or_template_solved() -> None:
    facts = generate_facts(19, 128)
    recency = [{"value": facts[-1]["value"], "provenance": facts[-1]["provenance"], "hit": 1} for _ in facts]
    shuffled = [{"value": row["value"], "provenance": row["provenance"], "hit": 1} for row in facts[-1:] + facts[:-1]]
    assert sum(row["exact_success"] for row in score_reads(facts, recency)) <= 1
    assert sum(row["exact_success"] for row in score_reads(facts, shuffled)) == 0


def test_high_density_cell_exact_retrieval_and_controls() -> None:
    caps = capacities()
    facts = generate_facts(23, 256)
    cell = HighDensityNeuronCell(caps)
    cell.write(facts)
    reads = [cell.read(tuple(row["key"])) for row in facts]
    assert all(row["exact_success"] == 1.0 for row in score_reads(facts, reads))
    disabled = HighDensityNeuronCell(caps)
    disabled.write(facts, disabled=True)
    disabled_reads = [disabled.read(tuple(row["key"])) for row in facts]
    assert all(row["exact_success"] == 0.0 for row in score_reads(facts, disabled_reads))
    assert all(key_to_index(tuple(row["key"]), caps) == row["key_index"] for row in facts)


def test_high_density_summary_reports_partial_not_strict_breakthrough() -> None:
    summary = build_summary("smoke", seed=313)
    assert summary["local_100k_high_density_cell_engineering_pass"] == 1.0
    assert summary["local_100k_high_density_cell_exact_retrieval_success"] >= 0.95
    assert summary["local_100k_high_density_cell_params_only_multiplier"] >= 600.0
    assert summary["local_100k_high_density_cell_params_only_600x_pass"] == 1.0
    assert summary["local_100k_high_density_cell_strict_600x_pass"] == 0.0
    assert summary["local_100k_high_density_cell_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_high_density_cell_claim_downgraded_to_params_only"] == 1.0
    assert summary["local_100k_high_density_cell_controls_collapse"] == 1.0
    assert summary["local_100k_high_density_cell_recency_only_success"] <= 0.01


def test_high_density_registry_entry() -> None:
    assert "local_100k_high_density_cell" in SIMULATION_SPECS
    assert "local_100k_high_density_cell" in SUITES["compression_mirror"]
    assert "local_100k_high_density_cell" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_high_density_cell"]
    assert spec.metrics_filename == "local_100k_high_density_cell_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_high_density_cell_params_only_600x_pass"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_high_density_cell_strict_breakthrough_authorized"] == 0.0
