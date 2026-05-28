from neuroloc.simulations.memory.local_100k_unknown_structure_density_probe import (
    UnknownStructureCompressedCell,
    build_random_twin,
    build_summary,
    build_unknown_facts,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_unknown_structure_fact_pack_is_deterministic_and_unique() -> None:
    first, first_corpus, first_manifest = build_unknown_facts(631, 128)
    second, second_corpus, second_manifest = build_unknown_facts(631, 128)
    third, _, _ = build_unknown_facts(632, 128)
    assert first == second
    assert first_corpus == second_corpus
    assert first_manifest == second_manifest
    assert first != third
    assert len({row["key"] for row in first}) == len(first)
    assert len({row["value"] for row in first}) == len(first)


def test_unknown_structure_cell_reads_real_corpus_but_not_random_twin() -> None:
    facts, corpus, manifest = build_unknown_facts(631, 128)
    random_twin = build_random_twin(631, facts)
    cell = UnknownStructureCompressedCell(corpus, manifest)
    real_reads = [cell.read(int(row["key"])) for row in facts]
    twin_reads = [cell.read(int(row["key"])) for row in random_twin]
    real_success = sum(row["exact_success"] for row in score_reads(facts, real_reads)) / len(facts)
    twin_success = sum(row["exact_success"] for row in score_reads(random_twin, twin_reads)) / len(random_twin)
    assert real_success == 1.0
    assert twin_success == 0.0


def test_unknown_structure_summary_is_boundary_not_breakthrough() -> None:
    summary = build_summary("smoke", seed=631)
    assert summary["local_100k_unknown_structure_density_probe_engineering_pass"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_corpus_probe_pass"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_exact_retrieval_success"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_random_label_twin_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_strict_density"] < summary["local_100k_unknown_structure_density_probe_target_density"]
    assert summary["local_100k_unknown_structure_density_probe_strict_600x_pass"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_standard_codec_dependency"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_sequence_offset_key_target"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_associative_random_key_target"] == 0.0


def test_unknown_structure_controls_collapse() -> None:
    summary = build_summary("smoke", seed=631)
    assert summary["local_100k_unknown_structure_density_probe_controls_collapse"] == 1.0
    assert summary["local_100k_unknown_structure_density_probe_no_memory_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_read_disabled_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_decoder_disabled_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_shuffled_key_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_shuffled_value_success"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_shuffled_provenance_success"] == 0.0


def test_unknown_structure_accounting_charges_decoder_and_manifest() -> None:
    summary = build_summary("smoke", seed=631)
    assert summary["local_100k_unknown_structure_density_probe_decoder_bits"] >= 65536.0
    assert summary["local_100k_unknown_structure_density_probe_manifest_bits"] > 0.0
    assert summary["local_100k_unknown_structure_density_probe_committed_state_bits"] > summary["local_100k_unknown_structure_density_probe_compressed_bytes"] * 8.0
    assert summary["local_100k_unknown_structure_density_probe_formula_or_schema_labels_present"] == 0.0
    assert summary["local_100k_unknown_structure_density_probe_seed_oracle_authorized"] == 0.0


def test_unknown_structure_registry_entry() -> None:
    assert "local_100k_unknown_structure_density_probe" in SIMULATION_SPECS
    assert "local_100k_unknown_structure_density_probe" in SUITES["compression_mirror"]
    assert "local_100k_unknown_structure_density_probe" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_unknown_structure_density_probe"]
    assert spec.metrics_filename == "local_100k_unknown_structure_density_probe_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_unknown_structure_density_probe_corpus_probe_pass"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_unknown_structure_density_probe_strict_breakthrough_authorized"] == 0.0
