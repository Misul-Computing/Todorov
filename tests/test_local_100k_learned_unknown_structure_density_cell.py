from neuroloc.simulations.memory.local_100k_learned_unknown_structure_density_cell import (
    LearnedUnknownStructureDensityCell,
    build_facts,
    build_random_twin,
    build_summary,
    decode_tokens,
    encode_chunk,
    learn_dictionary,
    score_reads,
)
from neuroloc.simulations.suite_registry import SIMULATION_SPECS, SUITES


def test_learned_unknown_structure_fact_pack_is_deterministic_and_source_heldout() -> None:
    first_train, first_test, first_corpus, first_manifest = build_facts(733, 128, 64)
    second_train, second_test, second_corpus, second_manifest = build_facts(733, 128, 64)
    third_train, third_test, _, _ = build_facts(734, 128, 64)
    assert first_train == second_train
    assert first_test == second_test
    assert first_corpus == second_corpus
    assert first_manifest == second_manifest
    assert (first_train, first_test) != (third_train, third_test)
    assert {row["source"] for row in first_train}.isdisjoint({row["source"] for row in first_test})
    assert not {tuple(row["key"]) for row in first_train}.intersection({tuple(row["key"]) for row in first_test})


def test_learned_unknown_structure_keys_are_not_offsets() -> None:
    _, facts, _, _ = build_facts(733, 128, 64)
    offsets = {int(row["offset"]) for row in facts}
    key_words = {word for row in facts for word in tuple(row["key"])}
    assert not offsets.intersection(key_words)
    assert all(len(tuple(row["key"])) == 4 for row in facts)
    assert len({tuple(row["key"]) for row in facts}) == len(facts)


def test_learned_dictionary_roundtrips_heldout_chunks_exactly() -> None:
    train, facts, _, _ = build_facts(733, 128, 64)
    dictionary = learn_dictionary(train, 128)
    for fact in facts[:32]:
        chunk = bytes.fromhex(str(fact["value"]))
        tokens = encode_chunk(chunk, dictionary)
        assert decode_tokens(tokens, dictionary) == chunk


def test_learned_unknown_structure_cell_reads_real_facts_but_not_random_twin() -> None:
    train, facts, _, manifest = build_facts(733, 128, 64)
    random_twin = build_random_twin(733, facts)
    cell = LearnedUnknownStructureDensityCell(train, facts, manifest)
    real_reads = [cell.read(tuple(row["key"])) for row in facts]
    twin_reads = [cell.read(tuple(row["key"])) for row in random_twin]
    real_success = sum(row["exact_success"] for row in score_reads(facts, real_reads)) / len(facts)
    twin_success = sum(row["exact_success"] for row in score_reads(random_twin, twin_reads)) / len(random_twin)
    assert real_success == 1.0
    assert twin_success == 0.0


def test_learned_unknown_structure_random_twin_can_be_stored_by_same_table_path() -> None:
    train, facts, _, manifest = build_facts(733, 128, 64)
    random_twin = build_random_twin(733, facts)
    cell = LearnedUnknownStructureDensityCell(train, random_twin, manifest)
    reads = [cell.read(tuple(row["key"])) for row in random_twin]
    success = sum(row["exact_success"] for row in score_reads(random_twin, reads)) / len(random_twin)
    assert success == 1.0


def test_learned_unknown_structure_summary_is_hard_defeat_not_breakthrough() -> None:
    summary = build_summary("smoke", seed=733)
    assert summary["local_100k_learned_unknown_structure_density_cell_engineering_pass"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_hard_defeat"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_heldout_exact_retrieval_success"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_random_label_twin_success"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_random_label_twin_storage_success"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_random_label_cross_label_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_random_label_control_collapse"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_learned_cell_pass"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_strict_600x_pass"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_strict_breakthrough_authorized"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_general_unknown_structure_breakthrough_authorized"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_full_nm_authorized"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_no_per_fact_committed_rows"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_beats_charged_codec_baseline"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_beats_all_reported_baselines"] == 0.0


def test_learned_unknown_structure_controls_collapse() -> None:
    summary = build_summary("smoke", seed=733)
    assert summary["local_100k_learned_unknown_structure_density_cell_controls_collapse"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_no_memory_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_write_disabled_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_read_disabled_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_decoder_disabled_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_dictionary_disabled_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_residual_disabled_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_shuffled_key_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_shuffled_value_success"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_shuffled_provenance_success"] == 0.0


def test_learned_unknown_structure_accounting_charges_side_information() -> None:
    summary = build_summary("smoke", seed=733)
    assert summary["local_100k_learned_unknown_structure_density_cell_parameter_count"] < 100000.0
    assert summary["local_100k_learned_unknown_structure_density_cell_dictionary_bits"] > 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_residual_payload_bits"] > 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_associative_assignment_bits"] > 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_training_supervision_bits"] > 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_query_key_bits"] > 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_committed_state_bits"] > summary["local_100k_learned_unknown_structure_density_cell_useful_retrievable_bits"] * 0.25
    assert summary["local_100k_learned_unknown_structure_density_cell_formula_or_schema_labels_present"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_seed_oracle_authorized"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_sequence_offset_key_target"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_associative_random_key_target"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_learned_path_used"] == 0.0
    assert summary["local_100k_learned_unknown_structure_density_cell_learned_dictionary_used"] == 1.0
    assert summary["local_100k_learned_unknown_structure_density_cell_residual_table_path_used"] == 1.0


def test_learned_unknown_structure_registry_entry() -> None:
    assert "local_100k_learned_unknown_structure_density_cell" in SIMULATION_SPECS
    assert "local_100k_learned_unknown_structure_density_cell" in SUITES["compression_mirror"]
    assert "local_100k_learned_unknown_structure_density_cell" in SUITES["precompute"]
    spec = SIMULATION_SPECS["local_100k_learned_unknown_structure_density_cell"]
    assert spec.metrics_filename == "local_100k_learned_unknown_structure_density_cell_metrics.json"
    assert dict(spec.minimum_summary_values)["local_100k_learned_unknown_structure_density_cell_hard_defeat"] == 1.0
    assert dict(spec.maximum_summary_values)["local_100k_learned_unknown_structure_density_cell_strict_breakthrough_authorized"] == 0.0
