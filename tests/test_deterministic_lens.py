"""The deterministic lens: positive cases, the false-positive guards that earn
its silence, determinism, and a run over the three real committed manuscripts.

The guards are the load-bearing half. A checker that flags "Received: 1 Mar
2024" is worse than no checker, so every guard has a test that asserts silence,
and the real-manuscript result is pinned as whatever it actually is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.agents.deterministic_lens import (
    CHECK_GENE_SYMBOL,
    CHECK_REFERENCE_INTEGRITY,
    CHECK_STAT_SANITY,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    check_gene_symbol_corruption,
    check_reference_integrity,
    check_stat_sanity,
    run_deterministic_lens,
)
from peerpanel.manuscripts.store import read_manuscript

MANUSCRIPTS_DIR = Path(__file__).resolve().parent.parent / "manuscripts"

EN_DASH = "\N{EN DASH}"
PLUS_MINUS = "\N{PLUS-MINUS SIGN}"
LESS_EQUAL = "\N{LESS-THAN OR EQUAL TO}"
GREATER_EQUAL = "\N{GREATER-THAN OR EQUAL TO}"

# The three committed manuscripts and what the lens actually finds in each. The
# zeros are the measured truth, not an aspiration: all three are peer-reviewed
# preprints, and the JATS extraction strips their reference lists, so the two
# reference checks that need a numbered list have nothing to count against.
# A new manuscript must re-pin this table (TestRealManuscripts guards that).
EXPECTED_REAL_FINDINGS: dict[str, int] = {
    "caprin-heterochromatin.txt": 0,
    "met17-auxotroph.txt": 0,
    "oleaginous-yeast-biopolymer.txt": 0,
}


def details(findings: list[object]) -> str:
    """All finding details joined — for readable substring assertions."""
    return " | ".join(getattr(f, "detail", "") for f in findings)


class TestGeneSymbolCorruption:
    def test_every_date_form_of_a_month_prefixed_symbol_is_flagged(self) -> None:
        text = (
            "Differential expression analysis\n"
            "The most strongly induced transcripts were 1-Mar, Mar-1 and 01-Mar in the\n"
            "knockout background, together with 2-Sep, Sep-2, March-1 and 1-Dec.\n"
        )
        findings = check_gene_symbol_corruption(text)
        flagged = details(list(findings))
        for token in ("1-Mar", "Mar-1", "01-Mar", "2-Sep", "Sep-2", "March-1", "1-Dec"):
            assert f'"{token}"' in flagged, f"{token} should be flagged"
        assert len(findings) == 7

    def test_the_family_the_symbol_belongs_to_is_named(self) -> None:
        (finding,) = check_gene_symbol_corruption("The 2-Sep transcript was induced.")
        assert "SEPTIN" in finding.detail
        assert finding.check == CHECK_GENE_SYMBOL
        assert finding.severity == SEVERITY_MAJOR

    def test_severity_is_major(self) -> None:
        findings = check_gene_symbol_corruption("Induced genes: 1-Mar and 1-Dec.")
        assert {f.severity for f in findings} == {SEVERITY_MAJOR}

    def test_location_names_the_line(self) -> None:
        text = "Introduction\n\nThe induced transcript 1-Mar was confirmed.\n"
        (finding,) = check_gene_symbol_corruption(text)
        assert finding.location == "line 3"

    def test_a_received_date_line_is_not_flagged(self) -> None:
        text = "Received: 1 Mar 2024; Accepted: 2-Sep-2024; Published: 14 Mar 2025.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_a_date_inside_the_references_section_is_not_flagged(self) -> None:
        """The reference entry carries no other cue — only the section guard.

        (Its wording avoids "revised", "accessed" and the rest on purpose, so
        this test binds the references-section skip and nothing else.)
        """
        text = (
            "Introduction\n"
            "Nothing corrupted here.\n"
            "\n"
            "References\n"
            "\n"
            "1. GEO series GSE12345. Matrix file rebuilt 1-Dec. NCBI, Bethesda.\n"
        )
        assert check_gene_symbol_corruption(text) == []

    def test_a_citation_line_with_a_year_is_not_flagged(self) -> None:
        text = "As Ziemann et al. (2016) noted, the 1-Mar entry recurs in supplements.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_a_date_carrying_its_year_is_not_flagged(self) -> None:
        for line in (
            "Sampling ran until 1-Mar 2024 without loss of viability.",
            "Sampling ran until 1-Mar, 2024 without loss of viability.",
            "Sampling ran until 1-Mar-2024 without loss of viability.",
        ):
            assert check_gene_symbol_corruption(line) == [], line

    def test_a_calendar_day_beyond_the_family_range_is_not_flagged(self) -> None:
        text = "Cultures were harvested on 22-Mar and again on 30-Sep of that season.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_a_month_with_no_gene_family_is_not_flagged(self) -> None:
        assert check_gene_symbol_corruption("Plates were read on 3-Aug and 1-Feb.") == []

    def test_exponent_corruption_is_flagged_in_identifier_context(self) -> None:
        text = "The gene identifier was recorded as 2.31E+13 in the supplementary table.\n"
        (finding,) = check_gene_symbol_corruption(text)
        assert "2.31E+13" in finding.detail
        assert "scientific notation" in finding.detail
        assert finding.severity == SEVERITY_MAJOR

    def test_an_exponent_with_no_identifier_context_is_not_flagged(self) -> None:
        """No unit either — so only the identifier-context guard keeps it quiet."""
        text = "The signal ratio reached 2.31E+13 during the calibration run.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_an_exponent_carrying_a_unit_is_not_flagged(self) -> None:
        text = "Gene expression assays reached 1.5E+06 cells/mL at the plateau.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_a_negative_exponent_is_never_flagged(self) -> None:
        text = "The gene decay constant was 4.2E-05 per hour in the accession set.\n"
        assert check_gene_symbol_corruption(text) == []

    def test_empty_text(self) -> None:
        assert check_gene_symbol_corruption("") == []


class TestReferenceIntegrity:
    NUMBERED = (
        "Introduction\n"
        "Sulfur metabolism matters [1]. Later work [3,4] agrees.\n"
        "A much later claim rests on [42].\n"
        "\n"
        "References\n"
        "\n"
        "1. Ziemann M. Gene name errors are widespread. doi:10.1186/s13059-016-1044-7\n"
        "2. Abeysooriya M. Lessons not learned. doi: 10.1371/journal.pcbi.1008984\n"
        "3. Someone A. A paper. doi:10.1000/xyz123\n"
        "4. Another B. A second paper. doi:10.1000/abc999\n"
    )

    def test_a_citation_past_the_reference_list_is_major(self) -> None:
        findings = check_reference_integrity(self.NUMBERED)
        over = [f for f in findings if "points past" in f.detail]
        assert len(over) == 1
        assert "[42]" in over[0].detail
        assert "4 entries" in over[0].detail
        assert over[0].severity == SEVERITY_MAJOR
        assert over[0].location == "line 3"
        assert over[0].check == CHECK_REFERENCE_INTEGRITY

    def test_citations_within_the_list_are_silent(self) -> None:
        findings = check_reference_integrity(self.NUMBERED)
        assert not [f for f in findings if "[1]" in f.detail or "[3]" in f.detail]

    def test_a_citation_range_is_expanded(self) -> None:
        text = (
            f"Body text citing [7{EN_DASH}9] and [11-12].\n"
            "\n"
            "References\n"
            "\n"
            "1. Only one entry. doi:10.1000/only\n"
        )
        over = [f for f in check_reference_integrity(text) if "points past" in f.detail]
        assert sorted(f.detail.split("[")[1].split("]")[0] for f in over) == [
            "11",
            "12",
            "7",
            "8",
            "9",
        ]

    def test_no_reference_section_means_the_count_check_stays_quiet(self) -> None:
        text = "A body with citations [1], [3,4] and [42] but no reference list at all.\n"
        assert check_reference_integrity(text) == []

    def test_a_malformed_doi_is_minor(self) -> None:
        text = "See doi: 10.1371.journal.pbio.3002439 for the published version.\n"
        (finding,) = check_reference_integrity(text)
        assert "not DOI syntax" in finding.detail
        assert finding.severity == SEVERITY_MINOR

    def test_a_short_doi_prefix_is_flagged(self) -> None:
        text = "Archived at https://doi.org/10.13/journal.pbio.3002439 for the record.\n"
        (finding,) = check_reference_integrity(text)
        assert "not DOI syntax" in finding.detail

    def test_a_valid_doi_is_not_flagged(self) -> None:
        text = (
            "The published twin is doi:10.1371/journal.pbio.3002439 and the method is\n"
            "https://doi.org/10.1186/s13059-016-1044-7 (accessed 2025).\n"
        )
        assert check_reference_integrity(text) == []

    def test_a_valid_doi_with_trailing_prose_punctuation_is_not_flagged(self) -> None:
        text = "See doi:10.1371/journal.pcbi.1008984, which reports the survey.\n"
        assert check_reference_integrity(text) == []

    def test_a_decimal_number_is_not_mistaken_for_a_doi(self) -> None:
        text = "Cultures were supplemented with 10.5 mM glucose and 10.1371 g of salt.\n"
        assert check_reference_integrity(text) == []

    def test_a_duplicate_reference_doi_is_minor(self) -> None:
        text = (
            "Body.\n"
            "\n"
            "References\n"
            "\n"
            "1. Someone A. A paper. doi:10.1000/xyz123\n"
            "2. Another B. A second paper. doi:10.1000/abc999\n"
            "3. Duplicate C. The same record again. doi:10.1000/XYZ123\n"
        )
        dupes = [f for f in check_reference_integrity(text) if "duplicate" in f.detail]
        assert len(dupes) == 1
        assert "10.1000/xyz123" in dupes[0].detail
        assert dupes[0].location == "line 7"
        assert dupes[0].severity == SEVERITY_MINOR

    def test_a_doi_cited_twice_in_the_body_is_not_a_duplicate_entry(self) -> None:
        text = (
            "We follow doi:10.1000/xyz123 here and doi:10.1000/xyz123 again there.\n"
            "\n"
            "References\n"
            "\n"
            "1. Someone A. A paper. doi:10.1000/xyz123\n"
        )
        assert not [f for f in check_reference_integrity(text) if "duplicate" in f.detail]

    def test_empty_text(self) -> None:
        assert check_reference_integrity("") == []


class TestStatSanity:
    def test_a_p_value_of_exactly_zero_is_flagged(self) -> None:
        (finding,) = check_stat_sanity("The effect was significant (p = 0) throughout.")
        assert "exactly zero" in finding.detail
        assert finding.severity == SEVERITY_MINOR
        assert finding.check == CHECK_STAT_SANITY

    def test_a_padded_zero_p_value_is_flagged(self) -> None:
        (finding,) = check_stat_sanity("The contrast gave p=0.000 across replicates.")
        assert '"p=0.000"' in finding.detail

    def test_a_small_but_real_p_value_is_not_flagged(self) -> None:
        assert check_stat_sanity("The contrast gave p = 0.0001 in every replicate.") == []

    def test_a_bounded_p_value_is_not_flagged(self) -> None:
        text = f"Significance was taken at p < 0.05, with p {LESS_EQUAL} 0.01 for the trend.\n"
        assert check_stat_sanity(text) == []

    def test_a_p_value_above_one_is_flagged(self) -> None:
        (finding,) = check_stat_sanity("An impossible value of p = 1.4 was reported.")
        assert "above 1" in finding.detail

    def test_a_p_value_above_one_with_a_unicode_operator_is_flagged(self) -> None:
        (finding,) = check_stat_sanity(f"Values of p {GREATER_EQUAL} 2 were reported.")
        assert "above 1" in finding.detail

    def test_ph_is_never_read_as_a_p_value(self) -> None:
        text = "The pH was adjusted to 3 and the final pH = 6.9 in buffered medium.\n"
        assert check_stat_sanity(text) == []

    def test_a_p_carrying_a_unit_is_not_a_p_value(self) -> None:
        assert check_stat_sanity("The medium contained P = 100 mM phosphate.") == []

    def test_a_partition_summing_far_over_one_hundred_is_flagged(self) -> None:
        text = "Of the samples, 60% were resistant, 55% were tolerant and 40% sensitive.\n"
        (finding,) = check_stat_sanity(text)
        assert "155%" in finding.detail
        assert finding.severity == SEVERITY_MINOR

    def test_a_partition_that_sums_correctly_is_not_flagged(self) -> None:
        text = "The fraction consisted of glucose (88%), galactose (6%) and mannose (5%).\n"
        assert check_stat_sanity(text) == []

    def test_rounding_inside_the_tolerance_is_not_flagged(self) -> None:
        text = "Of the samples, 34% were A, 34% were B and 34% were C after rounding.\n"
        assert check_stat_sanity(text) == []

    def test_reagent_purities_are_not_a_partition(self) -> None:
        """Four percentages summing to 389% on one line, and no partition claim."""
        text = (
            "One liter contained peptone (>95%), glucose monohydrate (>99.5%), "
            "magnesium sulfate (>99%) and ethanol (96% v/v), each weighed separately.\n"
        )
        assert check_stat_sanity(text) == []

    def test_two_shares_alone_are_not_treated_as_a_partition(self) -> None:
        """Two "of the isolates" claims about different cohorts must not be summed."""
        text = (
            "Resistance accounted for 60% of the isolates here, while the same trait "
            "accounted for 55% of the isolates in the second cohort.\n"
        )
        assert check_stat_sanity(text) == []

    def test_an_error_bar_is_read_as_its_share_not_as_a_share_of_its_own(self) -> None:
        """The real oleaginous-yeast sentence: 88.83 + 5.50 + 4.80 = 99.13%."""
        text = (
            f"The carbohydrate fraction consisted of glucose (88.83 {PLUS_MINUS} 4.87%), "
            f"galactose (5.50 {PLUS_MINUS} 1.50%) and mannose (4.80 {PLUS_MINUS} 1.57%).\n"
        )
        assert check_stat_sanity(text) == []

    def test_a_partition_written_with_error_bars_still_sums(self) -> None:
        text = (
            f"Of the samples, 60.0 {PLUS_MINUS} 2.0% were resistant, "
            f"55.0 {PLUS_MINUS} 3.0% were tolerant and 40.0 {PLUS_MINUS} 1.0% sensitive.\n"
        )
        (finding,) = check_stat_sanity(text)
        assert "155%" in finding.detail

    def test_two_partitions_joined_by_a_semicolon_are_not_summed(self) -> None:
        text = (
            "Strain A consisted of mannose (92.8%), glucose (7.2%) and traces (0.1%); "
            "strain B consisted of mannose (83%), glucose (10%) and galactose (7%).\n"
        )
        assert check_stat_sanity(text) == []

    def test_empty_text(self) -> None:
        assert check_stat_sanity("") == []


class TestRunDeterministicLens:
    MIXED = (
        "Results\n"
        "The induced transcript 1-Mar was confirmed (p = 0) in all replicates.\n"
        "A later claim rests on [42].\n"
        "\n"
        "References\n"
        "\n"
        "1. Someone A. A paper. doi:10.1000/xyz123\n"
    )

    def test_empty_text_finds_nothing(self) -> None:
        assert run_deterministic_lens("") == []

    def test_whitespace_only_finds_nothing(self) -> None:
        assert run_deterministic_lens("\n\n   \n\t\n") == []

    def test_all_three_checks_are_reachable_in_one_pass(self) -> None:
        findings = run_deterministic_lens(self.MIXED)
        assert {f.check for f in findings} == {
            CHECK_GENE_SYMBOL,
            CHECK_REFERENCE_INTEGRITY,
            CHECK_STAT_SANITY,
        }

    def test_findings_are_ordered_by_line(self) -> None:
        findings = run_deterministic_lens(self.MIXED)
        lines = [int(f.location.removeprefix("line ")) for f in findings]
        assert lines == sorted(lines)

    def test_every_finding_carries_the_full_shape(self) -> None:
        for finding in run_deterministic_lens(self.MIXED):
            assert finding.check
            assert finding.severity in {SEVERITY_MAJOR, SEVERITY_MINOR}
            assert finding.detail
            assert finding.location.startswith("line ")

    def test_same_input_twice_gives_identical_output(self) -> None:
        first = run_deterministic_lens(self.MIXED)
        second = run_deterministic_lens(self.MIXED)
        assert [f.model_dump() for f in first] == [f.model_dump() for f in second]

    def test_the_lens_reaches_no_model(self) -> None:
        """The whole point of this lens: nothing here can call out to anything.

        Its only imports are the standard library's `re` and the finding schema,
        and no provider name appears in the source at all.
        """
        from peerpanel.agents import deterministic_lens as lens

        source = Path(lens.__file__).read_text(encoding="utf-8")
        imports = sorted(
            line.strip()
            for line in source.splitlines()
            if line.startswith(("import ", "from ")) and "__future__" not in line
        )
        assert imports == ["from .schemas import DeterministicFinding", "import re"]
        for forbidden in ("ChatProvider", "ollama", "httpx", "openai", "requests", "urllib"):
            assert forbidden not in source, f"the deterministic lens must not reach {forbidden}"


class TestRealManuscripts:
    def test_the_pinned_table_covers_every_committed_manuscript(self) -> None:
        on_disk = {path.name for path in MANUSCRIPTS_DIR.glob("*.txt")}
        assert on_disk == set(EXPECTED_REAL_FINDINGS), (
            "a manuscript was added or removed — re-run the lens and re-pin "
            "EXPECTED_REAL_FINDINGS with what it actually finds"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_REAL_FINDINGS))
    def test_the_lens_runs_and_finds_what_it_finds(self, name: str) -> None:
        _, body = read_manuscript(MANUSCRIPTS_DIR / name)
        assert len(body) > 20_000, "the body must be the real manuscript, not a stub"
        findings = run_deterministic_lens(body)
        assert len(findings) == EXPECTED_REAL_FINDINGS[name], (
            f"{name} now yields {[f.detail for f in findings]}"
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_REAL_FINDINGS))
    def test_the_real_run_is_deterministic(self, name: str) -> None:
        _, body = read_manuscript(MANUSCRIPTS_DIR / name)
        first = [f.model_dump() for f in run_deterministic_lens(body)]
        second = [f.model_dump() for f in run_deterministic_lens(body)]
        assert first == second

    def test_the_clean_result_is_not_a_dead_check(self) -> None:
        """A null on real text only means something if a planted defect fires.

        The same real body, with one corrupted sentence spliced in, must be
        caught — otherwise the zeros above would prove nothing.
        """
        _, body = read_manuscript(MANUSCRIPTS_DIR / "met17-auxotroph.txt")
        planted = body + (
            "\n\nThe supplementary gene list named 1-Mar and 2-Sep among the induced\n"
            "transcripts, with the accession recorded as 2.31E+13 (p = 0).\n"
        )
        findings = run_deterministic_lens(planted)
        flagged = details(list(findings))
        assert '"1-Mar"' in flagged
        assert '"2-Sep"' in flagged
        assert "2.31E+13" in flagged
        assert "exactly zero" in flagged
