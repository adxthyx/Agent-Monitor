from agentmon.models import load_suite


def test_ground_truth_paths_resolve_against_the_profile_file(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "facts.md").write_text("Refunds within 30 days.")
    (shared / "profile.yaml").write_text("purpose: p\nground_truth: [facts.md]\n")
    (shared / "probes.yaml").write_text("- {id: p1, class: canary, turns: [hi], expect: {outcome: o}}\n")
    suite_dir = tmp_path / "suites"
    suite_dir.mkdir()
    (suite_dir / "suite.yaml").write_text(
        "id: s\ntarget: {id: t, name: t, adapter: {kind: openai_chat, url: 'http://x'}}\n"
        "profile: ../shared/profile.yaml\nprobes: [../shared/probes.yaml]\n"
    )
    suite = load_suite(suite_dir / "suite.yaml")
    assert suite.profile.ground_truth_text == "Refunds within 30 days."
