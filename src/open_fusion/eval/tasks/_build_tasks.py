"""
_build_tasks.py - regenerate the suite JSON files (draco / longhorizon / semiconductor).

Run:  python -m open_fusion.eval.tasks._build_tasks
Emits draco_tasks.json, longhorizon_tasks.json, semiconductor_dram_tasks.json next to
this file. Kept in-tree for auditability: every rubric criterion and every demo answer
is generated from one compact spec so they cannot silently drift apart.

A criterion spec is a tuple:
    (id, category, weight, any_patterns, demo_phrase, met, critical, order)
- any_patterns : regex(es) RuleGrader matches (case-insensitive); a substring is valid.
- demo_phrase  : literal text emitted into a demo answer when that variant 'meets' it
                 (must satisfy any_patterns). For penalties, it's the error phrasing.
- met          : subset of {"f","b"} -> emitted into the fused / baseline demo answers.
- critical     : long-horizon milestone that must hold for pass^k.
- order        : long-horizon milestone sequence index.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent

F = "factual_accuracy"
B = "breadth_depth"
P = "presentation"
C = "citation"

CITE = ["https?://", r"\[[0-9]+\]"]
CITE_PHRASE = "Primary sources: https://example.org/report [1]."
STRUCT = ["##", "summary", "trade-off", "trade-offs"]
STRUCT_PHRASE = "## Summary\nClear trade-offs and a concrete recommendation follow."


def crit(cid, cat, weight, anyp, demo, met, critical=False, order=None):
    return {"id": cid, "category": cat, "weight": weight, "any": anyp,
            "demo": demo, "met": set(met), "critical": critical, "order": order}


def build(task_id, domain, prompt, specs, *, kind="draco", horizon=None, excluded=None):
    rubric = []
    fused, baseline = [], []
    intended = {}
    for s in specs:
        c = {"id": s["id"], "category": s["category"], "weight": s["weight"],
             "description": s["demo"][:90], "check": {"any": s["any"]}}
        if s.get("critical"):
            c["critical"] = True
        if s.get("order") is not None:
            c["order"] = s["order"]
        rubric.append(c)
        intended[s["id"]] = {"f": "f" in s["met"], "b": "b" in s["met"]}
        if "f" in s["met"]:
            fused.append(s["demo"])
        if "b" in s["met"]:
            baseline.append(s["demo"])
    fused_text = " ".join(fused) if fused else "No substantive content."
    base_text = " ".join(baseline) if baseline else "A brief, partial answer."
    return {
        "id": task_id, "domain": domain, "kind": kind, "prompt": prompt,
        "horizon": horizon, "excluded_domains": excluded or [],
        "rubric": rubric, "_intended": intended,
        "demo": {"fused": fused_text, "baseline": base_text,
                 "panel": base_text + " (panel member perspective)"},
    }


# ============================================================================ #
# DRACO suite — the article's 10 domains, one task each.
# fused meets every positive and no penalty; baseline meets a subset and trips
# a penalty on the safety-sensitive domains. (Demo magnitudes are illustrative.)
# ============================================================================ #
DRACO = [
    build("draco_academic", "academic_research",
          "Summarize the current evidence on whether spaced repetition improves long-term retention, and the main methodological caveats.",
          [crit("f1", F, 3, ["spaced repetition", "spacing effect"], "The spacing effect is well replicated for long-term retention.", "fb"),
           crit("f2", F, 2, ["retrieval practice", "testing effect"], "Retrieval practice (the testing effect) compounds the benefit.", "f"),
           crit("b1", B, 2, ["effect size", "heterogeneity", "moderators"], "Effect sizes vary by material and spacing schedule (moderators).", "f"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "fb"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -3, ["proves you never need to review", "guarantees perfect recall"], "It proves you never need to review again.", "b")]),

    build("draco_finance", "finance",
          "What are the strongest arguments for and against a broad carbon tax, and the key design choices that determine its effects?",
          [crit("f1", F, 3, ["price externalit", "marginal damage"], "A carbon tax prices the externality at its marginal damage.", "fb"),
           crit("f2", F, 2, ["revenue recycling", "rebate", "dividend"], "Revenue recycling or a dividend offsets regressivity.", "f"),
           crit("b1", B, 2, ["regressiv", "border adjustment", "leakage"], "Without border adjustment, leakage and regressivity bite.", "f"),
           crit("b2", B, 2, ["vs cap-and-trade", "price certainty"], "Versus cap-and-trade it gives price certainty, not quantity certainty.", "fb"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f")]),

    build("draco_law", "law",
          "Explain the doctrine of fair use in US copyright and how courts weigh the four factors.",
          [crit("f1", F, 3, ["four factors", "17 U.S.C", "section 107"], "Fair use weighs the four statutory factors of section 107.", "fb"),
           crit("f2", F, 2, ["transformative", "purpose and character"], "The first factor turns on transformative purpose and character.", "f"),
           crit("f3", F, 2, ["market effect", "fourth factor"], "The fourth factor is the effect on the market for the work.", "f"),
           crit("b1", B, 2, ["fact-specific", "no bright-line"], "It is fact-specific with no bright-line rule.", "fb"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -3, ["fair use is a blanket right", "always legal to copy"], "Fair use is a blanket right to copy anything.", "b")]),

    build("draco_medicine", "medicine",
          "What is the current first-line approach to managing type 2 diabetes in adults, and the monitoring that matters most?",
          [crit("f1", F, 3, ["metformin"], "Metformin remains first-line pharmacotherapy.", "fb"),
           crit("f2", F, 2, ["lifestyle", "diet", "exercise"], "Lifestyle change (diet and exercise) is foundational.", "fb"),
           crit("f3", F, 2, ["HbA1c", "A1c"], "Monitor HbA1c to target.", "f"),
           crit("b1", B, 2, ["SGLT2", "GLP-1", "cardiorenal"], "Add SGLT2/GLP-1 agents for cardiorenal benefit.", "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -4, ["stop taking insulin", "ignore your doctor"], "You can stop taking insulin on your own.", "b")]),

    build("draco_technology", "technology",
          "Compare REST and gRPC for a new internal microservice API, and when to choose each.",
          [crit("f1", F, 3, ["protobuf", "HTTP/2", "streaming"], "gRPC uses protobuf over HTTP/2 with streaming.", "fb"),
           crit("f2", F, 2, ["human-readable", "broad tooling", "browser"], "REST is human-readable with broad tooling and browser support.", "f"),
           crit("b1", B, 2, ["latency", "schema evolution", "contract"], "Trade-offs: latency, schema evolution, contract enforcement.", "f"),
           crit("b2", B, 2, ["when to choose"], "When to choose each: gRPC for internal high-throughput and REST for public APIs.", "fb"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f")]),

    build("draco_ux", "ux_design",
          "What are evidence-based principles for designing an onboarding flow that reduces drop-off?",
          [crit("f1", F, 3, ["progressive disclosure", "reduce cognitive load"], "Use progressive disclosure to reduce cognitive load.", "fb"),
           crit("f2", F, 2, ["activation", "aha moment", "time to value"], "Drive users to the activation / time-to-value moment fast.", "f"),
           crit("b1", B, 2, ["measure", "funnel", "drop-off"], "Measure the funnel and instrument drop-off points.", "fb"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f")]),

    build("draco_general", "general_knowledge",
          "Explain why the sky is blue and why sunsets are red, at a level a curious adult can follow.",
          [crit("f1", F, 3, ["Rayleigh scattering"], "Rayleigh scattering favors short blue light.", "fb"),
           crit("f2", F, 2, ["longer path", "more scattering", "horizon"], "At sunset light travels a longer path, scattering out blue.", "f"),
           crit("b1", B, 2, ["wavelength", "inverse", "fourth power"], "Scattering scales inversely with the fourth power of wavelength.", "f"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "fb"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f")]),

    build("draco_needle", "needle_retrieval",
          "From a long policy document, extract the exact data-retention period for customer logs and the section that states it.",
          [crit("f1", F, 4, ["90 days", "ninety days"], "Customer logs are retained for 90 days.", "fb"),
           crit("f2", F, 2, ["section 7.2", "retention clause"], "Stated in section 7.2 (retention clause).", "f"),
           crit("b1", B, 1, ["verbatim", "quote"], "Quotes the clause verbatim.", "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -4, ["30 days", "indefinitely"], "Logs are retained indefinitely.", "b")]),

    build("draco_personal", "personalized_assistance",
          "Given a vegetarian user with a nut allergy training for a 10K, suggest a weekly meal framework.",
          [crit("f1", F, 3, ["vegetarian", "no nuts", "nut-free"], "Plan is vegetarian and strictly nut-free.", "fb"),
           crit("f2", F, 2, ["protein", "legumes", "tofu"], "Protein from legumes, tofu, dairy/eggs.", "f"),
           crit("b1", B, 2, ["carbohydrate", "endurance", "training load"], "Carbohydrate timing matched to endurance training load.", "f"),
           crit("p1", P, 1, STRUCT, STRUCT_PHRASE, "fb"),
           crit("n1", F, -4, ["almond", "peanut", "cashew"], "Add a handful of almonds for crunch.", "b")]),

    build("draco_product", "product_comparison",
          "Compare ridge, lasso, and elastic-net regression and when to prefer each.",
          [crit("f1", F, 3, ["L2", "ridge shrinks"], "Ridge applies L2 shrinkage, keeping all features.", "fb"),
           crit("f2", F, 2, ["L1", "lasso", "sparse", "feature selection"], "Lasso (L1) yields sparsity / feature selection.", "fb"),
           crit("f3", F, 2, ["elastic-net", "combines", "correlated"], "Elastic-net combines L1+L2, good for correlated features.", "f"),
           crit("b1", B, 2, ["when to prefer", "high-dimensional"], "Prefer lasso/elastic-net in high-dimensional sparse settings.", "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f")]),
]


# ============================================================================ #
# Long-horizon suite — modeled on tau-bench (pass^k), RoadmapBench (subtasks),
# RetailBench / Vending-Bench (coherence over many periods, recovery).
# Demo is engineered for an HONEST, article-aligned outcome: Fusion and a strong
# single model are COMPARABLE on reliability (pass^k), with Fusion slightly ahead
# on milestone coverage -> verdict says "decide on cost, not reliability".
#   vending : fused trips a coherence drift (critical) -> fused FAILS, baseline PASSES
#   policy  : baseline violates policy (critical)       -> fused PASSES, baseline FAILS
#   roadmap : both complete the critical subtasks       -> both PASS
# => pass^k: fusion 2/3, baseline 2/3 (comparable); milestones: fusion higher.
# ============================================================================ #
LONGHORIZON = [
    build("lh_vending", "operations",
          "Operate a vending-machine business over 12 simulated weeks: set restock and pricing each week, track cash flow, and recover from a supplier delay in week 4. Keep the plan internally consistent across weeks.",
          [crit("m1", B, 3, ["weekly plan", "week-by-week", "per week"], "A week-by-week plan covers all 12 weeks.", "fb", critical=True, order=1),
           crit("m2", F, 3, ["cash flow", "cash-flow", "balance"], "Cash-flow balance is tracked each week.", "fb", critical=True, order=2),
           crit("m3", F, 3, ["recover", "supplier delay", "contingency"], "Recovers from the week-4 supplier delay with a contingency.", "fb", critical=True, order=3),
           crit("m4", B, 2, ["price elasticity", "demand"], "Pricing reflects demand/elasticity.", "f", order=4),
           crit("d1", F, -5, ["restock from the delayed supplier in week 4", "ignores the week-4 delay"], "Plans to restock from the delayed supplier in week 4 (contradicts the delay).", "f", critical=True)],
          kind="longhorizon", horizon="12 weekly steps"),

    build("lh_policy", "support_agent",
          "As a support agent bound by a refund policy (refunds only within 30 days with proof of purchase), handle a multi-step case: verify eligibility, apply the policy exactly, update the record, and reply to the customer.",
          [crit("m1", F, 3, ["verify", "eligibility", "within 30 days", "proof of purchase"], "Verifies eligibility: within 30 days and proof of purchase.", "fb", critical=True, order=1),
           crit("m2", F, 3, ["apply the policy", "approve", "deny"], "Applies the policy exactly (approve/deny per rule).", "fb", critical=True, order=2),
           crit("m3", B, 2, ["update", "record", "ticket"], "Updates the record/ticket.", "fb", order=3),
           crit("m4", P, 2, ["reply", "communicate", "customer message"], "Communicates clearly to the customer.", "fb", order=4),
           crit("d1", F, -5, ["refund outside the 30-day window", "waive the policy"], "Issues a refund outside the 30-day window without proof.", "b", critical=True)],
          kind="longhorizon", horizon="multi-step tool+policy episode"),

    build("lh_roadmap", "software_migration",
          "Plan and execute a 5-subtask migration of a service from framework v2 to v3, preserving behavior and passing tests.",
          [crit("m1", B, 2, ["inventory", "audit", "breaking changes"], "Subtask 1: inventory breaking changes.", "fb", critical=True, order=1),
           crit("m2", B, 2, ["dependency", "upgrade order"], "Subtask 2: order dependency upgrades.", "fb", critical=True, order=2),
           crit("m3", F, 2, ["adapter", "shim", "compatibility layer"], "Subtask 3: add a compatibility shim.", "fb", critical=True, order=3),
           crit("m4", F, 2, ["migrate", "incrementally", "module by module"], "Subtask 4: migrate module by module.", "fb", critical=True, order=4),
           crit("m5", F, 2, ["tests pass", "regression suite", "green"], "Subtask 5: full regression suite green.", "fb", critical=True, order=5),
           crit("d1", F, -5, ["skips the regression", "big-bang rewrite breaks"], "Does a big-bang rewrite that breaks tests.", "", critical=True)],
          kind="longhorizon", horizon="5 ordered subtasks"),
]


# ============================================================================ #
# Semiconductor / DRAM suite — grounded in the 2026 memory supercycle facts.
# 2 deep-research (DRACO) tasks + 1 long-horizon capacity-ramp planning task.
# ============================================================================ #
SEMI = [
    build("dram_landscape", "semiconductor",
          "Assess the 2026 DRAM/HBM competitive landscape and the HBM3E->HBM4 transition: who leads, the key technology shifts, and the main supply and geopolitical risks.",
          [crit("f1", F, 3, ["SK hynix", "SK Hynix"], "SK hynix leads HBM with roughly half or more of the market.", "fb"),
           crit("f2", F, 3, ["HBM4", "mass production"], "HBM4 enters mass production in 2026 (11+ Gbps, >2 TB/s, 12/16-hi hybrid bonding).", "fb"),
           crit("f3", F, 2, ["HBM3E"], "HBM3E remains the flagship, ~two-thirds of 2026 HBM shipments.", "f"),
           crit("f4", F, 2, ["DDR5", "6400"], "DDR5 at 6400 MT/s is mainstream; DDR4 production was extended on price inversion.", "f"),
           crit("b1", B, 2, ["CXMT", "export control", "geopolitical"], "Supply/geopolitical risk: China (CXMT) ramp and US export controls.", "f"),
           crit("b2", B, 2, ["custom HBM", "base die", "TSMC"], "Custom HBM with a logic base die on an advanced foundry node (TSMC).", "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -4, ["Intel leads the HBM market", "HBM4 is slower than HBM3E"], "Intel leads the HBM market and HBM4 is slower than the prior generation.", "b")]),

    build("dram_roadmaps", "semiconductor",
          "Compare the three majors' DRAM process-node and HBM roadmaps and what differentiates their HBM4 approaches.",
          [crit("f1", F, 3, ["1-gamma", "1\u03b3", "1c", "1c DRAM"], "Micron pushes 1-gamma; Samsung uses 1c DRAM for HBM4.", "fb"),
           crit("f2", F, 2, ["EUV"], "All three rely on EUV at the leading edge.", "f"),
           crit("f3", F, 2, ["hybrid bonding", "TSV", "16-hi", "12-hi"], "HBM4 uses hybrid bonding for 12/16-hi stacks.", "fb"),
           crit("f4", F, 2, ["yield"], "Yield ramp is the key differentiator on HBM4.", "f"),
           crit("b1", B, 2, ["base die", "foundry", "TSMC", "logic die"], "SK hynix pairs the HBM4 logic base die with a TSMC foundry node.", "f"),
           crit("c1", C, 1, CITE, CITE_PHRASE, "f"),
           crit("n1", F, -4, ["DDR4 is fully discontinued", "no one makes DDR4"], "DDR4 is fully discontinued by all vendors.", "b")]),

    build("dram_ramp", "semiconductor",
          "Plan SK hynix's HBM4 capacity ramp quarter-by-quarter from 2026 H1 to 2027 H1: wafer allocation between HBM and general DRAM, packaging/hybrid-bonding capacity, customer commitments (NVIDIA Rubin), capex phasing, and a contingency for a 2027 oversupply risk. Keep the plan internally consistent across quarters.",
          [crit("m1", F, 3, ["wafer allocation", "3x wafers", "three times", "general DRAM"], "Allocates wafers between HBM (uses ~3x wafers) and general DRAM.", "fb", critical=True, order=1),
           crit("m2", F, 3, ["hybrid bonding", "packaging", "TSV", "advanced packaging"], "Ramps advanced packaging / hybrid-bonding capacity.", "fb", critical=True, order=2),
           crit("m3", B, 2, ["Rubin", "NVIDIA", "customer commitment"], "Secures customer commitments (NVIDIA Rubin).", "fb", order=3),
           crit("m4", B, 2, ["capex", "phasing", "fab"], "Phases capex across the M15X / new fabs.", "f", order=4),
           crit("m5", F, 3, ["oversupply", "2027", "contingency"], "Includes a 2027 oversupply contingency.", "fb", critical=True, order=5),
           crit("d1", F, -5, ["reallocates all wafers to general DRAM", "contradicts the HBM wafer plan"], "Later reallocates all wafers to general DRAM (contradicts the HBM plan).", "", critical=True)],
          kind="longhorizon", horizon="5 quarters (2026H1->2027H1)",
          excluded=["counterpointresearch.com", "trendforce.com"]),
]


def _verify():
    """Assert each demo answer triggers EXACTLY the criteria it is meant to.
    Catches both missed matches (phrase != its own regex) and cross-criterion
    false positives (a generic pattern matched another criterion's phrase)."""
    from ..rubric import evaluate_check
    problems = []
    for tasks in (DRACO, LONGHORIZON, SEMI):
        for t in tasks:
            intended = t["_intended"]
            for variant, key in (("fused", "f"), ("baseline", "b")):
                ans = t["demo"][variant]
                for c in t["rubric"]:
                    got = evaluate_check(ans, c["check"])
                    want = intended[c["id"]][key]
                    if got != want:
                        problems.append(f"{t['id']}:{c['id']} {variant} got={got} want={want}")
    if problems:
        print(f"VERIFY: {len(problems)} PROBLEM(S):")
        for p in problems:
            print("  -", p)
    else:
        print("verify: all demo answers match their detectors exactly (clean)")
    return not problems


def _serialize(tasks):
    out = []
    for t in tasks:
        t = {k: v for k, v in t.items() if k != "_intended"}
        out.append(t)
    return {"tasks": out}


def main():
    ok = _verify()
    if not ok:
        raise SystemExit("refusing to write task files: demo answers do not match detectors")
    (HERE / "draco_tasks.json").write_text(
        json.dumps(_serialize(DRACO), indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "longhorizon_tasks.json").write_text(
        json.dumps(_serialize(LONGHORIZON), indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "semiconductor_dram_tasks.json").write_text(
        json.dumps(_serialize(SEMI), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(DRACO)} draco, {len(LONGHORIZON)} longhorizon, {len(SEMI)} semiconductor tasks")


if __name__ == "__main__":
    main()
