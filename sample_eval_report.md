# Fusion evaluation — demo · 16 tasks (draco+longhorizon) · rule grader

**Verdict: Beyond-frontier lift — WORTH IT**  
Beyond-frontier lift. Fusion scored 97.2 vs the 38.8 baseline (+58.4 pts) at ~4.0x cost and ~1.0x latency. Worth it for high-stakes / ambiguous work: +58.4 pts at ~4.0x cost and ~1.0x latency. Safety gain: Fusion triggered 7 fewer penalty/error criteria than the baseline — valuable in medical/legal/compliance. Long-horizon: reliability is comparable — pass^3 75% vs 75%. Mean milestone completion 100% vs 92%; decide on cost, not reliability.

## Setup
- Panel: demo/panel-a, demo/panel-b
- Judge / synthesizer (caller): demo/judge / demo/judge
- Solo baseline: demo/baseline
- Grader: rule · passes/task: 3 · tasks: 16 · tools: False

## Headline scores (0–100, mean of passes)
| | Fusion | Baseline | Lift |
|---|---:|---:|---:|
| Overall | 97.16 | 38.76 | +58.4 |

## By rubric category
| Category | Fusion | Baseline |
|---|---:|---:|
| factual_accuracy | 94.79 | 42.09 |
| breadth_depth | 100.0 | 38.12 |
| presentation | 100.0 | 57.14 |
| citation | 100.0 | 0.0 |

## By domain
| Domain | n | Fusion | Baseline |
|---|---:|---:|---:|
| academic_research | 1 | 100.0 | 11.11 |
| finance | 1 | 100.0 | 45.45 |
| law | 1 | 100.0 | 20.0 |
| medicine | 1 | 100.0 | 10.0 |
| technology | 1 | 100.0 | 45.45 |
| ux_design | 1 | 100.0 | 55.56 |
| general_knowledge | 1 | 100.0 | 44.44 |
| needle_retrieval | 1 | 100.0 | 0.0 |
| personalized_assistance | 1 | 100.0 | 0.0 |
| product_comparison | 1 | 100.0 | 50.0 |
| operations | 1 | 54.55 | 81.82 |
| support_agent | 1 | 100.0 | 50.0 |
| software_migration | 1 | 100.0 | 100.0 |
| semiconductor | 3 | 100.0 | 35.43 |

## Cost & latency
- Cost multiplier: ~4.0x (1280 vs 320 tokens)
- Latency multiplier: n/a (demo) (0ms vs 0ms)
- Quality-per-dollar: 0.627 (relative score ÷ cost multiplier; >1 favours Fusion)
- Error/penalty criteria triggered: Fusion 1 vs baseline 8

## Long-horizon reliability
Metrics follow tau-bench (pass^k = success on ALL k passes) and RoadmapBench-style milestone completion. k = 3.
| Metric | Fusion | Baseline |
|---|---:|---:|
| Milestone completion (%) | 100.0 | 91.61 |
| pass^3 (% tasks solved every pass) | 75.0 | 75.0 |
| pass@1 (% passes solved) | 75.0 | 75.0 |
| Drift / coherence errors | 1 | 1 |

## Recommendations
- Worth it for high-stakes / ambiguous work: +58.4 pts at ~4.0x cost and ~1.0x latency.
- Safety gain: Fusion triggered 7 fewer penalty/error criteria than the baseline — valuable in medical/legal/compliance.
- Long-horizon: reliability is comparable — pass^3 75% vs 75%. Mean milestone completion 100% vs 92%; decide on cost, not reliability.
