# Multi-Model Fusion Superiority Benchmark

- Panel models: kimi-k2.6, deepseek-v4-flash, doubao-seed-2.0-pro
- Judge: glm-5.2 · grader: rule
- Tasks: 9 · grading passes/task: 1 · tie margin: ±1.0pts

## Verdict
- **vs_best_single**: Fusion trails the best single model by 4.62 points; the pipeline is not helping on this task set.
- **vs_oracle**: Fusion does not beat the oracle ceiling (-10.19 points). The best single model, chosen per task, still edges out fusion.

## Arm Scores (0–100, normalized)
| Arm | Model(s) | Mean Score |
|---|---|---:|
| solo_0 | kimi-k2.6 | **74.43** |
| solo_1 | deepseek-v4-flash | **60.29** |
| solo_2 | doubao-seed-2.0-pro | **66.5** |
| solo_best | kimi-k2.6 | **82.66** |
| fusion | kimi-k2.6, deepseek-v4-flash, doubao-seed-2.0-pro | **78.04** |
| oracle_best_single | ORACLE best-of-N (post-hoc per-task max) | **88.23** |

## Head-to-Head vs Fusion
| Opponent | Wins | Ties | Losses | Win Rate | Mean Lift | 95% CI | p | Cohen's d |
|---|---:|---:|---:|---|---:|---|---:|---:|
| best single model | 2 | 5 | 2 | 22% [19%–73%] | -4.62 | [-24.69, +9.18] | 0.642 | -0.16 |
| single model #0 (kimi-k2.6) | 2 | 6 | 1 | 22% [27%–81%] | +3.61 | [-3.92, +11.42] | 0.432 | 0.27 |
| single model #1 (deepseek-v4-flash) | 4 | 4 | 1 | 44% [35%–88%] | +17.75 | [-13.58, +47.02] | 0.220 | 0.36 |
| single model #2 (doubao-seed-2.0-pro) | 2 | 7 | 0 | 22% [27%–81%] | +11.53 | [+0.00, +27.58] | 0.198 | 0.46 |
| oracle best-of-N (single-model ceiling) | 0 | 7 | 2 | 0% [12%–65%] | -10.19 | [-26.65, +0.00] | 0.176 | -0.41 |

## Fusion Gain by Task Domain
| Domain | N | Fusion Mean | Best Solo Mean | Oracle Mean | Lift vs Best Solo | Lift vs Oracle | W/T/L vs Best |
|---|---:|---:|---:|---:|---:|---:|---|
| logical reasoning | 9 | **78.04** | 88.23 | 88.23 | -10.19 | -10.19 | 0/7/2 |

### Domain Key Observations
- Domains where models have complementary blind spots (multi-step reasoning, security, cross-domain knowledge) show the largest fusion gains.
- Pure mathematical calculation tasks may show smaller gains if any single model already computes the answer correctly.
- Code generation benefits from fusion when multiple edge cases need to be caught that different models miss individually.

## Interpretation Guide
- **p < 0.05** (marked ✓): statistically significant at the 95% confidence level (paired bootstrap)
- **Cohen's d**: 0.2 = small effect, 0.5 = medium, 0.8+ = large effect
- **Oracle best-of-N**: for each task, take the highest score ANY single model got. This is the theoretical upper bound of any single-model system (you would need perfect foresight to pick the right model per task). If Fusion beats this, it is definitively better.
- **Win rate CI**: Wilson score interval (more reliable than normal approx for small n)
