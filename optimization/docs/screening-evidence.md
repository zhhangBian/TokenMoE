# Local exploratory evidence

Source: `/home/youwei/bzh/dataset/tokenmoe_artifacts/qwen3-runtime-20261019-063626`, primary N1/N4. Read-only collection checks pass for 40 sessions, 20 task identities, 684 admitted requests, 370,477 new-prefill positions, 321,370 decode positions, and one pre-engine rejection. Source hashes are checked again after analysis.

Historical-rule reproduction: `/home/youwei/bzh/dataset/tokenmoe_artifacts/qwen-residency-screening-20261022-v1-r3`. Every design-screening value rounds identically at budget 64, using stable ascending expert-ID sorting and the final K entries.

| Fixed-set lookup coverage | New prefill | Decode |
|---|---:|---:|
| Global | 86.4014% | 87.5611% |
| Previous | 85.6853% | 89.2687% |
| Previous + global | 87.7591% | 90.6597% |
| Previous + tool | 89.0288% | 91.1638% |
| Previous + stage | 89.7669% | 90.9441% |
| Request EWMA | 86.6804% | 91.7639% |
| Hindsight fixed set | 94.7493% | 94.4388% |

On subsequent requests, historical tool categories add about 0.0974 percentage points for prefill and 0.2275 for decode over history+global. Against history+stage, the increments are about -0.0648 and +0.2688 points. EWMA's full-decode coverage exceeds history+tool. Tool-switch and multiple-tool groups include negative increments. These are lookup counts, without measured cache waits or speedups.

Cold-start diagnostics remain in `qwen-residency-screening-20261022-v1` and `...-v1-r2`. The first used pure conditional priors at cold start; the second used pure global. The historical definition uses global as missing history, then mixes the configured conditional prior at weight 0.5. The implementation and regression test preserve this definition. Every result directory is retained.

Revised category output: `qwen-residency-causal-20261022-v2-r2`. Live receipt delay is unmeasured: consumer request creation supplies a conservative upper-bound receipt proxy. Historical labels remain unchanged. An N1 executable-word audit found 99 changed labels across 10 category pairs:

| Command pattern | Historical category | Revised categories and limit |
|---|---|---|
| `find .../pytest -name logging.py` | test | inspect; pytest occurs in a path |
| `cat > test.py <<'EOF' ...` | inspect or test | unknown; heredoc parsing is unsupported |
| `find . -name '*.py' \| xargs grep ...` | inspect | inspect, unknown; wrapper inference is conservative |
| `git diff -- file > patch.txt` | edit | inspect, edit |
| `echo '...def test...' > file` | test | unknown, edit; quoted text is not an executed test |
| `cd /testbed && make html` | build_or_install | build_or_install, unknown |

Real long-tool, multi-role, G2/G3, calibrated replay, live policy comparison and larger-model milestones remain outstanding. Per-fold tables and task bootstrap intervals are in the output artifacts.
