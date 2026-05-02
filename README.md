# Hybrid Recovery in Distributed Systems

A SimPy-based discrete-event simulator that combines a deterministic rule engine with an LLM agent (Groq Llama 3.3-70B) to build self-healing distributed clusters of 5, 10, or 20 nodes.

This README is written for a reviewer reproducing the project from a fresh clone. Follow the steps in order; each command is self-contained.

---

## 1. What this project does

- Simulates a distributed cluster (Primary, Replicas, Workers, Gateways) running consensus, heartbeats, and request traffic.
- Injects 5 failure types (Crash, Network Partition, Latency Spike, Memory Leak, CPU Overload) on a 7-phase escalating schedule.
- Routes recovery decisions through a 3-tier confidence system: rules handle simple cases, the LLM handles ambiguous ones.
- Scores every decision on Speed (35%) + Stability (25%) + Cascade prevention (25%) + Message delivery (15%).
- Persists decision outcomes to `decision_history.json` for cross-run learning.

Outputs: a per-run log file (`simulation_<timestamp>.txt`), benchmark JSON, and PNG charts.

---

## 2. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.9 or newer | Tested on 3.9.6 (macOS) |
| pip | latest | `python3 -m pip install --upgrade pip` |
| OS | macOS, Linux, or WSL | No Windows-specific paths used |
| Groq API key | Required | Free tier sufficient — https://console.groq.com |
| Disk | ~50 MB | Source + venv + logs |

No GPU, no Docker, no database needed.

---

## 3. Reproduction flow (end-to-end)

### Step 1 — Clone and enter the repo

```bash
git clone <repo-url> Distributed_Simulator
cd Distributed_Simulator
```

### Step 2 — Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Pinned versions are listed in [requirements.txt](requirements.txt):
`simpy`, `groq`, `python-dotenv`, `python-pptx`, `matplotlib`, `pillow`, `flask`, `pytest`.

### Step 4 — Create the `.env` file

The repo's `.gitignore` excludes `.env`, so it must be created locally. Paste the block below into a new file at the repo root:

```bash
cat > .env <<'EOF'
GROQ_API_KEY=your_groq_key_here
LLM_MODEL=llama-3.3-70b-versatile
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.3

SIMULATION_TIME=60
COOLDOWN_TICKS=8
CACHE_MAX_SIZE=200
THREAD_POOL_WORKERS=6
OPTIMIZE_EVERY=20
RUNS_PER_SIZE=5

ENABLE_PREDICTIVE_ACTIONS=false
ENABLE_OPTIMIZATION=false
ENABLE_FEEDBACK_FOLLOWUPS=true
EOF
```

Replace `your_groq_key_here` with a real key from the Groq console. Only `GROQ_API_KEY` is mandatory; everything else has defaults defined in [config.py](config.py).

### Step 5 — (Optional) Reset learned state for a clean baseline

Cross-run learning persists past LLM decisions. To reproduce a "first run" exactly, delete the cache:

```bash
rm -f decision_history.json
rm -rf __pycache__
```

Skip this step if you want to demonstrate cumulative learning across runs.

### Step 6 — Run the unit tests (sanity check)

```bash
python -m pytest test_failures.py test_llm.py test_network.py -v
```

All tests should pass before moving on. The LLM test uses a stubbed client and does not consume API quota.

### Step 7 — Run a single simulation

```bash
python main.py 5            # 5-node cluster (default)
python main.py 10           # 10-node cluster
python main.py 20           # 20-node cluster
```

What happens:
- Simulator builds the cluster, starts SimPy event loop.
- Failures from [cluster_config.py](cluster_config.py) fire on schedule.
- Rule engine + LLM produce recovery decisions; scorer rates each one.
- Terminal prints compact one-liner per LLM decision.
- Full structured log written to `simulation_<YYYYMMDD_HHMMSS>.txt`.

Expected runtime: ~60-120 seconds at 5 nodes, longer at 20.

### Step 8 — (Optional) Run the benchmark sweep

Multi-size, multi-seed run for statistical comparison:

```bash
python benchmark.py 5,10 3            # sizes 5 and 10, 3 seeds each
python benchmark.py 5,10,20 5         # full sweep
```

Aggregated results land in `benchmark_results.json`.

### Step 9 — (Optional) Compare LLM against rule-only

```bash
python compare.py 5                   # cluster size 5, default seed
python compare.py 10 12345            # size 10, explicit seed
```

Runs the same scenario twice (once with LLM, once rules-only) and prints a side-by-side score breakdown.

### Step 10 — (Optional) Generate charts

```bash
python generated_charts.py            # 5/10-node charts → charts/
python generated_charts_20nodes.py    # 20-node charts   → charts_20/
```

PNGs are written into the corresponding folders.

### Step 11 — (Optional) Launch the web dashboard

```bash
python app.py
# open http://127.0.0.1:5000 in a browser
```

The dashboard runs simulations and renders metrics live via the Flask UI in [templates/dashboard.html](templates/dashboard.html).

---

## 4. Determinism and reproducibility

Two sources of randomness exist:

1. **Failure schedule** — fully deterministic; failure times are hardcoded in [cluster_config.py](cluster_config.py). The only stochastic injector, `schedule_random_failures`, runs on size ≥ 20 and is seedable via the `seed` argument to [benchmark.py](benchmark.py) / [compare.py](compare.py).
2. **LLM responses** — non-deterministic even at `temperature=0.3`. Two runs with identical inputs may diverge at decision boundaries. For exact replay, pre-populate `decision_history.json` and lower `LLM_TEMPERATURE` toward 0.

Same seed + cleared `decision_history.json` + same cluster size ⇒ identical failure schedule. LLM-driven branches will still vary slightly between runs.

---

## 5. Architecture

```
Cluster Layer        5/10/20 nodes (Primary, Replica, Worker, Gateway)
       |             Consensus, heartbeats, request simulation
       v
Failure Injection    Crash, Partition, Latency Spike, Memory Leak, CPU Overload
       |             7 escalating phases — warmup to multi-failure chaos
       v
Decision Engine      Rule confidence scoring + 3-tier routing
       |             LLM chain-of-thought + multi-turn verification
       v
Recovery Layer       Action execution, feedback at t+5, load shedding
       |             Predictive detection, cross-run learning
       v
Scoring System       Speed (35%) + Stability (25%) + Cascade (25%) + Delivery (15%)
```

### Tiered routing

| Tier | Confidence | Handler |
|------|------------|---------|
| 1 | ≥ 75% | Rules execute directly |
| 2 | 45–74% | LLM decides |
| 3 | < 45% | LLM emergency reasoning with full snapshot |

### Failure types

| Type | Mechanism | Duration |
|------|-----------|----------|
| Crash | Node offline, health → 0%, recovers at 40% | 3–8 ticks |
| Network Partition | Bidirectional link cut between two nodes | 3–7 ticks |
| Latency Spike | Response time × 2–5, triggers retries | 2–5 ticks |
| Memory Leak | Health drains 3–8%/tick, alert at 50% | Until intervention |
| CPU Overload | Load spike 30–60%, slows processing | 3–7 ticks |

---

## 6. File map

```
main.py                       Simulation loop — events, failures, scoring
network.py                    Cluster model — nodes, consensus, requests
failures.py                   Fault injector — 5 failure types
rule_based.py                 Rule engine — confidence + tiered routing
llm_agent.py                  LLM agent — Groq calls, CoT, load shedding
scorer.py                     4-component decision scorer
predictor.py                  Linear-regression health prediction
feedback.py                   t+5 success check + follow-up actions
decision_memory.py            Persistent cross-run learning
event_bus.py                  Thread-safe event queue
config.py                     Env-var loader
cluster_config.py             Cluster + failure-schedule generator
logger.py                     Structured log writer
app.py                        Flask web dashboard
benchmark.py                  Multi-size / multi-seed runner
compare.py                    LLM vs rule-only comparison
generated_charts.py           Charts for 5/10-node runs
generated_charts_20nodes.py   Charts for 20-node runs
templates/dashboard.html      Web UI template
test_failures.py              pytest — fault injection
test_llm.py                   pytest — LLM agent (stubbed)
test_network.py               pytest — cluster behavior
requirements.txt              Pinned dependencies
.env                          Local secrets (gitignored)
```

---

## 7. Configuration reference

All environment variables are read in [config.py](config.py).

| Variable | Default | Purpose |
|----------|---------|---------|
| `GROQ_API_KEY` | *(required)* | Groq cloud API auth |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `LLM_MAX_TOKENS` | `500` | Per-call token cap |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature |
| `SIMULATION_TIME` | `60` | Sim ticks per run |
| `COOLDOWN_TICKS` | `8` | Min ticks between repeat actions |
| `CACHE_MAX_SIZE` | `200` | LLM response cache size |
| `THREAD_POOL_WORKERS` | `4` | Concurrent LLM workers (auto-scales by cluster size) |
| `OPTIMIZE_EVERY` | `20` | Optimization-pass interval |
| `RUNS_PER_SIZE` | `5` | Benchmark seeds per cluster size |
| `ENABLE_PREDICTIVE_ACTIONS` | `false` | Proactive intervention on health forecast |
| `ENABLE_OPTIMIZATION` | `false` | Periodic cluster optimization |
| `ENABLE_FEEDBACK_FOLLOWUPS` | `true` | t+5 retry on failed actions |

---

## 8. Metrics produced

- MTTR (Mean Time to Recovery)
- Failure resolution rate
- Message delivery rate
- Write-commit success rate
- Request timeout rate
- Stale read rate
- Quorum-unavailable time
- Leader elections / split-brain events
- LLM vs Rule per-decision score

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Missing GROQ_API_KEY` | `.env` not loaded | Verify file at repo root, no quotes around value |
| `groq.AuthenticationError` | Invalid / expired key | Regenerate on Groq console |
| `ModuleNotFoundError: simpy` | venv not active or deps missing | `source .venv/bin/activate && pip install -r requirements.txt` |
| 429 rate limits | Groq free-tier cap (~30 req/min) | Lower `THREAD_POOL_WORKERS` or `RUNS_PER_SIZE` |
| Different scores across runs | LLM non-determinism | Expected — see Section 4 |
| Stale learning bias | `decision_history.json` accumulated | Delete file for clean baseline |
| `python` not found | macOS / fresh Linux | Use `python3` and `python3 -m pip` |

---

## 10. Expected results

Across runs, the hybrid system shows:

- **LLM outperforms rules by 12–23 points** on multi-failure scenarios.
- **Memory-leak MTTR drops from ~17 → ~5 ticks** with early intervention.
- **100% commit rate, 0% timeout rate** in stabilized runs.
- **Cross-run learning converges** — LLM stops repeating low-scoring actions.
- **Rules remain optimal for simple single-failure cases** (comparable scores at zero API cost).

Sample logs: `Simulations_Output/` and `simulation_<timestamp>.txt` files in repo root.
