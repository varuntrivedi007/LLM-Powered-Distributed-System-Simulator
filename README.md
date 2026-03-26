# Distributed System Simulation — Accuracy Upgrade

A SimPy-based distributed-systems simulator with hybrid rule-based and LLM-guided recovery.

This version adds three major accuracy upgrades:

1. **Quorum + leader election + replica lag**
2. **Per-request simulation with queues, retries, and timeouts**
3. **Correctness metrics** such as commit success, stale reads, timeout rate, quorum loss, and leader elections

---

## What changed

### 1. Consensus-aware cluster behavior
The simulator now tracks:
- current primary
- quorum availability across Primary/Replica voters
- automatic leader election after primary failure
- replica lag and catch-up after writes
- commit index progression

This makes partitions and crashes affect correctness, not just health/load.

### 2. Request-level communication model
Nodes now generate individual **read** and **write** requests.
Each request flows through:
- network dispatch
- packet loss / partition checks
- queue admission
- service time
- timeout watcher
- retry logic with exponential backoff

This replaces a purely aggregate delivery view with request-level outcomes.

### 3. Queueing and overload realism
Each node now has:
- a request queue
- queue-aware service delay
- queue pressure drops
- retry-induced pressure

So overload can increase latency and timeout risk more realistically.

### 4. Correctness metrics
`network.get_stats()` now reports:
- `commit_rate`
- `timeout_rate`
- `stale_read_rate`
- `quorum_unavailable_ticks`
- `leader_elections`
- `commit_index`

These are printed in `main.py` and summarized in `benchmark.py`.

---

## Files updated

- `network.py`
  - added request objects, queues, retries, timeouts
  - added quorum logic, primary election, replica catch-up
  - added correctness metrics
- `llm_agent.py`
  - richer cluster snapshot now includes consensus and correctness state
  - actions now clear queue state and respect consensus state better
- `main.py`
  - prints correctness metrics and queue/lag state
- `benchmark.py`
  - reports commit, timeout, stale-read, quorum-loss, and election stats

---

## Key metrics to report in your submission

Instead of only showing MTTR and message delivery, report:

- MTTR
- failures resolved
- message delivery
- **write commit success rate**
- **request timeout rate**
- **stale read rate**
- **quorum unavailable time**
- **leader elections**

That makes the project look much more like a true distributed-systems simulation.

---

## Setup

```bash
pip install simpy groq python-dotenv
cp .env
# Add your GROQ_API_KEY to .env
python main.py
```

## Benchmark

```bash
python benchmark.py 5,10 3
```

The benchmark summary now includes both resilience metrics and correctness metrics.

---

## Recommended claim for the project report

> We extended the simulator from a health/load fault model into a more faithful distributed-systems model by adding quorum-aware failover, request-level queueing and retries, and correctness metrics such as commit success and stale reads.

