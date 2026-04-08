# Smart Factory Scheduling Environment

An [OpenEnv](https://github.com/openenv/openenv)-compliant RL environment simulating real-world industrial scheduling: assign jobs to machines, handle breakdowns, and maximise throughput within deadlines.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `machines` | List[Machine] | id, status (idle/busy/broken), current_job, failure_rate |
| `pending_jobs` | List[Job] | id, remaining_time, deadline, priority (1-3), assigned_machine |
| `completed_jobs` | List[Job] | Jobs finished this episode |
| `time` | int | Current time step |
| `max_steps` | int | Episode length |
| `done` | bool | Episode terminated |
| `reward` | float | Reward from last action |

## Action Space

| Action | Effect |
|--------|--------|
| `assign_job <job_id> <machine_id>` | Assign pending job to idle machine |
| `repair <machine_id>` | Restore broken machine to idle |
| `wait` | Advance time with no change |

## Reward Function

| Event | Reward |
|-------|--------|
| Job completed on time | +1.00 + 0.20 × priority |
| Job completed late | +0.30 |
| Valid assignment | +0.10 |
| Invalid action | −0.10 |
| Idle machine (pending jobs exist) | −0.05 per machine |
| Job past deadline | −0.10 per step |
| Repair broken machine | +0.05 |

## Tasks

| Task | Machines | Jobs | Failure Rate | Max Steps | Baseline Score |
|------|----------|------|-------------|-----------|----------------|
| easy | 2 | 3 | 0% | 20 | 1.000 |
| medium | 4 | 7 | 8% | 30 | ~0.557 |
| hard | 6 | 12 | 15% | 40 | ~0.457 |

**Score formula:** `0.5 × completion_rate + 0.3 × on_time_rate + 0.2 × utilization_bonus`

## Setup

```bash
pip install -r requirements.txt
```

### Run HTTP Server (HF Space)
```bash
python server.py
# Routes: GET /health  POST /reset  POST /step  GET /state  GET /schema
```

### Run Inference (LLM agent)
```bash
export OPENAI_API_KEY=<your-key>
export FACTORY_TASK=easy   # easy | medium | hard
python inference.py
```

### Run RL Training
```bash
python train.py --task easy --episodes 10 --provider openai
python train.py --task medium --episodes 10 --provider claude
```

### Interactive Demo
```bash
python app.py   # opens at http://localhost:7860
```

### Docker
```bash
docker build -t factory-env .
docker run -e OPENAI_API_KEY=<key> -e FACTORY_TASK=easy -p 7860:7860 factory-env
```

## Baseline Scores

| Task | Score | Steps |
|------|-------|-------|
| easy | 1.000 | 4 |
| medium | ~0.529 | 12 |
| hard | ~0.533 | 34 |

## Project Structure

```
├── factory_env/
│   ├── env.py       # FactoryEnv (openenv.core.Environment)
│   ├── models.py    # FactoryAction, FactoryObservation, FactoryState
│   ├── tasks.py     # Task configurations
│   └── grader.py    # Score computation
├── inference.py     # LLM baseline agent
├── train.py         # Multi-episode RL training loop
├── server.py        # FastAPI HTTP server for HF Space
├── app.py           # Gradio interactive demo
├── openenv.yaml     # OpenEnv metadata
└── Dockerfile
```
