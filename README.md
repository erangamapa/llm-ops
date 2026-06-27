# llm-ops

This is the runtime plane for the models in my sportsbook platform. model-foundry builds them. evalgate decides if they ship. This repo watches them live and feeds the failures back.

It does four jobs. Know exactly what is running. Roll changes out gradually. Notice decay fast. Turn incidents into tests.

## How the three repos fit

I build a model in model-foundry. I fine tune an adapter, freeze a golden set and push a card. I run it through evalgate against that golden set and get a pass or fail with a score. If it passes, I pin it here, shadow it, canary it and promote it. From then on this repo owns it. When a production answer is wrong, I promote the trace into a staged golden case and the next build has to get it right.

Nothing here calls a model API. Model calls are injected, and every test runs offline with no keys.

```
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/ruff check .
```

## Prompts and pins

A service never holds a raw prompt string. It pins a task and a version. Prompts live in `registry/prompts/<task>/<version>.md` with a front matter header holding task, version, owner and a one line changelog. `PromptRegistry.load(task, version)` returns that exact prompt. `latest(task)` compares versions as numbers, so v10 beats v9. Two files claiming the same version are refused, and so is registering over an existing one.

Models are pinned in `registry/pins.yaml`. Each task maps to a tier, provider, model, revision, adapter and the prompt version it runs. The loader refuses any task it does not know. `diff(old, new)` prints one line per changed field, so a deploy review reads like this.

```
tweet_classifier: revision 8f3c2a1 -> aa11bb2
tweet_classifier: prompt_version v3 -> v4
```

## Rollout ladder

Every candidate walks the same ladder. Shadow, then canary at a percent, then full.

- Shadow runs the candidate beside the live model, writes both outputs to a comparison record and never serves the candidate. If the candidate throws, the error is recorded and serving carries on.
- Canary starts at 1 to 99 percent and can only grow.
- Full is reachable only from canary. Shadow straight to full raises.
- A metric breach event rolls the rollout back from any active stage. Rollback is final. A new attempt is a new controller.

Transitions are logged with a reason, so the history explains itself later.

## Drift and cost

Drift watches two things. The first is the judge score. I keep a rolling window of daily mean scores and alert when the window mean falls more than a set delta below the baseline recorded at release. A series shorter than the window never alerts. The second is the input distribution. I compare the centroid of recent input vectors to the centroid saved at release using cosine distance. Numpy only.

Cost reads the trace export. Each model version has a price per thousand input and output tokens in `data/prices.yaml`. Math uses decimals so rollups are exact. There is a daily rollup per task and a paid versus open table that shows what each request cost on each side and what the migration saved.

## The feedback loop

`promote` takes a trace row and a corrected label from a human. It validates the label against the golden tweet schema, refuses a duplicate by input hash, appends to a staging jsonl and rewrites the manifest in the same shape model-foundry reads, a sha256 and a row count. I review the staging file, then merge it into the golden set and freeze it there. A bad label writes nothing.

## Fleet status

`render_status` builds one page, html and markdown, with a row for every pinned task. Model, revision, adapter, prompt version, rollout stage, last eval score and seven day cost. A task with no rollout in flight shows as full.

## Layout

```
llmops/registry    prompts and model pins
llmops/rollout     controller and shadow runner
llmops/monitors    drift and cost
llmops/feedback    incident promotion
llmops/reports     fleet status page
registry/          prompt files and pins.yaml
data/              trace fixture and price table
```

## Notes

The trace fixture is synthetic. The eval score input is a small json file keyed by task, written by the gate. I kept the format simple on purpose.

TODO. Wire the breach events to the alerting stream once the metrics service exists.
