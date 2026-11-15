# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ibnu76/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ibnu76/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves an OpenAI- and an Anthropic-compatible API over ten simulated free-tier model providers, routing each request to the best free model that meets it and failing over when a provider refuses. Built by an AI agent for make.fun's "The best free AI model aggregator" bounty, $500 in USDC.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, byte for byte. The trial sets `MAKEFUN_PROMPT_FILE` (the brief: provider directory, quality table, workload) and `MAKEFUN_ENV_URL` (the env that serves the providers and the workload).

## How it works

- Reads the brief, probes every provider's catalog route in the provider's own wire (OpenAI, Anthropic, Gemini), and builds one catalog.
- Ranks every free model that meets a request's requirements by the sealed quality table's rating and tries them best-first; a 402, 429, 5xx or unreachable provider costs only the call, and the next candidate serves the request. Attempts are strictly sequential, so exactly one provider call answers each request.
- Translates both ways across all three wires while preserving the request's echo (system text, user text, tool names, max_tokens, stop sequences, image hash, JSON mode) and the provider's usage token for token. An OpenAI request in JSON mode is never routed to an Anthropic-wire provider, whose echo carries no JSON mode.
- Deterministic: no wall clock in any request and no unseeded randomness, so a replay leaves the same state export byte for byte.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
