# NeMo Guardrails configs for `NemoRailsLayer`

One directory per guardrail; a layer selects ONE rail type (`rail:`) and optionally a subset of
its `flows:`. Verified with nemoguardrails 0.24.1 (Colang 1.0). Model endpoints are filled in by
the layer (`base_url` param or `$GUARDSTACK_GUARD_URL`); the `base_url` written here is only a
fallback. `engine: vllm` = NeMo's built-in OpenAI-compatible client (no langchain needed).

| dir | rail(s) | LLM needed | source of prompts/flows |
|---|---|---|---|
| `self_check/` | input (`self check input`), output (`self check output`) | yes (guard LLM as `main`) | NeMo `examples/bots/abc/prompts.yml` (bundled in the wheel) |
| `content_safety/` | input / output (`content safety check ... $model=content_safety`) | Llama-3.1-NemoGuard-8B content-safety (LoRA on Llama-3.1-8B-Instruct) | NeMo `examples/configs/content_safety/prompts.yml` (GitHub, develop) |
| `jailbreak_heuristics/` | input (`jailbreak detection heuristics`) | no (GPT-2-large perplexity, in-process) | NeMo library |
| `retrieval/` | retrieval (`regex check retrieval`) on tool outputs | no | NeMo regex library + AgentDojo injection patterns |

Example stack entries:

    nemo_in:  {type: nemo_rails, stage: input,       params: {rail: input,     config_path: configs/nemo/self_check}}
    nemo_out: {type: nemo_rails, stage: output,      params: {rail: output,    config_path: configs/nemo/self_check}}
    nemo_cs:  {type: nemo_rails, stage: output,      params: {rail: output,    config_path: configs/nemo/content_safety}}
    nemo_jb:  {type: nemo_rails, stage: input,       params: {rail: input,     config_path: configs/nemo/jailbreak_heuristics}}
    nemo_ret: {type: nemo_rails, stage: tool_output, params: {rail: retrieval, config_path: configs/nemo/retrieval}}
