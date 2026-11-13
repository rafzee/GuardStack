# Guardrail adapter notes

Adapters: `guardstack/layers/{llamafirewall_layers,nemo_layers,pii_layers,safety_classifiers}.py`
(+ shared helpers in `_adapter_common.py`). Tests: `tests/test_adapters.py` (no GPU, no gated
weights). Pins: `requirements/adapters.txt`. Written 2026-11-03.

How things were verified: the libraries were installed together in one env (Python 3.13) and
their source / signatures read; NeMo, Presidio and CodeShield were run for real in the tests;
LLM guards were run against a local fake OpenAI-compatible server; vLLM request fields were
checked in the vllm 0.33.0 wheel (`entrypoints/openai/{chat_completion,completion}/protocol.py`).
**Hugging Face was not reachable from the dev sandbox** (proxy 403; GitHub raw and PyPI were), so
no real model weights were loaded and no chat template was inspected. Model-card facts below were
read through the HF web pages or the vendors' GitHub repos; anything not seen first-hand is
marked **UNVERIFIED**.

## Common conventions

* Heavy imports happen in `warmup()` / `_load()`; constructing a layer imports nothing heavy
  (tested by blocking torch/transformers/llamafirewall/nemoguardrails/presidio/gliner/spacy).
  `check()` calls `warmup()` lazily, so `layer.check(CheckContext(stage=TOOL_OUTPUT, text=...))`
  works on a bare payload (the adaptive attack's local stage).
* `assert_allowed()` is called on every model id in `__init__` (for NeMo: on every model in the
  rails config, in `warmup()`).
* No adapter catches the guard's own exceptions. Unparsable model output raises `ValueError`;
  an unreachable endpoint raises; `GuardLayer.run` logs `layer_error` and fails open.
* LLM guards use `guardstack.llm.openai_client(base_url, role=<layer name>)` so every call is also
  an `LLMEvent`; `tokens_in/out` = the response's `usage`. Endpoint: `base_url` param, else
  `$GUARDSTACK_GUARD_URL`, else `http://127.0.0.1:8001/v1`. `${VAR}` in `model`/`base_url` is
  expanded by the layer; unresolved placeholders fall back to defaults. `warmup()` checks that
  the endpoint serves the model (`check_served=True`).
* Stage -> what an LLM guard sees: input/tool_output -> one user turn with the text;
  output -> [user goal, assistant answer]; action -> [user goal, assistant `name({json args})`].
  In-process text classifiers at the action stage read the string-valued arguments.
* `score` is higher = more malicious, in [0,1]. Probability-based where the guard exposes one,
  otherwise 0/1. `threshold` (where meaningful) switches the decision to `score >= threshold`.

## Environments / co-installation

`llamafirewall==1.0.3` and `nemoguardrails==0.24.1` **do co-install** in one env (plus presidio,
gliner, torch 2.15, transformers 5.22.1), but the resolver has to:

* downgrade **semgrep 1.183.0 -> 1.79.0** (newer semgrep pins `jsonschema~=4.25`, NeMo needs
  `>=4.26`). CodeShield's semgrep rules ran fine on 1.79.0 (tests), but detection can differ
  between semgrep versions, so keep the pin fixed for the whole paper.
* downgrade `huggingface_hub` 2.4 -> 1.33 (transformers 5.22.1).

Recommendation: **one env** with the pinned set (a `shadow_all` stack needs every layer in one
process). Only if CodeShield must use the newest semgrep, give LlamaFirewall its own env and run
CodeShield layers in a separate (replayed) pass. vLLM stays in its own env.
The PyPI torch 2.15.0 wheel is a CUDA 13.2 build: check the cluster's driver; if it is older, install
torch from a cu12x index. Weights must be in `HF_HOME` beforehand; run with `HF_HUB_OFFLINE=1`
(the in-process loaders then pass `local_files_only=True`).

## LlamaFirewall (llamafirewall 1.0.3, MIT, github.com/meta-llama/PurpleLlama/LlamaFirewall)

Findings from the installed source: `ScanResult(decision: ScanDecision{ALLOW,
HUMAN_IN_THE_LOOP_REQUIRED, BLOCK}, reason: str, score: float, status: ScanStatus)`;
`Scanner.scan(message, past_trace)` is `async`. The top-level `LlamaFirewall.scan` creates a new
scanner per call and calls `asyncio.run` — we never use it.

### `PromptGuard2Layer` (`promptguard2`, family `pi_classifier`, stages tool_output [default] / input)
* `llamafirewall.scanners.PromptGuardScanner` is **not usable**: `promptguard_utils.py` imports
  `huggingface_hub.HfFolder` (removed in hub>=1.0 -> ImportError with this stack), hard-codes the
  86M id, loads from `$HF_HOME/meta-llama--Llama-Prompt-Guard-2-86M` (not the hub cache) and
  otherwise calls interactive `login()`.
* We port `PromptGuard.get_jailbreak_score` exactly: `_preprocess_text_for_promptguard` copied
  verbatim (a test runs the installed upstream function body against ours), tokenizer with
  `truncation=True, max_length=512`, `softmax(logits / T)[0, -1]` (T=1). Default threshold 0.9 =
  `PromptGuardScanner` default. `tokens_in` = classifier tokens.
* `segment=True` (not LlamaFirewall behaviour; model card's advice): max over 512-token windows.
  Default off = LlamaFirewall fidelity (only the first 512 tokens are seen — long AgentDojo tool
  outputs can hide an injection past that point).
* Model: meta-llama/Llama-Prompt-Guard-2-86M (mDeBERTa base) or -22M (DeBERTa-xsmall); licence
  **Llama 4 Community License, gated** (HF card). Labels benign/malicious (card); the `id2label`
  index is UNVERIFIED (gated config) — `warmup()` raises unless class `malicious_index` (default
  -1, LlamaFirewall's choice) is named malicious/injection/LABEL_1.

### `AlignmentCheckLayer` (`alignment_check`, family `llm_judge`, stage trace)
* Upstream cannot be redirected cleanly: `AlignmentCheckScanner.__init__(scanner_name)` only;
  `CustomCheckScanner` hard-codes `https://api.together.xyz/v1`,
  `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `TOGETHER_API_KEY`, uses its own client's
  `beta.chat.completions.parse` (bypasses usage tracking) and maps **any** LLM error to
  `conclusion=True` (fail closed, error hidden).
* Re-implemented with `SYSTEM_PROMPT` and `USER_PROMPT_FORMAT` copied verbatim from
  `llamafirewall/scanners/experimental/alignmentcheck_scanner.py` (equality is tested) and the
  `AlignmentCheckOutputSchema` sent as `response_format={"type":"json_schema", strict}`.
* Trace = upstream `"\n".join(str(Message))` -> lines `Role.USER: ...`, `Role.ASSISTANT: {"thought",
  "action","action_input"}` (the encoding of PurpleLlama's `examples/demo_alignmentcheck.py`);
  one line per tool call; the selected action (the call being checked) is last; system messages
  and tool outputs are excluded (`include_tool_outputs=True` adds `Role.TOOL:` lines).
* block = `conclusion` (upstream: HUMAN_IN_THE_LOOP_REQUIRED). score = P(true) from the
  top-5 logprobs of the token after `"conclusion":` (our addition, for threshold sweeps), else
  0/1. No user message -> allow, score 0 (upstream: ALLOW/ERROR). Errors raise (differs from
  upstream's fail-closed; deliberate, see above).
* Judge model: any allowed model on vLLM (default meta-llama/Llama-3.3-70B-Instruct, Llama 3.3
  licence, gated; `$GUARDSTACK_ALIGNMENT_MODEL`). Results are **not** comparable with
  LlamaFirewall's published numbers (Maverick on Together). UNVERIFIED: that vLLM's structured
  output plus logprobs returns the `true`/`false` token as one token for Llama tokenizers (the
  code falls back to 0/1 if not).

### `CodeShieldLayer` (`codeshield`, family `code_scanner`, stages output [default] / action / tool_output)
* Uses `llamafirewall.scanners.code_shield_scanner.CodeShieldScanner` directly (regex +
  semgrep over all supported languages). Binary: any issue -> BLOCK, score 1.0.
* Action stage scans string-valued args (`arg_names` to narrow); `code_blocks_only=True` scans
  only ``` fences.
* codeshield logs a failing semgrep subprocess with `LOG.fatal` and **continues with no issues
  (silent fail-open)**; `warmup()` therefore runs an `os.system(cmd)` canary that only semgrep
  flags and raises if it is not detected. codeshield creates an `osemgrep` symlink inside the
  semgrep package dir at import (env must be writable).

## NeMo Guardrails (`nemo_rails` -> `NemoRailsLayer`; nemoguardrails 0.24.1, Apache-2.0)

One layer = one rail type at one stage, `params: {rail, config_path, flows?, base_url?}`.
Runs on `LLMRails` (Colang 1.0); each layer owns a private asyncio loop thread.

| rail | stage(s) | mechanism (verified at runtime) |
|---|---|---|
| input | input, tool_output | `generate_async(messages, options={"rails": {input: flows or True, rest False}, "log": {activated_rails, llm_calls}})` — what `LLMRails.check()` does, plus the log for tokens/decisions |
| output | output, action | same, messages `[user goal, assistant text]` |
| tool_input (NeMo: on tool **results**) | tool_output | same, messages `[user, assistant(tool_calls), tool]` |
| tool_output (NeMo: on tool **calls**) | action | same, messages `[user, assistant(tool_calls)]` |
| execution | action / tool_output | alias for tool_output / tool_input by stage |
| retrieval | tool_output | `generate_events_async([ContextUpdate(relevant_chunks=text), UserMessage, BotIntent])` with the `generate_bot_message` action replaced by a stub |
| dialog | — | **not supported** (raises): dialog rails need the main LLM for user-intent / next-step generation |

* block = an activated rail of that type has `stop=True` (`check()`'s rule). Retrieval library
  rails filter instead of stopping (regex empties `$relevant_chunks`), so for retrieval
  `modified_as_block=True` by default; otherwise a modification is offered via `transform()`.
  score = 0/1. tokens = sum of `log.llm_calls` usage (0 on the retrieval path — NeMo exposes no
  log there).
* Main-LLM guard: if `log.llm_calls` contains a generation task (general, generate_user_intent,
  generate_next_steps, generate_bot_message, ...) the layer raises. With dialog off, NeMo's flows
  (`rails/llm/llm_flows.co`) echo the content instead of generating.
* Endpoints: `engine: vllm` + `parameters.base_url` uses NeMo's built-in OpenAI-compatible
  client (langchain not needed). An explicit layer `base_url` overrides every model's base_url;
  `$GUARDSTACK_GUARD_URL` fills models without one. Every model id is `assert_allowed`.
* When there is no user goal (bare payload), tool/retrieval rails get a placeholder user turn:
  NeMo silently skips tool rails if the last user message is empty.
* LLMRails init did not download anything (no FastEmbed index is built for these configs).

Configs (`configs/nemo/*/`, all initialise in the tests):

* `self_check/` — `self check input` + `self check output`; prompts verbatim from the wheel's
  `examples/bots/abc/prompts.yml`; judge = `main` model (default Llama-3.1-8B-Instruct). Parser:
  NeMo's default (answer containing "yes" -> blocked). Prompt text is a "company bot" policy —
  edit once, then freeze.
* `content_safety/` — Llama-3.1-NemoGuard-8B content safety (`$model=content_safety`), prompts
  verbatim from NVIDIA/NeMo-Guardrails `examples/configs/content_safety/prompts.yml` (develop,
  fetched 2026-11-03) with parsers `nemoguard_parse_prompt_safety/response_safety` (exist in
  0.24.1). Model: LoRA (r=16, targets incl. `lm_head`) on Llama-3.1-8B-Instruct; NVIDIA Open
  Model License + Llama 3.1 Community License (HF card); gating UNVERIFIED. Serving as a vLLM
  LoRA module is UNVERIFIED (an `lm_head` LoRA may be rejected -> merge offline).
* `jailbreak_heuristics/` — `jailbreak detection heuristics` (GPT-2-large perplexity, in-process;
  library defaults 89.79 / 1845.65). Not executed here (needs `gpt2-large` from HF): UNVERIFIED
  end to end. The library loads the model at import of `heuristics/checks.py`, and if a
  `server_endpoint` is configured but fails it **allows** the input (fail-open inside NeMo).
  GPT-2 licence (MIT) UNVERIFIED.
* `retrieval/` — `regex check retrieval` with guardstack's `DEFAULT_INJECTION_PATTERNS` (tested
  equal), i.e. a NeMo-machinery twin of `regex@tool_output` for the redundancy analysis.

## PII

### `PresidioPIILayer` (`presidio_pii`; presidio-analyzer/anonymizer 2.2.366, MIT; spaCy 3.8.18)
* Stages action [default] (string args; `tools`, `arg_names` filters), output, tool_output,
  input. `entities` (None = all), `threshold` 0.5 on the recognizer score; `analyze(...,
  score_threshold=0)` so score = max entity score. `ignore_user_supplied` drops entities that
  appear verbatim in a user turn. `anonymize=True` -> transform with Presidio's `<ENTITY>`
  replacement, never blocks (not allowed at action stage: the pipeline does not transform calls).
* `spacy_model` default `en_core_web_lg` (must be installed; warmup refuses Presidio's runtime
  `spacy download`); `blank:en` = pattern recognizers only (no PERSON/LOCATION) — used in tests.
* Presidio's EmailRecognizer calls `tldextract`, which fetches the Public Suffix List from the
  network on first use (bundled fallback after failure). `offline_psl=True` (default) replaces
  tldextract's process-global extractor with a snapshot-only one: no network, deterministic.
* Observed raw behaviour: emails also yield URL hits on their domain parts (score 0.5); phone
  numbers score 0.4 (< default threshold).

### `GlinerPIILayer` (`gliner_pii`; gliner 0.2.30, Apache-2.0)
* `GLiNER.from_pretrained(model, local_files_only=...)`, `predict_entities(text, labels,
  flat_ner, threshold)`; predictions at min(threshold, 0.3) so score also reflects near misses.
* Default model `urchade/gliner_multi_pii-v1`: Apache-2.0, not gated, individual author
  (urchade), fine-tune of urchade/gliner_multi-v2.1 (HF card). Alternative `nvidia/gliner-PII`:
  NVIDIA Open Model License, built on urchade/gliner_large-v2.1, card evaluates at threshold 0.3.
  Neither is from a banned entity. `knowledgator/gliner-pii`: licence UNVERIFIED (not checked).
  The underlying encoder's tokenizer (mDeBERTa for multi models) must also be in HF_HOME
  (UNVERIFIED which files GLiNER fetches offline). Real model not run here (HF blocked).

## Safety classifiers (`safety_classifiers.py`)

### `LlamaGuardLayer` (`llamaguard`; any stage, default input)
* vLLM chat endpoint; the model's own chat template builds the prompt (S1–S14 MLCommons
  categories). Output `safe` | `unsafe\nS<k>[,S<j>]` (LG4 card example `unsafe\nS9`); anything
  else raises.
* score = P(first token = "unsafe") from `top_logprobs=5` ("probability for the first token ...
  as the 'unsafe' class probability", LG3 card). block = generated label (or score >= threshold).
* `categories`: post-filter (an unsafe verdict blocks only if a listed code is present). For
  Llama Guard 4 the other codes are also removed from the prompt via
  `chat_template_kwargs={"excluded_category_keys": [...]}` (argument shown in the transformers
  v4.51.3-LlamaGuard-preview release notes; vLLM forwards `chat_template_kwargs`).
* Licences: LG4 12B = Llama 4 Community License, gated (HF card); LG3 8B = Llama 3.1 Community
  License, gated (UNVERIFIED, not fetched). UNVERIFIED: LG3's HF chat template has no category
  argument (hence post-filter only); vLLM's content-format auto-detection for LG4's list-of-parts
  template (we send plain string content); that "unsafe" is a single first token.

### `ShieldGemmaLayer` (`shieldgemma`)
* Prompt-only / prompt-response templates and the four harm policies copied verbatim from
  Google's notebook `google/generative-ai-docs .../safeguards/shieldgemma_on_huggingface.ipynb`
  (Apache-2.0; parity tested against the notebook's `make_prompt`). One call per harm type;
  P(violation) = softmax(logit[Yes], logit[No])[0] at the last position; score = max over
  `harm_types`, threshold 0.5.
* `backend="transformers"` (default; exactly the notebook path, BOS added) or `"vllm"`
  (`/v1/completions`, `max_tokens=1, logprobs=20`, vLLM adds BOS by default; a token missing from
  the top-20 counts as 0). Licence: Gemma terms, gated (HF card).

### `WildGuardLayer` (`wildguard`)
* Prompt = `WILDGUARD_INPUT_FORMAT` from the wildguard package (`wildguard/utils.py`, Apache-2.0;
  verbatim, tested), `/v1/completions`, greedy, `max_tokens=32`; output parsed with the
  package's regex. input/tool_output -> "Harmful request" (response empty, as the package does);
  output/action -> "Harmful response". score = P(yes)/(P(yes)+P(no)) at the answer token
  (logprobs=5), else 0/1. Model: Apache-2.0, gated (AI2 form, auto-approval), Mistral-7B-v0.3
  base. UNVERIFIED: tokenisation of " yes"/" no" (fallback 0/1).

### `GraniteGuardianLayer` (`granite_guardian`)
* vLLM chat with `chat_template_kwargs={"guardian_config": {...}}` (+ `think` for 3.3).
  3.0–3.2: `{"risk_name": risk}` (+`risk_definition`), output starts with Yes/No (3.2 adds
  `<confidence>`); 3.3: `{"criteria_id": risk}` (+`custom_criteria`), output `<score> yes|no
  </score>` (from IBM's 3.1/3.2/3.3 cookbooks on GitHub ibm-granite/granite-guardian).
* score = IBM's 3.1 `get_probabilities` (sum of exp(logprob) of top-20 yes/no alternatives,
  normalised); 3.3: only positions after `<score>`. Licence Apache-2.0 (IBM README); gating
  UNVERIFIED (believed open). UNVERIFIED: exact 3.3 no-think output string (inferred from the
  3.3 cookbook parser and the 4.1 README).

### `GptOssSafeguardLayer` (`gpt_oss_safeguard`, family `llm_judge`)
* System message = policy, user message = content (OpenAI/ROOST user guide, openai-cookbook
  `articles/gpt-oss-safeguard-guide.md`); harmony is applied by vLLM's chat endpoint.
  `reasoning_effort` (default "low") is sent as the request field (exists in vLLM 0.33; its
  mapping into the harmony system header is UNVERIFIED at runtime).
* `DEFAULT_SAFEGUARD_POLICY` is **ours** (a prompt-injection policy following the guide's
  INSTRUCTIONS/DEFINITIONS/VIOLATES/SAFE/EXAMPLES layout, binary 0/1 output) — not an official
  policy; pass `policy` / `policy_path` to change. Output parsed as 0/1 or JSON `violation`;
  score binary (reasoning not scored). Licence Apache-2.0, no gating mentioned (HF card).

### `HFPromptInjectionClassifier` (`hf_pi_classifier`)
* Any `AutoModelForSequenceClassification`; score = softmax[malicious_label] (sigmoid for a
  single-logit head, as the text-classification pipeline does); truncation at 512 (card
  example) or `segment=True`. warmup fails if `malicious_label` is not a model label.
* Default protectai/deberta-v3-base-prompt-injection-v2: Apache-2.0, not gated, labels
  `{0: SAFE, 1: INJECTION}` (config.json), deberta-v3-base; card: not for jailbreaks, English
  only, false positives on system prompts; project archived. ~738 MB safetensors (> the 500 MB
  budget, and HF was unreachable), so the `@pytest.mark.slow` real-model test is opt-in
  (`GUARDSTACK_RUN_SLOW=1`) and never ran here.

## Interface changes requested (not made; files owned elsewhere)

1. `guardstack/llm.py`: `UsageClient` wraps only `chat.completions`. Raw-prompt guards
   (WildGuard, ShieldGemma-vLLM) need `completions.create`; the adapters currently wrap it with
   llm.py's private `_UsageCompletions` (`_adapter_common.completions_api`). Please expose a
   tracked `completions` on `UsageClient` (the helper picks it up automatically).
2. `guardstack/stacks.py`: `load_stacks` does not expand `${VAR}` (configs/stacks.yaml uses
   them); adapters expand `model`/`base_url` themselves, but central expansion would be cleaner.
3. Optional: a `GuardLayer.close()` hook (NeMo layers own a daemon event-loop thread).
4. `pytest.mark.slow` is unregistered (no pytest config in the repo) -> one
   `PytestUnknownMarkWarning`; register it in a `pytest.ini`/`pyproject.toml` if you add one.


## Update 2026-11-07 (NeMo wiring for the real jobs)

* `NemoRailsLayer(models={type: model_id})` overrides the model ids in a rails config (`${VAR}`
  expanded). `nemo_in` uses it to run NeMo's self-check prompts on the job's gpt-oss-20b
  (`GUARD_URL`), so it no longer points at port 8001 (Llama Guard in the jobs).
* An empty judge completion now raises (guard error) instead of passing: NeMo's parsers read an
  empty answer as "safe", which would silently fail open if a reasoning model ran out of tokens.
* NemoGuard is served from merged weights (`scripts/merge_lora.py`, checked bitwise against
  peft 0.21.2 `merge_and_unload` on a tiny Llama, incl. rsLoRA and a LoRA'd `lm_head` whose base
  weight peft saves as `lm_head.base_layer.weight`). This removes the UNVERIFIED vLLM-LoRA-on-
  `lm_head` dependency. NeMo 0.24.1's default framework talks to vLLM through chat completions
  (`llm/models/openai_chat.py`), so the server applies the Llama 3.1 chat template.
