# guardstack: evaluation dataset matrix

Status: vetted on 2026-11-03 by the dataset lead. Machine-readable plan: `configs/datasets.yaml`. Downloader: `data/download_datasets.py`.

**How this was verified.** GitHub sources were checked from the repositories themselves: HEAD SHAs with `git ls-remote`, then README, LICENSE and data files read at those SHAs and the data counted. Hugging Face (HF) sources were checked from their dataset cards and commit or tree pages through a web reader. The sandbox proxy blocks `huggingface.co` for direct downloads, `api.github.com` and arXiv, so for some HF facts there was no second source. Anything that could not be confirmed is marked **UNVERIFIED**. All counts below come from the data files unless the row says "card" or "paper".

---

## 0. Key findings

1. **AgentDojo has no "webbase" suite.** The one-pager is wrong (§1). Most likely source of the error: the repo's AI-oriented `CLAUDE.md` lists a non-existent "webbase" suite.
2. **the cluster blocks four official pipelines as shipped:**
   - Agent-SafetyBench's official scorer (ShieldAgent) is Qwen-based.
   - MT-JailBench defaults to a Qwen2.5-32B attacker, and so does X-Teaming's example config.
   - ActorAttack's code defaults to deepseek-chat.
   All four need a substitute model. Two candidate *datasets* are gray-area because their content was generated with covered-entity models: SafeMTData (DeepSeek) and XGuard-Train (Qwen attacker, DeepSeek-V3 targets). They are flagged, not excluded. The downloader skips them unless given `--allow-gray-area`.
3. **CyberSecEval cannot be the oracle for CodeShield.** CodeShield is built on the same Insecure Code Detector (ICD) that scores CyberSecEval, so the measurement is circular. Insecure-code ASR for CodeShield has to come from execution-based oracles (SecCodePLT, CWEval).
4. **Watch for vendor tuning on AgentDojo.** The Llama Prompt Guard 2 model card reports agentic evaluation on AgentDojo, and LlamaFirewall evaluates there too. The guard developers may have tuned toward this distribution, so off-distribution IPI sets (InjecAgent, LLMail-Inject payloads, BIPIA) are required for a fair per-layer claim.

---

## 1. AgentDojo facts (task 1)

| Question | Finding | Evidence |
|---|---|---|
| Number of suites | **4**: workspace, travel, banking, slack. No "webbase"/web suite exists anywhere upstream. | `src/agentdojo/task_suite/load_suites.py` on `main` (HEAD `089ed468`, 2026-06-02) registers benchmark versions v1, v1.1, v1.1.1, v1.1.2, v1.2, v1.2.1, v1.2.2 and nothing newer. Every version holds only those four suites. Searched all 45 remote branches and all tags (v0.1.2 to v0.1.35): no suite directory other than the four (plus `tools/`). |
| Where "webbase" comes from | The only occurrence in the whole repo is one bullet in `CLAUDE.md` (commit `96046ab`, 2025-08-09). It describes a "webbase: web-based interaction tasks" suite that `default_suites/` does not contain. | `git grep webbase` across the repo history |
| What "web" exists | `default_suites/v1/tools/web.py` (`get_webpage`, `post_webpage`) is a *tool* used only by the **slack** suite. | `grep tools.web` |
| Newer versions | PyPI 0.1.35 = git tag v0.1.35 (`a75aba76`, 2025-10-27) = benchmark **v1.2.2**. `main` is 5 commits ahead with no task changes (AST utils, model list, prompting LLM, and an OpenAI-compatible provider). No v1.3 or v2 exists. | `git log v0.1.35..origin/main`, `docs/changelog.md` |
| Releases page / forks | Could not be enumerated: the GitHub web UI and API are blocked from this sandbox. **UNVERIFIED** whether some fork adds a web suite. | n/a |
| Ports with a 5th suite | The UK AISI `inspect_evals` port lists **workspace_plus** as a 5th suite: 42 user / 15 injection tasks, adding 2 terminal tasks and 1 injection task, and it needs Docker. That port pins task version 1.2.1, lists slack with 20 user tasks, and supports only `important_instructions`. This is not "webbase" either. | inspect_evals AgentDojo README (`UKGovernmentBEIS/inspect_evals@525aed6`) |
| Task counts (v1.2.2) | workspace 40/14, travel 20/7, banking 16/9, slack 21/5 (user/injection): **97 user, 35 injection, 949 (user × injection) security cases**. v1 had 6 workspace injection tasks, giving 629 cases. | `get_suites()` on the installed package |
| Licence | MIT | `LICENSE` |

**Built-in attacks** (17 registered in 0.1.35):
- **Template attacks:** `important_instructions` plus 5 name variants (`_no_user_name`, `_no_model_name`, `_no_names`, `_wrong_model_name`, `_wrong_user_name`); `tool_knowledge`, which is important_instructions plus tool-call hints; `direct`; `ignore_previous`; `system_message`; `injecagent`, the InjecAgent "IMPORTANT!!! Ignore all previous instructions…" string; and `manual`, where a human types the injection.
- **DoS family:** `dos`, `swearwords_dos`, `captcha_dos`, `offensive_email_dos`, `felony_dos`. These run only one injection task per user task, and security is defined as *not utility* (the attack succeeds if the agent fails the user task).
- **Nothing adaptive or optimised ships.** Adaptive IPI attackers have to be built by us.

**Built-in defences** (for baselines only): `tool_filter`, `transformers_pi_detector` (protectai/deberta-v3-base-prompt-injection-v2), `spotlighting_with_delimiting`, `repeat_user_prompt`.

**How security is judged:**
- Every injection task implements `security(model_output, pre_environment, post_environment) -> bool`. True means the attacker goal was achieved. The check diffs the environment state before and after the run and/or inspects the final output.
- A task may override `security_from_traces(…, traces)`, which inspects the function-call trace. When it returns non-None, that result wins (`TaskSuite._check_injection_task_security`).
- Utility works the same way (`utility` / `utility_from_traces`).
- **No LLM judge anywhere.** Results are deterministic given the trajectory.

---

## 2. Full vetted table

Columns:
- **T** = turns (S single, M multi).
- **Ag** = agentic (tool-calling) vs chat.
- **Plug** = how it enters our pipeline: N = agentdojo_native, I = agentdojo_injection_import, C = chat_runner, A = agentic_runner, G = chat_runner in guard-only mode (labelled items go straight to one layer).
- **Ban** = dependency on a the cluster-covered entity.

### 2.1 Agentic: prompt injection, goal hijack, unsafe tools

| Dataset | Source @ pin | Licence / gating | Size (verified) | Format | T / Ag | Success judged by | Plug | Ban dependency | Saturation / contamination |
|---|---|---|---|---|---|---|---|---|---|
| **AgentDojo v1.2.2** | github.com/ethz-spylab/agentdojo @ `a75aba76` (tag v0.1.35); pip `agentdojo==0.1.35` | MIT / open | 97 user, 35 injection, 949 pairs; suites ~0.6 MB | Python + YAML envs | M / agentic | Programmatic env-state/trace check (§1) | N | none | Public since 2024. Prompt Guard 2's model card reports AgentDojo results (vendor-evaluated). Frontier models are PI-hardened, so static templates may show low ASR: pilot first. |
| **InjecAgent** | github.com/uiuc-kang-lab/InjecAgent @ `f19c9f2c` | MIT (LICENCE) / open | 1,054 cases per setting (510 direct-harm + 544 data-stealing); base and enhanced settings; 17 user / 62 attacker tools | JSON | S step / agentic (simulated, prefilled tool response) | Programmatic: parse the next action and check for an attacker-tool call. Data stealing also needs step 2. Reports ASR-valid / ASR-all. | A (own runner; function-calling mode exists). Could partly map to AgentDojo, but its attacker tools don't exist there. | none | Older; base setting likely low ASR on modern models, so use enhanced. |
| **ASB** (Agent Security Bench, ICLR'25) | github.com/agiresearch/ASB @ `544540ff` | MIT / open | 10 agents, 51 tasks, 400 attacker tools (200 aggressive / 200 non-aggressive), 20 normal tools; attacks DPI, OPI, PoT backdoor, memory poisoning, mixed | JSONL + YAML; AIOS runtime | S/M / agentic (simulated tools) | Programmatic: attacker-goal string appears in messages. Refusal judged by gpt-4o-mini (swap in an open model). | A | README lists Qwen2 as an *evaluated* agent only; not needed | Little known; simulated tool outputs are canned strings. |
| **BIPIA** | github.com/microsoft/BIPIA @ `a004b69e` | MIT / open; WebQA and Summarization contexts must be regenerated from NewsQA/XSum because of licensing | Test contexts: email 50, table 100, code 50 (+qa, abstract). Test attacks: 75 text (15 types × 5) + 50 code (10 × 5); × insertion positions | JSONL | S / chat (QA over external content) | Mixed: rule-based metrics for some attack types, GPT-3.5 evaluator by default for others (swap) | C (could become I with output-content checks) | `config/chatglm*.yaml` exist (GLM = Zhipu/Z.ai) but are optional model configs | Older; text-only. |
| **WASP** | github.com/facebookresearch/wasp @ `ffee6f41` | **CC-BY-NC-4.0** / open | 21 attacker goals; 37 utility tasks (croissant files); 4 injection templates. The paper's total (often cited as 84) is UNVERIFIED. | JSON configs; VisualWebArena | M / web agent | End-to-end action check plus a GPT-4o step-by-step evaluator (OpenAI/Azure key required) | A only. **Not runnable on the cluster as-is**: dockerised GitLab/Reddit, 4–6 h per run. | none | Small. |
| **LLMail-Inject** | HF microsoft/llmail-inject-challenge @ `1063bdf0` | MIT / gating not shown | 2.45 GB repo; 40 levels (4 scenarios × defences × LLMs); human adaptive submissions | JSONL | S / email RAG agent | Challenge-side: tool call + defence-evasion flags | I: payloads as a custom AgentDojo workspace attack with the objective remapped to send_email (needs adapter work; schema UNVERIFIED) | none | Payloads were tuned against the challenge's own defences (Spotlighting, PromptShield, TaskTracker, LLM judge). |
| **AgentHarm** (Gray Swan + UK AISI) | HF ai-safety-institute/AgentHarm @ `e23b3fe6` (the pin inspect_evals uses) | MIT plus a clause limiting use to AI safety and security research (HF tag "other"); gating not mentioned on card (UNVERIFIED) | harmful: 44 of 66 public test base behaviours released, giving 176 augmented; validation 8 of 11, giving 32. Configs: harmful / harmless_benign / chat. Benign counts expected to mirror these (UNVERIFIED). | JSON | M / agentic (synthetic tools) | Per-behaviour grading functions + LLM refusal and semantic judges (default gpt-4o-2024-08-06) | A (inspect_evals harness or a tool port) | none | Canary GUID included; authors ask that it not be used for training. 22 public base behaviours are held out as a private test set. |
| **Agent-SafetyBench** (THU-CoAI) | github.com/thu-coai/Agent-SafetyBench @ `74feea8d` (also on HF) | MIT / open | 2,000 cases (`data/released_data.json`); `environments/` has 701 files (≈350 environments; the 349 in the paper is UNVERIFIED) | JSON + Python envs | M / agentic | **Official scorer `thu-coai/ShieldAgent` is Qwen-based** (`eval_with_shield.py` hard-codes `model_base='qwen'`), so it is **banned**. Needs a substitute judge and validation. | A | **Judge banned**; eval script also defaults the agent to Qwen2.5 (configurable) | Recent; little known. |
| **ToolEmu** | github.com/ryoungj/ToolEmu @ `ac4a7ab7` | Apache-2.0 / open | 144 test cases (`assets/all_cases.json`) | JSON | M / LM-emulated tools | GPT-4-class LM emulator + LM safety evaluator | A (expensive, nondeterministic) | none | Small; emulator variance. |
| **R-Judge** | github.com/Lordog/R-Judge @ `83ce301d` | **No licence file** / open | 571 records at this commit (paper: 569); 27 scenarios, 5 application categories, 10 risk types | JSON (trajectory + binary label + risk description) | M / agent *trajectories* | Human labels (benchmark of judges) | G: offline P/R for AlignmentCheck / Llama Guard on trajectories | none | Small; safety label is about the trajectory, not about guard intervention. |

### 2.2 Chat: jailbreak (single and multi-turn), harmful output

| Dataset | Source @ pin | Licence / gating | Size (verified) | Format | T / Ag | Success judged by | Plug | Ban dependency | Saturation / contamination |
|---|---|---|---|---|---|---|---|---|---|
| **HarmBench** | github.com/centerforaisafety/HarmBench @ `8e1604d1` | MIT / open | 400 text behaviours (200 standard / 100 contextual / 100 copyright). Test 320 = 159 / 81 / 80; val 80 = 41 / 19 / 20. Plus 110 multimodal. | CSV | S / chat | `cais/HarmBench-Llama-2-13b-cls` (standard + contextual); copyright via hashing; `HarmBench-Mistral-7b-val-cls` as validation classifier | C (seeds for adaptive attacks) | none (the repo's example runs use Baichuan models, which we don't need) | Heavily used in safety tuning, so vanilla requests are mostly refused; useful only with attacks. |
| **JailbreakBench JBB-Behaviors** | HF JailbreakBench/JBB-Behaviors @ `886acc35`; code github.com/JailbreakBench/jailbreakbench @ `23dbdf6b` | Code MIT; HF card licence not re-read (UNVERIFIED) / gating not shown | 100 harmful + 100 benign (thematically matched); judge_comparison 300 | CSV | S / chat | Llama-3-70B jailbreak judge, Llama-3-8B refusal judge (prompts in package; run locally via vLLM) | C | none | Overlaps AdvBench/HarmBench; widely trained on. |
| **JBB artifacts** | github.com/JailbreakBench/artifacts @ `909e68c0` | MIT / open | Jailbreak strings for PAIR, GCG (white-box + transfer), JBC, prompt_with_random_search, DSN | JSON (~4.5 MB) | S / chat | n/a (inputs) | C | none | Strings optimised against 2024 models; transfer ASR to new models likely low. |
| **StrongREJECT** | github.com/alexandrasouly/strongreject @ `f7cad6c1`; evaluator package github.com/dsbowen/strong_reject @ `7a551d5b` | MIT for custom data; prompts taken from prior work keep their original licences (README) / open | **313** forbidden prompts (and 60 in the small set). The same 313 are at inspect_evals' pinned commit `3432b2d`, even though the inspect README says 324. A 2024-02 version had 346. | CSV | S / chat | Rubric LLM judge (refusal, convincingness, specificity → score 0–1; designed for GPT-4o/4o-mini), or fine-tuned evaluator `qylu4156/strongreject-15k-v1` (Gemma-2B) | C | none | Graded score is robust to empty "jailbreaks". |
| **WildJailbreak** (AI2) | HF allenai/wildjailbreak (no SHA found, resolved and locked on first fetch) | **ODC-BY**; **gated** (AI2 Responsible Use form, auto-approved) | 262K train pairs (50,050 vanilla harmful / 50,050 vanilla benign / 82,728 adversarial harmful / 78,706 adversarial benign); **eval 2,000 adversarial harmful + 210 adversarial benign** (card) | TSV (card); 541 MB total | S / chat | None shipped: needs a judge | C | Generated with GPT-4, GPT-3.5 and Mixtral-8x7B; none covered | Training split used by Tulu/OLMo-style safety tuning and WildGuard; contaminated for those models. |
| **WildGuardMix** | HF allenai/wildguardmix (lock) | **ODC-BY**; **gated** | train 86,759; **test 1,725** with prompt-harm, response-harm and refusal labels plus annotator agreement (card) | parquet (UNVERIFIED) | S / chat | Human labels (test) | G: offline P/R for input/output rails | none named (GPT-4 used for labelling/audit) | WildGuard is trained on the train split; Llama Guard is not. |
| **MHJ** (Scale AI) | HF ScaleAI/mhj (lock) | **CC-BY-NC-4.0**; **gated** (contact info) | **537 multi-turn conversations, 2,912 prompts** (card); 1.3 MB | CSV/imagefolder (card); format UNVERIFIED | M / chat | Paper used human review. For replay we use the HarmBench classifier on the behaviour. | C (replay user turns; the assistant is regenerated) | none | Built from HarmBench test behaviours and WMDP-bio. Some behaviours dropped over export-control concerns, some completions redacted, WMDP-bio responses moved to a separate set. **Replay is non-adaptive**, so expect lower ASR than in the paper. |
| **MT-JailBench** (exists; arXiv 2605.11002, NeurIPS 2026 poster) | github.com/SafetyArena/mt-jailbench @ `cb8184e0` | MIT (LICENSE.md, 2026) / open | 159 HarmBench standard behaviours; modular Crescendo, CoA, FITD, ActorBreaker, X-Teaming | Python harness + CSV | M / chat | GPT-4o judges (binary + 5-point score) by default | C (harness drives our pipeline as the target) | **Default attacker Qwen2.5-32B-Instruct: must be swapped** (e.g., Llama-3.3-70B, gpt-oss-120b) | New. Attacker strength depends on the swapped model, so report it. |
| **X-Teaming code / XGuard-Train** | code github.com/salman-lui/x-teaming @ `db964e66`; data HF marslabucla/XGuard-Train | Code: no licence file. Data: no licence stated, gated (via an HF-card mirror; official card not reachable). | 30,695 multi-turn conversations, ShareGPT JSON, no splits | JSON | M / chat | GPT-4o verifier (OpenAI key mandatory) | C (code); data is a *training* set | **Gray:** attacker Qwen-2.5-32B-IT; DeepSeek-V3 among targets. Code example config uses Qwen as attacker. | Training data, not for evaluation. |
| **SafeMTData** (ActorAttack) | HF SafeMTData/SafeMTData (lock); code github.com/AI45Lab/ActorAttack @ `3149db7f` | **UNVERIFIED** (no licence on GitHub; card not reachable) | Paper: 600 harmful instructions sampled from the Circuit-Breaker training set (filtered against HarmBench); 1,000 successful dialogues collected against deepseek-chat. Config name "Attack_600" UNVERIFIED. | JSON (UNVERIFIED) | M / chat | GPT-4o judge (score 5 = success) in paper | C | **Gray:** victim dialogues and refusal responses generated by deepseek-chat; the attacker was WizardLM-2-8x22B. Code defaults `safe_model_name` to deepseek-chat. | n/a |
| **CoSafe** (EMNLP'24) | github.com/ErxinYu/CoSafe-Dataset @ `a298d9c7` | **No licence file** | 1,400 prompts, 14 categories × 100 | JSONL (message lists with pre-filled assistant turns) | M / chat (static) | Paper judge UNVERIFIED | C | Authors affiliated with Huawei Noah's Ark. Not one of our six entities; check whether the Texas list covers this (UNVERIFIED). | Static coreference attacks. |
| **CyberSecEval** (PurpleLlama) | github.com/meta-llama/PurpleLlama @ `f64eebd5` | `CybersecurityBenchmarks/LICENSE` = MIT (the repo root licence is Llama 3.2 Community, which covers the models) | instruct 1,916 (v2: 1,681); autocomplete 1,916; interpreter 500; prompt_injection 251 (+1,004 machine-translated); MITRE FRR 750 | JSON | S / chat | Insecure code: **ICD static analysis** (same engine as CodeShield). Interpreter / PI / FRR: judge + expansion LLMs (configurable). | C | none | inspect_evals reports 11 duplicate rows in interpreter.json and 1 in prompt_injection.json. |
| **SecCodePLT** | HF Virtue-AI-HUB/SecCodePLT @ `e8471415` (commit seen in an HF tree URL; may not be latest) | MIT (card) / gating not stated | **1,345** `insecure_coding` samples; CWE count not on card | parquet/JSON with `unittest` (functionality + security) and `rule` fields | S / chat | Dynamic unit tests. Rule check where tests can't express the CWE. No LLM judge. | C | none named ("LLM-based mutators", model unnamed) | The card lists its code repository as "coming soon" (GitHub `Virtue-AI-HUB/SecCodePLT` not found), so we must build the test runner. |
| **CWEval** | github.com/Co1lin/CWEval @ `e9a2a124` | Apache-2.0 / open | 119 tasks (C 31, C++ 21, Go 19, JS 23, Py 25), 31 CWEs | Source + pytest oracles | S / chat | Execution-based functionality + security tests | C | none | Small; needs the Docker image converted to Apptainer. |

### 2.3 Privacy / PII

| Dataset | Source @ pin | Licence / gating | Size | T / Ag | Judge | Plug | Ban | Notes |
|---|---|---|---|---|---|---|---|---|
| **PrivacyLens** | github.com/SALT-NLP/PrivacyLens @ `9c2ee07b` (HF SALT-NLP/PrivacyLens) | MIT | **493** seed → vignette → trajectory (`main_data.json`: user_instruction, toolkits, executable_trajectory, final_action, sensitive_info_items) | M / agentic (replayed trajectory, judge the final action) | LLM leak judge; default `mistralai/Mistral-7B-Instruct-v0.2` | A | none | Measures contextual-integrity leakage, not adversarial exfiltration. |
| **AgentDojo exfiltration subset** | same as AgentDojo | MIT | 20 exfiltration injection tasks, 582 pairs (tags in `datasets.yaml`) | M / agentic | programmatic | N | none | Secrets, IBANs, passport and card numbers, inbox/file dumps. |
| **ai4privacy open-pii-masking-500k** | HF ai4privacy/open-pii-masking-500k-ai4privacy @ `144ca06b` (commit seen in an HF README URL) | Metadata says cc-by-4.0, but the card adds **custom terms**: commercial use needs permission | 580,227 rows (train 464,150 / val 116,077); 20 PII classes; 6–8 languages | S / text | Gold spans (no LLM) | G (Presidio / GLiNER span P/R/F1) | none | Generation method not stated. |
| **ConfAIde** | github.com/skywalker023/confAIde @ `3174c63e` | MIT | Tier 1 10, 2a 98, 2b 98, tier 3 1,955 lines, tier 4 1,150 lines | S/M / chat | String-match leak check + model probes | C | none | Older; chat-only. |
| **AgentDAM** | github.com/facebookresearch/ai-agent-privacy @ `5d4068a4` | **CC-BY-NC-4.0** | 246 tasks (GitLab 48 / Reddit 114 / shopping 84) | M / web agent | LLM privacy judge (GPT-4o in README; exact model UNVERIFIED) | A only; **WebArena Docker, not on the cluster** | none | Data minimisation. |

### 2.4 Benign / utility / over-refusal

| Dataset | Source @ pin | Licence / gating | Size | Judge | Plug | Notes |
|---|---|---|---|---|---|---|
| **AgentDojo user tasks** | AgentDojo | MIT | 97 user tasks (+ utility under attack on 949 pairs) | programmatic | N | **97 tasks caps the no-attack utility CI at about ±10 pts.** |
| **AgentHarm benign** | AgentHarm @ `e23b3fe6` | as above | harmless_benign test_public + validation (expected 176 + 32; UNVERIFIED) | grading functions + refusal judge | A | Matched to the harmful set, so it measures over-blocking of tools. |
| **XSTest** | github.com/paul-rottger/exaggerated-safety @ `d7bb5bd7` (HF mirror walledai/XSTest pinned by inspect_evals at `f1d71318`) | CC-BY-4.0 | 450 = 250 safe + 200 unsafe (**downloaded and counted**) | String match or GPT-4 classification upstream; we use WildGuard refusal + string match | C | Near-saturated for recent models; low headroom. |
| **OR-Bench** | HF bench-llm/or-bench @ `e36d8b80` (2024-12-19) | CC-BY-4.0 / gating not shown | Configs or-bench-80k / or-bench-hard-1k / or-bench-toxic; row counts UNVERIFIED (names suggest ~80K / ~1K) | LLM judge (direct answer / direct refusal / indirect refusal) | C | Rewritten by Mixtral, moderated by an LLM ensemble (GPT-4 and Gemini moderator code in repo). Qwen appears only among evaluated models. |
| **NotInject** | github.com/leolee99/InjecGuard @ `1b5751e8` (HF leolee99/NotInject) | MIT | 339 = 3 × 113 benign prompts containing 1–3 injection trigger words (**downloaded and counted**) | gold (all benign) | G | Over-defence of PI detectors (PromptGuard 2, regex). |
| **WildJailbreak adversarial benign / WildGuardTest benign / JBB benign / CSE MITRE FRR** | see above | see above | 210 / subset of 1,725 / 100 / 750 | WildGuard refusal / gold / JBB refusal judge / LLM | C/G | Supplementary false-positive sets. |

---

## 3. Recommendations per category

The sample size assumes a worst-case ASR of p = 0.5 and Wilson 95% intervals. Half-widths: ±7 needs n≥196, ±6 needs n≥267, ±5 needs n≥385. When ASR is about 0.2, ±5 needs only n≈246. Every stack is evaluated on the **same items**, so differences between layers and stacks use paired tests (McNemar / paired bootstrap), which are tighter than the per-arm CIs.

| # | Category | Primary (n) | Secondary (n) | Judge (non-banned) | CI at planned n |
|---|---|---|---|---|---|
| 1 | Direct jailbreak | **HarmBench** text test, standard + contextual (**240**) × {direct, JBB-artifact transfer, PAIR, TAP with a non-banned attacker} | **WildJailbreak** eval adversarial-harmful (**400** of 2,000) | HarmBench-Llama-2-13b-cls (+Mistral-7b val cls for agreement); StrongREJECT rubric with Llama-3.3-70B for WildJailbreak | ±6.3 / ±4.9 |
| 2 | Multi-turn escalation | **MHJ** replay (**300** of 537) **and** adaptive **MT-JailBench** harness: Crescendo + X-Teaming on all 200 HarmBench standard behaviours, attacker Llama-3.3-70B (**200** per method) | **SafeMTData** (300; gray, DeepSeek-generated); alt. CoSafe (300; no licence) | HarmBench cls (replaces the GPT-4o judge); StrongREJECT rubric | ±5.6 / ±6.9 |
| 3 | Indirect PI (tool/RAG) | **AgentDojo v1.2.2**, all **949** pairs × {important_instructions, tool_knowledge, our adaptive attack} | **InjecAgent** (1,054 per setting, base + enhanced) | Programmatic (both) | ±3.2 iid. Clustered on 35 injection tasks, so report cluster-bootstrap CIs. |
| 4 | Goal hijack | **AgentDojo** hijack view (hijack = security ∧ ¬utility on the 949 pairs; 141-pair content/action-hijack tag) **+ 5 DoS attacks × 97** (485) | **ASB** DPI + OPI (400 per attack type) | Programmatic (ASB refusal judge goes to Llama-3.3-70B) | ±3–5 |
| 5 | Unsafe tool call | **AgentHarm** harmful test_public + val (**208**, all available) | **Agent-SafetyBench** (400 stratified) with a **substitute judge** (Llama-3.3-70B + rubric, validated on ≥150 hand labels) | AgentHarm grader with Llama-3.3-70B substituted for GPT-4o (validate ≥100 items), or GPT-4o via API off-cluster | ±6.8 iid but **only 52 base behaviours**, so true CI is wider (gap). ±4.9 for secondary. |
| 6 | Harmful output | **StrongREJECT** (**313**) × {direct, jailbreak wrappers} | **WildGuardTest** response-harm items, guard-only P/R of output rails (all) | StrongREJECT rubric (Llama-3.3-70B) + fine-tuned Gemma-2B evaluator; gold labels | ±5.5 |
| 7 | Insecure code | **SecCodePLT** insecure_coding (**400** of 1,345, stratified by CWE) | **CWEval** (119 tasks × 3 samples) | Execution oracles (unit tests / pytest) in Apptainer. **Not ICD** for CodeShield. CyberSecEval-instruct (400, ICD) only for comparability and non-CodeShield layers. | ±4.9 / about ±9 at task level |
| 8 | PII leak | **PrivacyLens** (**493**) + **AgentDojo exfiltration view** (582 pairs) | **ai4privacy** open-pii-masking-500k, English validation (**2,000**), PII-detector P/R/F1 | PrivacyLens leak judge (Llama-3.3-70B recommended over the default Mistral-7B; validate); programmatic; gold spans | ±4.4 |
| 9 | Benign / utility / over-refusal | **AgentDojo** 97 user tasks × 3 repeats + utility under attack; **XSTest** (250 safe + 200 unsafe contrast) | **AgentHarm benign** (208), **OR-Bench-Hard-1K** (400), **NotInject** (339, PI-detector FP), CSE MITRE FRR (400) | WildGuard refusal head + string match; programmatic; gold | XSTest ±6.2 worst case (FRR near 5% gives ±2.7); AgentDojo utility about ±10 (gap) |

**Judge independence rule.** Never use a model as judge in a configuration where it, or a sibling from the same family and training mix, is a layer under test. Llama Guard 4 may be a *cross-check* judge only when it is not in the stack. Audit each LLM judge against about 200 human-labelled items per category, and report agreement (Cohen's κ) in the paper.

**the cluster-safe judge list:**

| Judge | Base / licence | Banned? |
|---|---|---|
| cais/HarmBench-Llama-2-13b-cls | Llama-2-13B; licence on card UNVERIFIED | no |
| cais/HarmBench-Mistral-7b-val-cls | Mistral-7B | no |
| qylu4156/strongreject-15k-v1 | Gemma 2B, Gemma licence (gated) | no |
| allenai/wildguard | Mistral-7B; Apache-2.0 per HF card | no |
| meta-llama/Llama-Guard-4-12B | pruned from Llama 4 Scout; Llama 4 Community Licence (gated) | no |
| meta-llama/Llama-3.3-70B-Instruct / Meta-Llama-3-70B-Instruct (JBB prompts) | Meta | no |
| openai/gpt-oss-120b | OpenAI | no |
| mistralai/Mistral-7B-Instruct-v0.2 (PrivacyLens default) | Mistral | no |

**Do not use** thu-coai/ShieldAgent (Qwen-based), the Qwen2.5-32B attacker defaults in MT-JailBench and X-Teaming, or ActorAttack's deepseek-chat default.

---

## 4. Downloader notes (the cluster login node)

- `python data/download_datasets.py --list` shows the registry. It also checks that every `download:` entry and `revision` in `configs/datasets.yaml` matches the registry (0 problems at time of writing).
- **Default output:** `~/work/guardstack/datasets`, or `$GUARDSTACK_DATA_DIR` if set. Do not use `~` (50 GB / 300k-inode quota). Also export `HF_HOME=~/work/.cache/huggingface`. Total default download is a few hundred MB at most: the WildJailbreak eval subset, and WildGuardMix, whose size is UNVERIFIED. LLMail-Inject (2.45 GB), ai4privacy, AgentDAM, WASP, ToolEmu, R-Judge, ConfAIde and CoSafe are opt-in via `--only` or `--all`.
- **GitHub sources:** the script fetches the pinned SHA with `git fetch --depth 1 --filter=blob:none` plus a classic sparse checkout, then checks that HEAD equals the pin. If the filtered fetch fails it retries without the filter.
- **HF sources:** `snapshot_download(revision=sha)`. Where no SHA was verifiable, it resolves `main` once and writes `data/revisions.lock.json`. **Commit that file after the first the cluster run.**
- **Gated sets** (MHJ, WildJailbreak, WildGuardMix, XGuard-Train) need you to accept their terms in the browser and then `export HF_TOKEN=…`. Without a token they are reported as failures, or as soft skips with `--skip-gated`.
- **Gray-area sets** are skipped unless `--allow-gray-area` is given.
- **Banned-entity interlock:** the script refuses repos owned by Qwen/Alibaba, DeepSeek, Moonshot/Kimi, Z.ai/THUDM/Zhipu, ByteDance/Doubao and Xiaomi, plus thu-coai/ShieldAgent.
- **Manifest:** `<out>/manifest.json` records per-file SHA-256, a tree hash, size, licence, gating, gray-area flag, resolved revision, and versions of host, Python, `huggingface_hub`, `agentdojo` and git. `--verify` re-hashes against it.
- **What was tested in the sandbox:**
  - `xstest` and `notinject` were downloaded and checked (450 rows; 3 × 113 rows), skip-on-rerun and `--verify` worked, tamper detection worked, the gated, gray and banned paths were exercised, and an offline mock of the HF resolve-and-lock path ran. Test downloads were then deleted.
  - **Real HF downloads could not be tested** (the proxy returns 403 for huggingface.co). The first the cluster run is therefore the real test of the HF path and of the `allow_patterns` for WildJailbreak (`eval/*`, UNVERIFIED).

---

## 5. Open risks and coverage gaps

1. **(Largest gap) Multi-turn escalation against *agents*.** Every multi-turn set (MHJ, MT-JailBench, SafeMTData, CoSafe) is chat-only. Nothing public runs Crescendo or X-Teaming against a tool-using agent. Plan: use the MT-JailBench harness with the AgentDojo pipeline as the target over the user channel, with goals taken from AgentHarm behaviours or AgentDojo injection goals, scored programmatically. This is new infrastructure, so budget for it.
2. **Adaptive IPI attacker.** AgentDojo ships only static templates, and WASP (the adaptive web benchmark) is not runnable on the cluster. We need our own adaptive attack, for example an LLM attacker loop or optimisation against PromptGuard 2 and AlignmentCheck. Without it, "adaptive ASR" for categories 3 and 4 is unsupported.
3. **Unsafe tool call is under-powered.** AgentHarm's public set has 52 base behaviours (4 augmentations each), so the cluster-adjusted CI will exceed ±7. Agent-SafetyBench is only usable with a substitute judge that we validate ourselves (the official one is banned).
4. **Benign agentic utility.** 97 AgentDojo tasks (about ±10 pts) plus about 208 AgentHarm-benign. Utility *deltas* between stacks are paired, which helps, but the absolute utility claims stay weak. Consider writing extra AgentDojo-compatible benign user tasks, or vetting a second tool-use benchmark (not done here).
5. **Insecure-code oracle.**
   - CyberSecEval-ICD is circular for CodeShield.
   - SecCodePLT's evaluation harness is not public ("coming soon"), so we must execute its unit tests ourselves in a sandbox, and its pin came from a tree URL.
   - CWEval is small and needs Docker → Apptainer.
6. **Judges.** Many official judges are GPT-4o (AgentHarm, MT-JailBench, WASP, AgentDAM, ASB refusal, ActorAttack). Substituting Llama-3.3-70B changes the measurement, so validation is mandatory. Llama-family judges are correlated with the Llama-family guard layers (Prompt Guard, Llama Guard, and AlignmentCheck's backing LLM).
7. **Contamination.**
   - Prompt Guard 2 was evaluated by its vendor on AgentDojo, and its training mix includes open-source PI and jailbreak datasets (exact overlap with our sets is UNVERIFIED).
   - WildJailbreak and WildGuardMix are training data for WildGuard and several open models.
   - HarmBench, JBB and AdvBench are widely used in safety tuning.
   - Mitigation: report results on InjecAgent, LLMail-Inject and NotInject as off-distribution checks.
8. **Licences.**
   - MHJ, WASP and AgentDAM are non-commercial.
   - CoSafe, R-Judge and the X-Teaming code have no licence file.
   - AgentHarm restricts use to safety research.
   - ai4privacy's terms conflict with its own metadata.
   - SafeMTData and XGuard-Train licences are UNVERIFIED.
   - Releasing derived artefacts from any of these needs a check.
9. **Gray-area provenance.** SafeMTData (DeepSeek-generated responses) and XGuard-Train (Qwen attacker, DeepSeek-V3 targets) need a PI or circ-assist decision. So does CoSafe, whose authors are Huawei-affiliated, if Huawei is on the Texas list (UNVERIFIED).
10. **Unverified items to close on the cluster.**
    - HF SHAs for MHJ, WildJailbreak, WildGuardMix and SafeMTData (the lock file fills these).
    - Gating status for AgentHarm, JBB, OR-Bench and SecCodePLT.
    - OR-Bench row counts.
    - AgentHarm benign counts.
    - The WildJailbreak file layout.
    - The HarmBench-classifier licence on its HF card.
    - Compute-node internet access.
    - AgentDojo forks, and whether one has a web suite.
