# Annotated Bibliography: The Marginal Cost of Defense-in-Depth for LLM Agents

Compiled 2026-11-03 for the IEEE S&P 2027 (cycle 2) submission. All summaries are paraphrased.

**Verification legend**
- **[V] Verified**: title, authors and date (and venue where given) were checked this session on the arXiv abstract/PDF/HTML page, ACL Anthology, PMLR, the NeurIPS proceedings, a publisher page, or a DOI resolver.
- **[P] Partial**: the work exists and its title is confirmed, but at least one field (authors, venue, pages) was not confirmed. These are **not** in `refs.bib` unless the missing field is simply left out.
- **[U] Unverified**: could not be confirmed. Not in `refs.bib`.

The BibTeX key in `paper/refs.bib` is shown as `key`.

**Title changes to watch** (arXiv versions vs. the names people use):
- Meta SecAlign: v3 is titled *Meta-SecAlign: Training LLMs against Prompt Injection for Robust Agents*.
- Progent: the current version is *Progent: Securing AI Agents with Privilege Control*. v1 was *Programmable Privilege Control for LLM Agents*.
- ActorAttack (arXiv 2410.10700): now titled *LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts*. It was first posted as *Derail Yourself: Multi-turn LLM Jailbreak Attack through Self-discovered Clues*.
- InjecGuard became **PIGuard** at ACL 2025, with two more authors.
- Siege: the arXiv title says "Autonomous"; the ICLR'25 workshop title does not.
- He et al. 2017: the real title is "Adversarial Example **Defenses**: Ensembles of Weak Defenses are not Strong", plural, not "defense".

---

## (a) Guardrail systems and classifiers

**LlamaFirewall: An open source guardrail system for building secure AI agents** [V] `chennabasappa2025llamafirewall`
- Chennabasappa, Nikolaidis, Song, Molnar, Ding, Wan, Whitman, Deason, Doucette, Montilla, Gampa, de Paola, Gabi, Crnkovich, Testud, He, Chaturvedi, Zhou, Saxe (Meta). arXiv 2505.03574, May 2025. https://arxiv.org/abs/2505.03574
- Summary: Meta's open-source agent guardrail framework, made of PromptGuard 2 (a small DeBERTa jailbreak/injection classifier), AlignmentCheck (an LLM that audits the agent's chain of thought for goal drift), CodeShield (static analysis of generated code), and custom regex/LLM scanners. On static AgentDojo replays (the `important_instructions` attack, Llama 4 Maverick as the AlignmentCheck backbone), ASR/utility were: no guardrail 17.6%/47.7%; PG2 only 7.5%/47.0%; AlignmentCheck only 2.9%/43.1%; both 1.75%/42.7%. PG2-86M takes about 92 ms per 512-token classification on an A100. CodeShield finishes in about 60 ms for roughly 90% of inputs.
- Why it matters: this is the **closest existing per-layer measurement**: a two-layer stack, static attacks only. Their numbers imply a static MBR for AlignmentCheck after PG2 of about (7.53−1.75)/7.53 ≈ 0.77. Our H1 predicts this falls sharply under adaptive attack. It is also our primary layer source.

**NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails** [V] `rebedea2023nemo`
- Rebedea, Dinu, Sreedhar, Parisien, Cohen (NVIDIA). EMNLP 2023 System Demonstrations, pp. 431–445. DOI 10.18653/v1/2023.emnlp-demo.40. arXiv 2310.10501.
- Summary: A runtime toolkit that wraps an LLM application in programmable "rails" (input, dialog, retrieval, execution, output), written in the Colang DSL and run by an event-driven runtime. Rails can call classifiers, LLM self-checks, or external tools.
- Why it matters: the second major open-source stack in our pool. Its rail types map directly onto our layer positions.

**Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations** [V] `inan2023llamaguard`
- Inan, Upasani, Chi, Rungta, Iyer, Mao, Tontchev, Hu, Fuller, Testuggine, Khabsa. arXiv 2312.06674, Dec 2023.
- Summary: Instruction-tunes Llama-2-7B to classify prompts and responses against a configurable safety taxonomy. The taxonomy is supplied in the prompt, so the policy can be changed at inference time.
- Why it matters: the archetype of "LLM-as-safety-classifier". Llama Guard variants are the most likely same-type pair in our H3 redundancy test (for example LG3 vs LG4, or LG vs ShieldGemma).

**Llama Guard 3-1B-INT4: Compact and Efficient Safeguard for Human-AI Conversations** [V] `fedorov2024llamaguard3`
- Fedorov, Plawiak, Wu, Elgamal, Suda, Smith, Zhan, Chi, Hulovatyy, Patel, Liu, Zhao, Shi, Blankevoort, Pasupuleti, Soran, Delpierre Coudert, Alao, Krishnamoorthi, Chandra. arXiv 2411.17713, Nov 2024.
- Summary: A pruned and quantized 1B Llama Guard 3 for on-device moderation, reporting accuracy comparable to larger guards at a fraction of the memory.
- Why it matters: a cheap, same-family variant that is useful for the cost axis and for testing whether a small variant is redundant with the 8B guard.

**Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations** [P] (no bib)
- arXiv 2411.10414. The title was confirmed via a mirror. Authors were not verified. Multimodal only; low relevance.

**Llama Guard 4 (model card)** [P] `meta_llamaguard4`
- Meta, developer.meta.com model card (the URL redirects to dev.meta.ai). I confirmed the page exists, but its content was not readable through the fetch tool. There is no paper, so cite it as a model card. The 12B size is widely reported but **not verified here**.

**Llama Prompt Guard 2 (model card, 22M/86M)** [V] `meta2025promptguard2`
- Meta, Hugging Face model card `meta-llama/Llama-Prompt-Guard-2-22M`, Llama 4 collection (2025).
- Summary: Binary benign/malicious classifiers that flag explicit attempts to override instructions. The 22M model is DeBERTa-v3-xsmall per its card; the 86M backbone was not checked. They have a 512-token window; the card recommends chunking longer inputs and itself warns that adaptive attackers may evade the model.
- Why it matters: this is our Layer-1 candidate. The 512-token window is the structural common-mode weakness that Prompt Overflow exploits (cluster c).

**ShieldGemma: Generative AI Content Moderation Based on Gemma** [V] `zeng2024shieldgemma`
- Zeng, Liu, Mullins, Peran, Fernandez, Harkous, Narasimhan, Proud, Kumar, Radharapu, Sturman, Wahltinez (Google). arXiv 2407.21772, Jul 2024.
- Summary: Gemma-2-based policy classifiers (2B/9B/27B) for four harm types, trained in part on synthetic data. The paper reports gains over Llama Guard on public benchmarks.
- Why it matters: a second-vendor LLM classifier. It is a natural redundancy partner for Llama Guard in H3.

**WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs** [V] `han2024wildguard`
- Han, Rao, Ettinger, Jiang, Lin, Lambert, Choi, Dziri (AI2/UW). arXiv 2406.18495, Jun 2024. (NeurIPS 2024 D&B is commonly cited but was **not confirmed** here, so the bib uses arXiv.)
- Summary: A single 7B model that jointly detects prompt harmfulness, response harmfulness, and refusal. It is trained on the large WildGuardMix dataset, which includes adversarial (in-the-wild) jailbreaks.
- Why it matters: its refusal-detection head is relevant to how we measure utility and over-refusal.

**Granite Guardian** [V] `padhi2024granite`
- Padhi, Nagireddy, Cornacchia, Chaudhury, Pedapati, Dognin, Murugesan, Miehling, Santillan Cooper, Fraser, Zizzo, Hameed, Purcell, Desmond, Pan, Ashktorab, Vejsbjerg, Daly, Hind, Geyer, Rawat, Varshney, Sattigeri (IBM). arXiv 2412.07724, Dec 2024. A NAACL 2025 Industry Track version exists (ACL Anthology 2025.naacl-industry.49, seen in search; pages not verified).
- Summary: IBM's risk-detection guard family. It covers harm categories plus RAG-specific risks (groundedness, context relevance) and agentic function-call hallucination.
- Why it matters: it covers the RAG and tool-call positions, which gives us a heterogeneous layer to contrast with prompt classifiers.

**gpt-oss-safeguard technical report (research preview)** [V] `openai2025gptosssafeguard`
- OpenAI, 29 Oct 2025. https://openai.com/index/gpt-oss-safeguard-technical-report
- Summary: Open-weight 20B and 120B reasoning models, fine-tuned from gpt-oss, that classify content against a developer-supplied policy by reasoning over it at inference time.
- Why it matters: this is a "bring-your-own-policy" reasoning guard, the most expensive class of layer. It is useful for showing where latency cost buys (or fails to buy) marginal block rate.

**AEGIS: Online Adaptive AI Content Safety Moderation with Ensemble of LLM Experts** [V] `ghosh2024aegis`
- Ghosh, Varshney, Galinkin, Parisien (NVIDIA). arXiv 2404.05993, Apr 2024.
- Summary: Releases a human-annotated content-safety dataset and taxonomy, trains LlamaGuard-style guards on it, and proposes online selection among an ensemble of LLM "experts".
- Why it matters: it is NeMo's content-safety lineage, and an early ensemble-of-guards idea, though one that does not measure correlation.

**Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming** [V] `sharma2025constitutional`
- Sharma, Tong, Mu, Wei, Kruthoff, Goodfriend, et al. (43 authors, Anthropic). arXiv 2501.18837, Jan 2025.
- Summary: Input and output classifiers trained on synthetic data generated from a natural-language constitution, evaluated with large-scale human red-teaming for universal jailbreaks.
- Why it matters: a production input+output classifier stack evaluated against adaptive human attackers. It reports compute overhead, a reference point for our cost axis.

**Constitutional Classifiers++: Efficient Production-Grade Defenses against Universal Jailbreaks** [V] `cunningham2026ccpp`
- Cunningham, Wei, et al. (29 authors, Anthropic). arXiv 2601.04603, Jan 2026.
- Summary: Replaces separate input/output classifiers with an "exchange" classifier and deploys a two-stage cascade. A cheap linear probe on the model's activations screens all traffic and escalates about 5.5% to an external classifier, cutting overhead by about 40× relative to the single exchange classifier. The authors report that probe and classifier errors are complementary.
- Why it matters: evidence that heterogeneous layers can be complementary, the counterpoint to H3, and that **gating by cost** (ordering) dominates the cost of a stack. It is a proprietary chat system, not agents.

**Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models** [V] `bhatt2023cyberseceval`
- Bhatt, Chennabasappa, Nikolaidis, Wan, Evtimov, Gabi, et al. (21 authors, Meta). arXiv 2312.04724, Dec 2023.
- Summary: A benchmark of insecure-code generation and cyberattack helpfulness. It introduced the Insecure Code Detector static analyzer that later became CodeShield's basis. (The ICD link is from the LlamaFirewall paper; the CyberSecEval abstract page did not describe ICD.)
- Why it matters: the provenance of the insecure-code layer and its evaluation data.

**GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer** [V] `zaratiana2024gliner`
- Zaratiana, Tomeh, Holat, Charnois. NAACL 2024 (long), pp. 5364–5376. DOI 10.18653/v1/2024.naacl-long.300.
- Summary: A compact bidirectional-encoder NER model that extracts arbitrary entity types given as label prompts.
- Why it matters: the backbone of NeMo's GLiNER PII rail, our PII layer.

**Presidio: Data Protection and De-identification SDK** [P] `microsoft_presidio`
- Microsoft, https://microsoft.github.io/presidio (title confirmed via search; no paper). A rule- and NER-based PII analyzer/anonymizer. It is the second PII layer, which gives a same-task, different-mechanism pair for H3.

**SoK: Evaluating Jailbreak Guardrails for Large Language Models** [V] `wang2025sokguardrails`
- Wang, Ji, Wang, Li, Wu, Wang. arXiv 2506.10597, Jun 2025.
- Summary: A six-dimension taxonomy of over 40 jailbreak guardrails, plus a benchmark of nine (Perplexity, SmoothLLM, Llama Guard, GradSafe, GradientCuff, SelfDefend, WildGuard, Prompt Guard, GuardReasoner). It reports ASR, extra latency, GPU memory and FPR, including multi-turn (ActorAttack, X-Teaming) and adaptive (TAP, LLM-Fuzzer) attacks. It finds no single guardrail dominates, and that most session-level guards fail badly on X-Teaming.
- Why it matters: a **medium novelty threat**. It measures the same axes as us, but evaluates each guardrail **individually**, on chat, not agents. The abstract gestures at optimizing combinations, but the body does not evaluate stacks.

---

## (b) Prompt-injection defenses for agents

### Detection
**DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks** [V] `liu2025datasentinel`
- Liu, Jia, Jia, Song, Gong. IEEE S&P 2025 (proceedings ToC, ML Defenses session, starting p. 2190). arXiv 2504.11358.
- Summary: Fine-tunes a detector LLM through a minimax game against an adaptive injection generator, so the detector is trained against attacks optimized to evade it. It builds on the known-answer detection idea.
- Why it matters: a detector explicitly designed for adaptive robustness. A good candidate for "does adaptive training make a layer non-redundant?"

**PromptArmor: Simple yet Effective Prompt Injection Defenses** [V] `shi2025promptarmor`
- Shi, Zhu, Wang, Jia, Cai, Liang, Wang, Alzahrani, Lu, Kawaguchi, Alomair, Zhao, Wang, Gong, Guo, Song. arXiv 2507.15219, Jul 2025.
- Summary: Prompts an off-the-shelf LLM to find and extract injected instructions from tool output, then removes them by fuzzy matching before the agent sees the data.
- Why it matters: an LLM-sanitizer layer, cheap to add. Its overlap with AlignmentCheck is a natural H3 test.

**PIGuard: Prompt Injection Guardrail via Mitigating Overdefense for Free** (arXiv title: InjecGuard) [V] `li2025piguard`
- Li, Liu, Zhang, Xiao. ACL 2025 (long), pp. 30420–30437. DOI 10.18653/v1/2025.acl-long.1468. arXiv 2410.22770 (v3 still titled InjecGuard, with two authors).
- Summary: Shows that prompt-guard classifiers over-flag benign text containing trigger words, releases an over-defense benchmark (NotInject), and reduces over-defense by mining trigger words and synthesizing benign training text.
- Why it matters: over-refusal is a utility cost that compounds in stacks. The paper supplies a metric and a dataset.

**Attention Tracker: Detecting Prompt Injection Attacks in LLMs** [V] `hung2025attentiontracker`
- Hung, Ko, Rawat, Chung, Hsu, Chen. arXiv 2411.00348, Nov 2024. Findings of NAACL 2025 (anthology ID 2025.findings-naacl.123 seen; pages not verified).
- Summary: A training-free detector that watches whether specific "important" attention heads shift focus from the original instruction to injected text.
- Why it matters: a white-box, mechanism-diverse detector. It is the kind of layer that should be least correlated with text classifiers.

**PromptShield: Deployable Detection for Prompt Injection Attacks** [V] `jacob2025promptshield`
- Jacob, Alzahrani, Hu, Alomair, Wagner. arXiv 2501.15145, Jan 2025.
- Summary: A curated benchmark that separates conversational and application-structured data, plus fine-tuned detectors calibrated for very low FPR. It argues that deployable detectors must be judged at low-FPR operating points.
- Why it matters: it justifies reporting MBR at fixed low-FPR thresholds, not just AUC.

### Model-level
**StruQ: Defending Against Prompt Injection with Structured Queries** [V] `chen2025struq`
- Chen, Piet, Sitawarin, Wagner. USENIX Security 2025. arXiv 2402.06363.
- Summary: Separates instructions and data with reserved delimiter tokens and fine-tunes the model to follow only the instruction channel.

**SecAlign: Defending Against Prompt Injection with Preference Optimization** [V] `chen2025secalign`
- Chen, Zharmagambetov, Mahloujifar, Chaudhuri, Wagner, Guo. ACM CCS 2025, DOI 10.1145/3719027.3744836. arXiv 2410.05451.
- Summary: Uses preference optimization (DPO) to make the model prefer responses to the true instruction over the injected one, improving on StruQ against optimization-based attacks.

**Meta SecAlign** (current arXiv title: *Meta-SecAlign: Training LLMs against Prompt Injection for Robust Agents*) [V] `chen2025metasecalign`
- Chen, Zharmagambetov, Wagner, Guo. arXiv 2507.02735, Jul 2025.
- Summary: An openly released Llama-3.x model trained with an improved SecAlign recipe, evaluated on agentic benchmarks including AgentDojo.
- Why it matters (all three): model-level "layer 0". If the backbone is already SecAligned, H2 predicts that downstream detectors lose marginal value. That is a key ablation.

### System-level
**Defeating Prompt Injections by Design (CaMeL)** [V] `debenedetti2025camel`
- Debenedetti, Shumailov, Fan, Hayes, Carlini, Fabian, Kern, Shi, Terzis, Tramèr. arXiv 2503.18813, Mar 2025.
- Summary: A privileged planner LLM writes a program from the trusted query alone. A custom interpreter tracks control and data flow, and capability policies block untrusted data from steering tool calls.
- Why it matters: provable-by-design security on AgentDojo at a utility cost. It is the upper reference point on our security–utility frontier.

**Progent: Securing AI Agents with Privilege Control** (v1: *Programmable Privilege Control for LLM Agents*) [V] `shi2025progent`
- Shi, He, Wang, Li, Wu, Guo, Song. arXiv 2504.11703, Apr 2025.
- Summary: A DSL for fine-grained tool-call privilege policies, enforced deterministically. Policies can be hand-written or LLM-generated.
- Why it matters: a deterministic tool-gate layer. It is the "AgentShield-style tool gate" we should actually cite (see the note on AgentShield below).

**MELON: Provable Defense Against Indirect Prompt Injection Attacks in AI Agents** [V] `zhu2025melon`
- Zhu, Yang, Wang, Guo, Wang. ICML 2025 (poster). arXiv 2502.05174.
- Summary: Re-executes the agent with the user task masked. If the tool calls are the same with and without the user's intent, they are being driven by injected content.
- Why it matters: it roughly doubles agent cost per step, which makes it a high-cost layer for the frontier.

**IsolateGPT: An Execution Isolation Architecture for LLM-Based Agentic Systems** [V] `wu2025isolategpt`
- Wu, Roesner, Kohno, Zhang, Iqbal. NDSS 2025. arXiv 2403.04960.
- Summary: Hub-and-spoke isolation of apps/tools with mediated inter-app communication, analogous to OS process isolation.

**Securing AI Agents with Information-Flow Control (FIDES)** [V] `costa2025fides`
- Costa, Köpf, Kolluri, Paverd, Russinovich, Salem, Tople, Wutschitz, Zanella-Béguelin (Microsoft). arXiv 2505.23643, May 2025.
- Summary: A planner that taint-tracks confidentiality and integrity labels, enforces policies deterministically, and can hide labeled data from the LLM.

**DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents** [V] `li2025drift`
- Li, Liu, Chiu, Li, Zhang, Xiao. NeurIPS 2025 (poster). arXiv 2506.12104.
- Summary: Combines a secure planner, a dynamic validator that updates control/data constraints, and an injection isolator that scrubs conflicting instructions from memory.
- Why it matters: an internal three-component stack, though its components are not separately priced against adaptive attackers.

**System-Level Defense against Indirect Prompt Injection Attacks: An Information Flow Control Perspective (f-secure)** [V] `wu2024fsecure`
- Wu, Cecchetti, Xiao. arXiv 2409.19091, Sep 2024.
- Summary: An IFC-based architecture that keeps untrusted inputs out of the planning step via a context-aware pipeline and a security monitor.

**Defending Against Indirect Prompt Injection Attacks With Spotlighting** [V] `hines2024spotlighting`
- Hines, Lopez, Hall, Zarfati, Zunger, Kiciman (Microsoft). arXiv 2403.14720, Mar 2024.
- Summary: Prompt transformations (delimiting, datamarking, encoding) that mark untrusted text so the model can tell data from instructions. Near-zero cost.
- Why it matters: the cheapest layer in our pool. Expect high MBR under static attacks and low MBR under adaptive ones.

**The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents** [V] `jia2025taskshield`
- Jia, Wu, Qin, Squicciarini. ACL 2025 (long), pp. 29680–29697. DOI 10.18653/v1/2025.acl-long.1435.
- Summary: At each step, checks whether an instruction or tool call serves the user's stated goals, and blocks it if not.
- Why it matters: the same job as AlignmentCheck, which makes it a direct H3 same-type pair.

**Design Patterns for Securing LLM Agents against Prompt Injections** [V] `beurerkellner2025patterns`
- Beurer-Kellner, Buesser, Creţu, Debenedetti, Dobos, Fabian, Fischer, Froelicher, Grosse, Naeff, Ozoani, Paverd, Tramèr, Volhejn. arXiv 2506.08837, Jun 2025.
- Summary: Catalogs architectural patterns (plan-then-execute, dual LLM, action selector, etc.) and discusses their utility/security trade-offs through case studies.

**Formalizing and Benchmarking Prompt Injection Attacks and Defenses** [V] `liu2024formalizing`
- Liu, Jia, Geng, Jia, Gong. USENIX Security 2024. arXiv 2310.12815.
- Summary: A formal framework for prompt injection and a benchmark of attacks and defenses (Open-Prompt-Injection) across tasks and LLMs.

**Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** [V] `greshake2023indirect`
- Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz. arXiv 2302.12173, Feb 2023. (The AISec 2023 venue is commonly cited, not confirmed here.)
- Summary: The paper that introduced indirect prompt injection, with a taxonomy and real-application demonstrations.

**A Critical Evaluation of Defenses against Prompt Injection Attacks** [V] `jia2025critical`
- Jia, Shao, Liu, Jia, Song, Gong. arXiv 2505.18333, May 2025.
- Summary: Proposes evaluating defenses on both effectiveness against adaptive attacks and general utility, and finds existing defenses weaker than reported.

**Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks?** [V] `bhagwatkar2025firewalls`
- Bhagwatkar, Kasa, Puri, Huang, Rish, Taylor, Dvijotham, Lacoste. arXiv 2510.05244, Oct 2025.
- Summary: LLM "firewalls" at the tool boundary: a Minimizer on tool inputs and a Sanitizer on tool outputs. On AgentDojo with GPT-4o the Sanitizer reports about 0% ASR, and the combined pair reports 0% ASR but lower benign utility. The authors document AgentDojo/ASB/InjecAgent scoring flaws and release fixes, and show a Braille/obfuscation bypass.
- Why it matters: (i) a **two-layer stack on AgentDojo**, without adaptive evaluation or MBR; (ii) we must check whether **AgentDojo v1.2.2 contains their utility-check fixes**, or our utility numbers inherit the bugs.

**Lessons from Defending Gemini Against Indirect Prompt Injections** [V] `shi2025gemini`
- Shi, Lin, Song, Hayes, Shumailov, Yona, Pluto, Pappu, Choquette-Choo, Nasr, Sitawarin, Gibson, Terzis, Flynn (Google DeepMind). arXiv 2505.14534, May 2025.
- Summary: An industrial adaptive-evaluation framework (actor-critic, beam search, TAP, linear generation) for agentic Gemini. Defenses that look strong on static sets were often broken by small adaptations. Adversarial training helps, and the authors argue robustness needs defense-in-depth.
- Why it matters: strong support for H1, from a proprietary model and defenses. Per-layer marginal value is not reported in the portion read (the section on combined results was not visible).

**Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks** [V] `ji2025sokipi`
- Ji, Wang, Li, Ma, Gao, Wu, Yan, Tian, Wang. arXiv 2511.15203, Nov 2025 (SoK).
- Summary: Evaluates nine agent defenses (LlamaFirewall, Tool Filter, Meta SecAlign, IsolateGPT, ACE, CaMeL, Task Shield, MELON, Progent) on AgentDojo/ASB/InjecAgent for ASR, utility and overhead (time, tokens), and designs three adaptive attacks (semantic masquerading, cascading, isolation breach) that raise ASR 2–4×.
- Why it matters: a **medium threat**. Same benchmark and metrics, but every defense is evaluated **in isolation**. There is no MBR, redundancy, or ordering.

**The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis** [V] `wang2026landscape`
- Wang, Li, Xiang, Zhang, Li, Zhang, Wang, Tian. arXiv 2602.10453, Feb 2026 (SoK).
- Summary: A taxonomy plus the AgentPI benchmark for context-dependent tasks. It finds that no single defense is simultaneously trustworthy, useful and low-latency, and that some defenses look good only by suppressing context.
- Why it matters: it frames the three-way trade-off qualitatively. We quantify it per layer and per stack.

**Architecting Secure AI Agents: Perspectives on System-Level Defenses Against Indirect Prompt Injection Attacks** [V] `xiang2026architecting`
- Xiang, Zagieboylo, Ghosh, Kariyappa, Greshake, Xiao, Xiao, Suh (NVIDIA/JHU). arXiv 2603.30016, Mar 2026. Position paper, no experiments.
- Summary: Argues for orchestrator/approver/executor/enforcer architectures and criticizes AgentDojo for few replanning tasks and static attacks.
- Why it matters: supports our motivation (static benchmarks mislead).

**Note on "AgentShield"**: I found **no tool-gate paper named AgentShield**. One paper titled *AgentShield: Deception-based Compromise Detection for Tool-using LLM Agents* exists (via a third-party review page; arXiv ID and authors not verified), but it uses decoy-tool tripwires, not gating. For tool gates, cite **Progent**, **Tool Filter** (from AgentDojo), and **CaMeL**. [U]

---

## (c) Adaptive attacks and multi-turn attacks

**Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents** [V] `zhan2025adaptive`
- Zhan, Fang, Panchal, Kang. Findings of NAACL 2025, pp. 7116–7132. DOI 10.18653/v1/2025.findings-naacl.395. arXiv 2503.00061.
- Summary: Breaks eight IPI defenses (detection, input-level, model-level) on InjecAgent with GCG-style adaptive attacks, including multi-objective variants for detectors, with ASR above 50% for every defense. It also tests transfer across defenses.
- Why it matters: the methodological template for our per-layer adaptive attacker. The multi-objective GCG against a detector plus the agent is the minimal "attack the stack" baseline.

**The Attacker Moves Second: Stronger Adaptive Attacks Bypass Defenses Against LLM Jailbreaks and Prompt Injections** [V] `nasr2025attacker`
- Nasr, Carlini, Sitawarin, Schulhoff, Hayes, Ilie, Pluto, Song, Chaudhari, Shumailov, Thakurta, Xiao, Terzis, Tramèr. USENIX Security 2026 (presentation page seen). arXiv 2510.09023, Oct 2025.
- Summary: Bypasses 12 recent jailbreak/PI defenses, most at over 90% ASR, using gradient, RL, search-based and human adaptive attacks, though the defenses originally reported near-zero ASR.
- Why it matters: the canonical H1 citation. Its gap: defenses are attacked **one at a time**, not as composed stacks.

**On Adaptive Attacks to Adversarial Example Defenses** [V] `tramer2020adaptive`
- Tramèr, Carlini, Brendel, Madry. NeurIPS 2020 (vol. 33).
- Summary: Breaks 13 published defenses and distills guidance for designing defense-specific adaptive attacks.

**Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods** [V] `carlini2017detected`
- Carlini, Wagner. ACM AISec 2017 (per the author's PDF naming). arXiv 1705.07263.
- Summary: Shows ten adversarial-example detectors can be evaded once the attacker includes the detector in the loss.
- Why it matters: the direct ancestor of attacking a detector layer together with the model.

**Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples** [V] `athalye2018obfuscated`
- Athalye, Carlini, Wagner. ICML 2018. arXiv 1802.00420.

**On Evaluating Adversarial Robustness** [V] `carlini2019evaluating`
- Carlini, Athalye, Papernot, Brendel, Rauber, Tsipras, Goodfellow, Madry, Kurakin. arXiv 1902.06705, Feb 2019.
- Summary (for the two above): methodology for adaptive evaluation, including how to avoid gradient masking and how to report robustness honestly.

**Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks** [V] `andriushchenko2025simple`
- Andriushchenko, Croce, Flammarion. ICLR 2025 (mlanthology). arXiv 2404.02151.
- Summary: Random-search suffixes plus prompt templates, using logprob access, achieve near-100% jailbreak rates on many models.
- Why it matters: a cheap black-box adaptive baseline for each layer.

**STACK: Adversarial Attacks on LLM Safeguard Pipelines** [V] `mckenzie2025stack`
- McKenzie, Hollinsworth, Tseng, Davies, Casper, Tucker, Kirk, Gleave (FAR.AI/UK AISI/Oxford). arXiv 2506.24068, Jun 2025. A version appears in AAAI proceedings (OJS article 41108; year and pages not verified).
- Summary: Builds an open-weight defense-in-depth pipeline (input classifier, generator, output classifier, with ShieldGemma, Llama Guard 3/4, WildGuard and few-shot Gemma-2/Qwen3 classifiers) and a **staged** attack that jailbreaks each component separately and then composes the pieces. It reports 71% ASR (ClearHarm) with black-box feedback about which stage blocked, and 33% via transfer from a proxy pipeline. It recommends hiding which stage refused.
- Why it matters: a **high novelty threat** for "adaptive attacks against stacks", in chat/CBRN rather than agents. There is no systematic per-layer marginal value, no redundancy, no defender cost, and no utility frontier. **We should adopt STACK as our stack-adaptive attacker** (and note that its proxy uses Qwen3, which is banned on our cluster, so we need a substitute proxy).

**Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks** [V] `isbarov2026proxy`
- Isbarov, Kantarcioglu (Virginia Tech). arXiv 2602.05066, Feb 2026.
- Summary: An injection makes the agent itself repeat an optimized string in its reasoning or tool arguments, which then fools the monitor. Parallel-GCG optimizes one string for several insertion points. On AgentDojo it bypasses AlignmentCheck (>90% ASR with some monitors) and a **PromptGuard 2 + AlignmentCheck stack** (multi-objective GCG).
- Why it matters: a **medium-high threat**: an adaptive attack on exactly our LlamaFirewall stack in exactly our benchmark. It covers two layers, is white-box, and does not report utility, cost, MBR or redundancy.

**Learning to Inject: Automated Prompt Injection via Reinforcement Learning (AutoInject)** [V] `chen2026autoinject`
- Chen, Zhang, Tramèr (ETH). arXiv 2602.05746, 2026 (exact date not shown on the page read).
- Summary: A GRPO-trained 1.5B attacker policy with dense comparative rewards that generates transferable injection suffixes. It reaches 58% ASR on Gemini-2.5-flash in AgentDojo and about 22% on Meta-SecAlign-70B.
- Why it matters: a strong black-box adaptive attacker for AgentDojo. **Caveat:** the policy is Qwen2.5-1.5B, which is banned on our cluster, so it would need re-training on an allowed base.

**Prompt Overflow: What the Guardrail Inspects Is Not What the Model Infers** [V] `zhou2026overflow`
- Zhou, Zhu, Wang, He, Zhai, Sun, Wei, Xiong. arXiv 2605.23196, May 2026.
- Summary: Exploits the gap between a guardrail's short inspection window (about 512 tokens, with per-segment max aggregation) and the LLM's long context. Interleaving malicious fragments with benign filler bypasses PromptGuard 2 (over 99%), DeBERTa detectors, and Granite Guardian.
- Why it matters: a concrete **structural common-mode failure** shared by window-limited classifiers. It predicts high ρ between short-context layers under this attack (H3), regardless of training data.

**PRP: Propagating Universal Perturbations to Attack Large Language Model Guard-Rails** [P] (no bib)
- ACL 2024 (anthology 2024.acl-long.591). The authors appear to include Mangaokar, Hooda, Choi, Fawaz, Prakash and others, but the full list was **not verified**. It attacks guard+model pairs with universal perturbations.

### Multi-turn attacks
**Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack** [V] `russinovich2025crescendo` — Russinovich, Salem, Eldan. USENIX Security 2025. arXiv 2404.01833. Gradual benign-to-harmful escalation that references the model's own earlier outputs.

**X-Teaming: Multi-Turn Jailbreaks and Defenses with Adaptive Multi-Agents** [V] `rahman2025xteaming` — Rahman, Jiang, Shiffer, Liu, Issaka, Parvez, Palangi, Chang, Choi, Gabriel. arXiv 2504.13203, Apr 2025 (an icml.cc 2025 listing was seen; track not verified). A planner/attacker/verifier/optimizer multi-agent attack with very high multi-turn ASR, plus a released defense dataset.

**ActorAttack** (arXiv 2410.10700; now *LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts*) [V] `ren2024actorattack` — Ren, Li, Liu, Xie, Lu, Qiao, Sha, Yan, Ma, Shao. Oct 2024. Builds a network of semantically related "actors" to reach harmful goals through innocuous multi-turn queries.

**Foot-In-The-Door: A Multi-turn Jailbreak for LLMs** [V] `weng2025fitd` — Weng, Jin, Jia, Zhang. arXiv 2502.19820, Feb 2025. Escalates through small consistent commitments.

**Automated Red Teaming with GOAT: the Generative Offensive Agent Tester** [V] `pavlova2025goat` — Pavlova, Brinkman, Iyer, Albiero, Bitton, Nguyen, Canton Ferrer, Evtimov, Grattafiori (Meta). ICML 2025, PMLR 267:48470–48487. An attacker LLM that chains jailbreak techniques over turns.

**Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models (RACE)** [V] `ying2025race` — Ying, Zhang, Jing, Xiao, Zou, Liu, Liang, Zhang, Liu, Tao. arXiv 2502.11054, Feb 2025. Recasts harmful requests as benign reasoning tasks advanced over turns.

**Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search** [V] `zhou2025siege` — Zhou, Arel. arXiv 2503.10619, Mar 2025. ICLR 2025 BuildingTrust workshop. A breadth-first tree search over conversation branches that exploits partial compliance.

**LLM Defenses Are Not Robust to Multi-Turn Human Jailbreaks Yet (MHJ)** [V] `li2024mhj` — Li, Han, Steneker, Primack, Goodside, Zhang, Wang, Menghini, Yue (Scale AI). arXiv 2408.15221, Aug 2024. Human multi-turn red-teaming breaks defenses that resist automated single-turn attacks. Releases the MHJ dataset.

**Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security** [V] `jain2026adaptive` — Jain, Hartmann, Li. arXiv 2607.18063 (v2 Sep 2026; NeurIPS 2026 poster). An autonomous attacker adapts over up to 15 rounds. ASR rises from 0–1% (first round) to 7.9–16.8% (15 rounds), and pooling attackers finds 1.7–2.2× more unique successes. The "defenders" are LLMs, not guardrail stacks.
- Why multi-turn matters for us: turn budget is a first-class variable. MBR must be computed per turn **and** cumulatively per conversation.

---

## (d) Agent-security and safety benchmarks

**AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents** [V] `debenedetti2024agentdojo` — Debenedetti, Zhang, Balunović, Beurer-Kellner, Fischer, Tramèr. NeurIPS 2024 Datasets & Benchmarks. arXiv 2406.13352. Stateful suites (workspace, travel, banking, slack) with user tasks, injection tasks, and baseline defenses (PI detector, tool filter, spotlighting, sandwiching). **Our testbed.**

**InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents** [V] `zhan2024injecagent` — Zhan, Liang, Ying, Kang. Findings of ACL 2024, pp. 10471–10506. DOI 10.18653/v1/2024.findings-acl.624. A static single-step IPI benchmark with direct-harm and exfiltration cases.

**Agent Security Bench (ASB)** [V] `zhang2025asb` — Zhang, Huang, Mei, Yao, Wang, Zhan, Wang, Zhang. ICLR 2025. arXiv 2410.02644. Many scenarios, tools and attack types (DPI, IPI, memory poisoning, backdoor) with baseline defenses.

**AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents** [V] `andriushchenko2025agentharm` — Andriushchenko, Souly, Dziemian, Duenas, Lin, Wang, Hendrycks, Zou, Kolter, Fredrikson, Winsor, Wynne, Gal, Davies. ICLR 2025. arXiv 2410.09024. Explicitly malicious multi-step agent tasks. A candidate source for the "harmful output / unsafe tool call" categories.

**WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks** [V] `evtimov2025wasp` — Evtimov, Zharmagambetov, Grattafiori, Guo, Chaudhuri (Meta). NeurIPS 2025 D&B. arXiv 2504.18575. Realistic web-agent injections. It distinguishes partial hijacking from full attacker-goal completion.

**Agent-SafetyBench: Evaluating the Safety of LLM Agents** [V] `zhang2024agentsafetybench` — Zhang, Cui, Lu, Zhou, Yang, Wang, Huang. arXiv 2412.14470, Dec 2024. Many interactive environments and risk categories. Finds no tested agent scores high on safety.

**Identifying the Risks of LM Agents with an LM-Emulated Sandbox (ToolEmu)** [V] `ruan2024toolemu` — Ruan, Dong, Wang, Pitis, Zhou, Ba, Dubois, Maddison, Hashimoto. ICLR 2024. LLM-emulated tools plus an LLM safety evaluator.

**R-Judge: Benchmarking Safety Risk Awareness for LLM Agents** [V] `yuan2024rjudge` — Yuan, He, Dong, Wang, Zhao, Xia, Xu, Zhou, Li, Zhang, Wang, Liu. arXiv 2401.10019; Findings of EMNLP 2024 (anthology 2024.findings-emnlp.79 seen; pages not verified). Tests whether LLMs can judge agent trajectories as risky. Relevant to AlignmentCheck-type layers.

**The bitter lesson of misuse detection (BELLS)** [V] `mariaccia2025bells` — Mariaccia, Segerie, Dorn (CeSIA). arXiv 2507.06282, Jul 2025. Introduces the BELLS benchmark and finds that specialized guardrails underperform general LLMs asked directly about harm.

**GuardBench: A Large-Scale Benchmark for Guardrail Models** [V] `bassani2024guardbench` — Bassani, Sanchez. EMNLP 2024, pp. 18393–18409. DOI 10.18653/v1/2024.emnlp-main.1022. Many datasets unified for guard-model evaluation.

**JailbreakBench** [V] `chao2024jailbreakbench` — Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramèr, Hassani, Wong. NeurIPS 2024 D&B. arXiv 2404.01318. 100 harmful behaviors with matched benign prompts, a jailbreak-artifact repository, and a standardized judge and leaderboard. Alotaibi et al. use its behaviors, so it is a bridge for comparing our chat-level sanity checks with theirs.

**HarmBench** [V] `mazeika2024harmbench` — Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks. ICML 2024, PMLR 235:35181–35224. A standardized red-teaming evaluation with a fixed behavior set, a trained harm classifier, and a large comparison of attacks against models. Provides direct-jailbreak seeds and a harm judge we can validate against.

**OR-Bench: An Over-Refusal Benchmark for Large Language Models** [V] `cui2025orbench` — Cui, Chiang, Stoica, Hsieh. ICML 2025, PMLR 267:11515–11542. A large set of "seemingly toxic" but benign prompts, plus a hard subset, for measuring over-refusal at scale. Gives a per-layer FPR that is independent of AgentDojo utility.

**XSTest** [V] `rottger2024xstest` — Röttger, Kirk, Vidgen, Attanasio, Bianchi, Hovy. NAACL 2024, pp. 5377–5400. DOI 10.18653/v1/2024.naacl-long.301. A small hand-built suite of safe prompts that superficially resemble unsafe ones, with unsafe contrasts. The standard quick test for exaggerated safety.
- Why these four matter: standard harmful and over-refusal sets for measuring each layer's FPR independently of AgentDojo utility.

**GuardianAgentBench: Where Agents Fail and How to Guard Them** [V] `naik2026gabench` — Naik, Xu, Dong, Hassan, Pradhan, Mendelevitch, Shafaat, Irshad. arXiv 2607.20982, Jul 2026 (ICML 2026 Workshop on Failure Modes in Agentic AI). Its three guardrails are evaluated only as one combined unit, with no per-guard contribution or cost. Low threat.

---

## (e) Measurement validity — seed IDs from the slide deck (all checked)

| Claimed | Exists? | Title found | Matches claim? |
|---|---|---|---|
| Chouldechova et al., arXiv 2601.18076, NeurIPS 2025 position | **Yes** [V] | *Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming* — Chouldechova, Cooper, Barocas, Palia, Vann, Wallach | **Yes.** NeurIPS 2025 Position Paper Track confirmed in the proceedings. Note: arXiv v1 was posted 26 Jan 2026, after the conference. |
| MT-JailBench, arXiv 2605.11002 | **Yes** [V] | *MT-JailBench: A Modular Benchmark for Understanding Multi-Turn Jailbreak Attacks* — Zhang, Wei, Gong, Zheng, Zhang, Dong, Erichson | **Yes.** Also listed as a NeurIPS 2026 poster (neurips.cc/virtual/2026/poster/139838). |
| REDAgentBench, arXiv 2608.10669 | **Yes** [V via alphaXiv mirror] | *REDAgentBench: Executable Red Teaming and Faithful Measurement of LLM Agent Systems* — Chen, Liu, Zhu, Dou, Jiang, Li, Guo, Chen, Zhang; 11 Aug 2026 | **Yes** on ID and name. The full title emphasizes "faithful measurement". The arXiv abs page itself was not fetched (allowlist). |
| BELLS-O, arXiv 2606.20668 | **Yes** [V] | *BELLS-O: Evaluating the Operational Trade-offs of LLM Supervision Systems* — Waibl, Michalak, Mariaccia | **Yes.** Venue per arXiv: ICML 2026 Workshop (AI4GOOD). Also listed on the NeurIPS 2026 virtual site. If the slides call it a main-track paper, that is **not supported**. |
| "A Coin Flip for Safety", arXiv 2603.06594 | **Yes** [V via arXiv HTML v2] | *A Coin Flip for Safety: LLM Judges Fail to Reliably Measure Adversarial Robustness* — Schwinn, Ladenburger, Beyer, Mofakhami, Gidel, Günnemann | **Yes.** ICML 2026 poster confirmed on icml.cc. |

All five seed IDs resolve to the claimed papers, and no ID mismatches were found. I could not see the slide deck's specific claims about each paper, so only the identity, title and venue were checked.

**Comparison requires valid measurement: Rethinking attack success rate comparisons in AI red teaming** [V] `chouldechova2025comparison`
- Summary: Uses measurement theory and inferential statistics to state when two ASR numbers are comparable at all. Differences in threat model, judge, sampling and budget often make comparisons meaningless.
- Why it matters: every MBR/ASR comparison across layers in our paper must hold the attacker, budget, judge and task set fixed. Cite this when defining MBR.

**MT-JailBench** [V] `zhang2026mtjailbench`
- Summary: Breaks multi-turn attacks into five modules and finds that turn/retry budgets and judge choice substantially change measured ASR. Prompt generation drives most differences between attacks. A recombined attack (CrescendoX) beats its sources and stays effective against the defenses in their appendix.
- Why it matters: fix and report turn/retry budgets. CrescendoX is a candidate multi-turn adaptive attacker.

**REDAgentBench** [V] `chen2026redagentbench`
- Summary: 1,661 executable sandboxed cases, with violations verified by service logs and final state. ASR shifts with the harness and with what evidence the evaluator sees. About 1 in 5 violations happen after the agent itself had flagged the risk (a "recognition–execution gap"). A tool-time policy reminder cuts confirmed violations sharply.
- Why it matters: use **state-based** success checks (AgentDojo does this) rather than transcript judges where possible. The recognition–execution gap motivates execution-time gates over input classifiers.

**BELLS-O** [V] `waibl2026bellso`
- Summary: Detection rate, FPR, latency and cost for 28 supervisors from 17 providers (for example LlamaGuard-4, ShieldGemma-2, Lakera, frontier LLMs as judges). Specialized guards roughly match frontier LLMs on content moderation at 5–10× lower latency and about 10× lower cost. On jailbreak detection, frontier LLMs win but cost far more. Combinations, adaptive attacks, agents and multi-turn are **explicitly out of scope**.
- Why it matters: the best prior **cost/latency methodology** (batch size 1, single node, GPU-hour pricing) to align with. It leaves our whole delta open.

**A Coin Flip for Safety** [V] `schwinn2026coinflip`
- Summary: Using 6,642 human-verified labels, LLM safety judges degrade toward chance under shifts in attack, victim model and semantic category, inflating ASR. **Ensembling judges does not help, because they share failure modes**, and high inter-judge agreement does not imply correctness.
- Why it matters: (i) a threat to the validity of our harmful-output labels, so we need human-validated subsets and precision-corrected ASR; (ii) independent evidence for H3 (same-type LLM classifiers fail together), in the judge setting.

---

## (f) Theory we may borrow

**Adversarial Example Defenses: Ensembles of Weak Defenses are not Strong** [V] `he2017ensembles` — He, Wei, Chen, Carlini, Song. USENIX WOOT 2017. arXiv 1706.04701. Combining several weak detectors and defenses does not yield a robust one under an adaptive attacker who targets all of them. **The single most important prior for H2/H3**, from the pre-LLM era.

**An Experimental Evaluation of the Assumption of Independence in Multiversion Programming** [V] `knight1986experimental` — Knight, Leveson. IEEE TSE SE-12(1):96–109, 1986. DOI 10.1109/TSE.1986.6312924. Independently developed versions failed together far more often than independence predicts.

**A Theoretical Basis for the Analysis of Multiversion Software Subject to Coincident Errors** [V] `eckhardt1985theoretical` — Eckhardt, Lee. IEEE TSE SE-11(12):1511–1517, 1985. DOI 10.1109/TSE.1985.231895. Shows that variation in input "difficulty" alone induces positive failure correlation even between independently built versions. This is the formal model behind ρ(i,j) > 0.

**Conceptual Modeling of Coincident Failures in Multiversion Software** [V] `littlewood1989conceptual` — Littlewood, Miller. IEEE TSE 15(12):1596–1614, 1989. DOI 10.1109/32.58771. Extends Eckhardt–Lee to forced methodological diversity, which can reduce or even reverse coincident failure. This motivates choosing **heterogeneous** layer types.

**Human error: models and management** [V] `reason2000human` — Reason. BMJ 320(7237):768–770, 2000. DOI 10.1136/bmj.320.7237.768. The "Swiss cheese" model: harm occurs when holes in several defensive layers line up. (Reason's 1990 book *Human Error* is the earlier source, not verified here.)

**Swiss Cheese Model for AI Safety: A Taxonomy and Reference Architecture for Multi-Layered Guardrails of Foundation Model Based Agents** [V] `shamsujjoha2025swiss` — Shamsujjoha, Lu, Zhao, Zhu (CSIRO Data61). ICSA 2025 (research track listing seen). arXiv 2408.02205. A taxonomy and reference architecture only; it explicitly leaves operational validation out of scope.

**Optimal task sequencing with precedence constraints** [V-partial] `garey1973optimal` — Garey. *Discrete Mathematics*, 1973. DOI 10.1016/0012-365X(73)90113-1 (volume and pages not verified). The classic sequencing result for ordering independent tests by cost/(1−pass probability).

**Adaptive Ordering of Pipelined Stream Filters** [V] `babu2004adaptive` — Babu, Motwani, Munagala, Nishizawa, Widom. SIGMOD 2004, pp. 407–418. DOI 10.1145/1007568.1007615. The A-Greedy algorithm orders **correlated** filters using conditional selectivities, with a 4-approximation. Its conditional selectivity is exactly 1 − MBR(k), so this gives our ordering algorithm.

**The Pipelined Set Cover Problem** [V] `munagala2005pipelined` — Munagala, Babu, Motwani, Widom. ICDT 2005, LNCS 3363, pp. 83–98. DOI 10.1007/978-3-540-30570-5_6. Formalizes min-cost ordering of correlated filters as pipelined set cover, with greedy 4-approximation and hardness results.

**Algorithms for distributional and adversarial pipelined filter ordering problems** [V] `condon2009algorithms` — Condon, Deshpande, Hellerstein, Wu. ACM Trans. Algorithms 5(2), 2009. DOI 10.1145/1497290.1497300. Filter ordering when inputs are chosen **adversarially**, which is the closest theory to an adaptive attacker who moves second.

**An analysis of approximations for maximizing submodular set functions—I** [V] `nemhauser1978analysis` — Nemhauser, Wolsey, Fisher. Math. Programming 14:265–294, 1978. DOI 10.1007/BF01588971. Greedy gives a (1−1/e) approximation for monotone submodular maximization under a cardinality constraint. The "instances blocked by a layer set" coverage function is monotone submodular, so the bound applies to layer selection over a **fixed** attack set.

**The Hungarian method for the assignment problem** [V] `kuhn1955hungarian` — Kuhn. Naval Research Logistics Quarterly 2(1–2):83–97, 1955. DOI 10.1002/nav.3800020109.
- **Assessment of the collaborator's suggestion:** the Hungarian method solves a **one-to-one** layer↔attack assignment. Stacking is a **many-to-many coverage** problem (one layer covers several attack types, and several layers may be needed for one), with costs and correlated failures. Assignment is the wrong model except for the narrow question "which single layer is responsible for each attack type?", and that question is better answered descriptively from the MBR matrix. I recommend submodular/set-cover selection plus pipelined-filter ordering. Also, all of these optimize against a **fixed** attack distribution: once the stack is chosen, an adaptive attacker re-optimizes against it. Report the selected stack's ASR **after** re-attack (Condon et al. give the adversarial-ordering angle).

**AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures?** [V] `dung2025alignment` — Dung, Mai. arXiv 2510.11235, Oct 2025 (also an AAAI IASEAI OJS listing). A purely conceptual mapping of seven alignment techniques to seven failure modes. It argues that defense-in-depth requires uncorrelated failures and that RLHF-family methods share most failure modes. It makes the same conceptual argument as H3, at the training-technique level, with no measurement.

---

## (g) Cost/latency, over-refusal, and composed or stacked guardrails (incl. novelty-scan hits)

**Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers** [V] `alotaibi2026layered` — Alotaibi, Jabbar, Al-Azani, Ahmed (KFUPM/IAU/SDAIA). arXiv 2608.28327 (listed 31 Aug 2026).
- Summary: A seven-member defense stack (one per "dependency row", e.g., perplexity filter, token-anomaly filter, semantic classifier, mid-layer probe) on Vicuna-7B (replicated on Llama-2-7B), JailbreakBench behaviors, under a shared adaptive adversary (GCG-style suffix plus fluent search). They measure pairwise φ correlation of breach indicators: **all 15 pairs are positively correlated (φ 0.30–0.75), with joint residuals above the independence prediction**. Stratifying on difficulty removes most of the association, so the dependence is mostly common cause. The full stack refuses about 4/5 benign prompts yet is statistically indistinguishable from its strongest member. Cost is analyzed with FLOP-class arguments, not measured. Composition rules (cover tiers, engineer independence, order by cost class) are stated, not validated. The authors flag that their correlation assumes parallel evaluation while real stacks are sequential.
- Why it matters: **the #1 novelty threat**, covering H3 and part of H2 in chat. It leaves open: agents and IPI, deployed guardrail products, **sequential conditional MBR**, measured latency/cost, multi-turn, task-utility frontiers, and ordering optimization.

**Defenses at Odds: Measuring and Explaining Defense Conflicts in Large Language Models** [P] `meng2026odds` — Meng, Chen, Zang, Gao, Wang, Wang, Li, Guo. arXiv 2605.14514 (alphaXiv shows "Submitted 30 Sep 2026", which may be a v2 date). CONFLICTEVAL tests 144 **ordered compositions** of six (mostly training-time) safety/privacy/fairness defenses on six models. Interactions depend on pair, order and model, and later defenses can undo earlier protection. A low-medium threat for "order matters". Model-level, not inference-time guardrails, and no attacks.

**One Risk Down, Another Up: Cross-Risk Interactions Induced by LLM Defenses** [V] `meng2025crossrisk` — Meng, Cong, Wang, Chen, Li, Guo, Wang. arXiv 2510.07968, Oct 2025. One defense often amplifies other risks (77% of 166 evaluations show significant interactions). Single defenses, not stacks.

**OpenClaw PRISM: A Zero-Fork, Defense-in-Depth Runtime Security Layer for Tool-Augmented LLM Agents** [V] `li2026prism` — Li (UNSW). arXiv 2603.11853, Mar 2026. Ten lifecycle hooks with a heuristic plus LLM scanner cascade. An incremental "baseline ladder" (block rate 0 → 0.41 → 0.46 → 0.55 → 0.96) with p95 latency up to about 16 s from the local-LLM scanner. A hand-built 110-case corpus, a pilot, no adaptive attacks, no per-layer ablation. Low threat, but the ladder is a primitive cumulative-MBR curve.

**Who Guards the Guard? Evaluating Deterministic, LLM-Based, and Hybrid Firewalls for Tool-Using Agents** [V] `quincozes2026guard` — Quincozes, Souza, Araujo, Molinos, Quincozes. SBSeg 2026 (undergraduate/IC workshop), pp. 774–784. DOI 10.5753/sbseg_estendido.2026.29812. On held-out attacks, the hybrid's router **never invoked the semantic guard** (0/744), so the hybrid equaled the deterministic layer (54.8% ASR). This is an anecdotal **MBR = 0** case that supports H2. Low threat.

**Defense-in-Depth for Generative AI: An Architectural Evaluation and Comparison of LLM Guardrails** [V] `lima2026did` — Lima, Piantoni, Zambelli, Martinez Silva, Braga (CPQD). SBSeg 2026 industry track, pp. 901–908. DOI 10.5753/sbseg_estendido.2026.29281. A qualitative architectural comparison of Prompt Guard, Llama Guard, LlamaFirewall, NeMo and Bedrock, with no measurements. Low threat.

**Quantifying Return on Security Controls in LLM Systems** [V] `moulton2025roc` — Moulton, O'Brien, Hastings. arXiv 2512.15081, Dec 2025. Monte-Carlo expected-loss "return on control" for ABAC, Presidio redaction, and a NeMo output rail on a small RAG app with Garak probes. The NeMo rail barely changed expected loss. Controls are evaluated singly, despite "layered" framing. Low threat, but useful as a "security economics" citation.

**Governance Architecture for Autonomous Agent Systems (LGA)** [V] `ge2026lga` — Ge (York). arXiv 2603.07191, Mar 2026. A four-layer governance architecture. Mainly evaluates the LLM-judge layer and two-stage judge cascades. One end-to-end run (96% interception, about 980 ms p50). No per-layer ablation or correlation. Low threat.

**No Free Lunch With Guardrails** [V] `kumar2025nofreelunch` — authors listed on arXiv as "Kuamr" (sic; likely Kumar), Birur, Baswa, Agarwal, Harshangi (Enkrypt AI). arXiv 2504.00441, Apr 2025. A safety/utility/latency trade-off across commercial APIs, open classifiers and LLM judges, each evaluated individually. Note the conflict of interest: the authors' company sells one of the guardrails evaluated.

**Adversarial Prompt Evaluation: Systematic Benchmarking of Guardrails Against Prompt Input Attacks on LLMs** [V] `zizzo2025ape` — Zizzo, Cornacchia, Fraser, Hameed, Rawat, Buesser, Purcell, Chen, Sattigeri, Varshney (IBM). arXiv 2502.15427 (NeurIPS 2024 SafeGenAI workshop). Fifteen defenses, each individually, with per-sample latency/memory (e.g., Llama-Guard-2 about 0.14 s, SmoothLLM about 18 s on V100). Lightweight classifiers have the best cost-performance.

**Calibrated Deferral Routing for Cost-Efficient Guardrails** [V] `cornacchia2026deferral` — Cornacchia, Padhi, Nagireddy, Chaudhury, Dognin, Pedapati, Rawat, Purcell, Varshney, Sattigeri. ICML 2026 (icml.cc listing). A small-guard-first cascade that defers only uncertain cases to a large guard after calibration. A 2B→8B cascade comes within 1 point of 8B accuracy at 55% lower cost. Non-adversarial, chat.

**SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models** [P] (no bib) — arXiv 2502.12464. A Findings-ACL-2025 preview page was seen. Authors not verified (Haebin Seong appears to be an author). A router chooses a small or large guard per input.

**Evaluating the Robustness of Large Language Model Safety Guardrails Against Adversarial Attacks** [V] `young2025robustness` — Young (UNLV). arXiv 2511.22047, Nov 2025. Ten guard models on 1,445 single-turn prompts. 291 prompts fail on ≥7 of 10 models ("universally difficult"), which is informal evidence of correlated failure. There is no formal correlation analysis, and the paper has some internal inconsistencies.

**GuardNet: Ensemble Strategies of Shallow Neural Networks for Robust Prompt Injection and Jailbreak Detection** [V] `neves2026guardnet` — Neves et al. (10 authors). arXiv 2606.05566, 2026. Mean-probability ensemble of three BiLSTM heads. It does not measure error correlation or adaptive robustness, and the evaluation is small. Low threat.

**A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks** [V] `hossain2025multiagent` — Hossain, Shayoni, Ameen, Islam, Mridha, Shin. arXiv 2509.14285, Sep 2025. Coordinator and guard-agent pipelines report 0% ASR on 55 attacks, with internal inconsistencies, no adaptive attacks and no cost data. Low threat.

**Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline** [V] `rao2025multistage` — Rao, Singh, Saksena, Kumar (BITS Pilani). arXiv 2512.19011, Dec 2025. A TF-IDF+SVM stage plus regex/YARA/vector-DB/ShieldGemma stages. Argues for cheap-first ordering but does not test alternative orderings. Small, inconsistent samples. Low threat.

**Prompt Injection Detection is Regime-Dependent** [V] `akinrele2026regime` — Akinrele, Gowda. arXiv 2605.26999, May 2026. No detector wins across deployment regimes, and AUC overstates performance at strict FPR. Late-fusion hybrids give small, selective gains. Supports reporting TPR at fixed FPR.

**Toward Secure LLM Agents: Threat Surfaces, Attacks, Defenses, and Evaluation** [V] `ling2026survey` — Ling, Yu, Chen, Fang. arXiv 2606.10749, Jun 2026. A survey of 247 papers. It concludes that agent defenses are only weakly compositional, that the field has no stable compositional logic, and that utility, latency and cost are rarely measured. **This is a useful quotable gap statement** for our introduction.

**Injection-Execution Dissociation: A Mechanistic Evaluation of Persistent Memory Attacks and Defenses in Stateful LLM Agents** [P] (no bib) — Leong. arXiv 2605.08442 (v5 Aug 2026). Six defenses at four architectural layers, each evaluated individually. Finds that **position relative to the authority boundary**, not classifier quality, determines effectiveness. The BibTeX on alphaXiv shows a different title, so this needs re-checking before citing.

**Cascade: Composing Software-Hardware Attack Gadgets for Adversarial Threat Amplification in Compound AI Systems** [P] (no bib) — Banerjee, Sahu, Vahldiek-Oberwagner, Sanchez Vicarte, Tiwari. arXiv 2603.12023 (date not shown). Chains system-level gadgets (e.g., bit flips against Llama Guard) with jailbreaks. Shows that attacking the guard's infrastructure is another way around a layer. Out of our threat model.

---

## Items requested but not citable as papers
- **Llama Guard 2**: model card only; not verified this session.
- **PromptGuard (v1)**: model card only; not verified.
- **AlignmentCheck, CodeShield**: described in the LlamaFirewall paper; cite that paper.
- **Tool filter**: a baseline defense in AgentDojo; cite AgentDojo.
- **"AgentShield-style tool gates"**: no such paper found (see the note in (b)). Use Progent/CaMeL/AgentDojo tool filter.
