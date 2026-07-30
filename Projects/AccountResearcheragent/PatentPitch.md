# OCDM Agent Curated – AR: Concurrent Skill Selection and Recursive Intent Decomposition Engine

### Patent Case Submission — Invention Disclosure for Committee Review

*A stage-aware orchestration and evaluation framework for a natural-language-to-data (NL-to-Data) pipeline. An orchestrator is constrained to concurrent skill selection and routing-time dependency-topology judgment; each selected skill performs recursive intent decomposition under bounded per-skill context; cross-domain dependencies are compiled into a single governed query; and answer quality is judged stage-by-stage with combined deterministic + LLM validation off the critical path.*

> This submission is deliberately limited to the four items requested by the reviewer: **(1) Quantifiable Technical Benefit, (2) Why this is technically better, (3) Prior Art Comparison, (4) Strong Patent Angle.**

---

## 1. Quantifiable Technical Benefit

Not "more stages / better architecture / better explainability." **Measurable improvements**, each tied to a concrete mechanism and a metric that Qual Beat can report.

| Measurable Benefit | Mechanism That Produces It | Metric |
|---|---|---|
| **Reduced compute cost** | Fewer calls × smaller prompts; no per-hop re-planning loop | $/request; compute-seconds/request |
| **Fewer LLM calls** | One routing inference decides *what* + *dependency topology*; N dependent hops collapse to 1 reasoning pass + 1 compiled query | Model-calls per request |
| **Lower token consumption** | Router sees only 1-line skill descriptors; each skill reads only its own bounded slice; nested route never re-feeds intermediate rows through the model | Input/output tokens per request (single / independent-multi / nested) |
| **Faster response & evaluation time** | Independent domains fan out in parallel; a dependency resolves in one atomic round-trip; quality adjudication runs off the critical path | p50/p95 end-to-end latency; latency vs. #domains |
| **Reduced network / data round-trips** | Producer→consumer dependency compiled into one query (inner CTE + join), executed atomically in the data engine | Data-engine round-trips per dependent request |
| **Lower storage / state overhead** | No shared scratchpad or inter-skill memory; fixed request/result packages only | Retained intermediate-state size per request |
| **Better security / privacy** | Row-level-security predicate + canonical-identity join woven through every compiled stage; entity resolved once and broadcast unchanged | RLS-leak / identity-resolution-mismatch rate |
| **Higher accuracy with fewer human reviews** | Deterministic per-stage fallback + stage-wise LLM-plus-deterministic validation isolates failures to the exact stage | Degraded-success / terminal-failure rate; human-validation effort per 100 requests |
| **Flat scaling of the skill catalog** | New domain ships as its own bounded bundle; no central prompt grows | Routing accuracy & cost held flat vs. #skills |

> All figures to be populated from **Qual Beat** champion–challenger runs — same model family, permissions, data, and question set — against a controlled ReAct/LLMCompiler-style planner baseline, reported across single-domain, independent multi-domain, and nested requests. **Do not estimate.**

---

## 2. Why This Is Technically Better

**The reviewer's question:** *"If I use this stage-aware orchestration-and-evaluation approach instead of an existing end-to-end / planner-loop approach, what technical problem is solved and what measurable improvement do I get?"*

**The problem with existing approaches — a planner loop walks a mutable graph:** plan → call → observe → re-feed → re-plan. That design carries four built-in costs:

- A **model round-trip per hop** → latency + tokens grow with the number of steps.
- A **fragile row hand-off** between steps (intermediate results re-fed through the model) → correctness risk.
- **Security re-applied per hop** → a leak surface at every tool call.
- A **central prompt that grows per tool** → accuracy and cost decay as the catalog scales.
- **Terminal-output evaluation only** → when the final answer is wrong, the failing step is unknown and must be debugged manually.

**What this invention solves, mechanism by mechanism:**

- **Topology chosen once, not iterated.** One routing inference emits either a flat set of independent skills or a single `Nesting` signal; the execution path is fixed before any tool runs → deterministic, cacheable, inspectable. *Measured as:* fewer model calls, lower p95 on dependent questions.
- **Dependencies compiled, not sequenced.** Producer becomes an inner CTE, consumer a join against it, executed atomically → join semantics guarantee correctness and remove the re-feed round-trip. *Measured as:* fewer LLM calls and fewer data round-trips per dependent request.
- **Identity resolved once.** The entity is resolved to a canonical identifier and broadcast unchanged → smaller error and attack surface. *Measured as:* identity-resolution-mismatch rate.
- **Bounded per-skill reasoning.** Each skill reads only its own slice, in parallel → no token blow-up, no throttling from a monolithic all-metadata prompt. *Measured as:* tokens/request held flat vs. #skills.
- **Stage-aware evaluation with combined deterministic + LLM validation, off the critical path.** Each stage carries a deterministic fallback and is validated independently, so a failure is isolated to the exact stage rather than inferred from a wrong final answer → automated root-cause identification and earlier failure detection. *Measured as:* reduced human-validation effort and fewer expensive re-runs.

**Measurable improvement (to be populated from Qual Beat — placeholders, do not estimate):**

- ~X% fewer model calls (N dependent hops → 1 reasoning pass + 1 compiled query)
- ~X% fewer tokens (no intermediate-row re-feed; bounded per-skill context)
- ~X% lower p95 latency on dependent questions
- ~X% fewer duplicate cross-skill queries
- ~X% reduction in human-validation effort via stage-level failure isolation
- ~X% fewer expensive re-runs via earlier, stage-specific failure detection
- Cost and routing accuracy held ~flat as the number of skills grows

---

## 3. Prior Art Comparison

**Representative prior art:** ReAct (2210.03629) · Plan-and-Solve (2305.04091) · LLMCompiler (2312.04511) · Tree-of-Thoughts (2305.10601) · LangGraph · CrewAI · AutoGen · Semantic Kernel · OpenAI / Bedrock / Vertex agent stacks.

| Dimension | Existing Solutions (end-to-end / planner-loop) | Proposed Solution (stage-aware, skill-owned) |
|---|---|---|
| **Control flow** | Mutable task graph re-evaluated after each step | Topology fixed once at routing, then executed |
| **Where decomposition happens** | Central orchestrator plans and splits tasks | Each skill decomposes intent recursively (two altitudes) |
| **Dependency handling** | Sequenced tool calls; rows re-fed through the model | One atomic governed query (inner CTE + join) |
| **Round-trips** | One reasoning pass per hop; grows with hops | 1 routing + 1 execution; independents in parallel |
| **Identity resolution** | Re-resolved per tool | Resolved once → broadcast unchanged |
| **Duplicate work** | Reconciled by planner or shared memory | Dropped locally via per-skill "not-mine" metadata, pre-query |
| **Security** | Re-applied per tool call → leak surface | RLS + canonical join woven into every compiled stage |
| **Evaluation scope** | Evaluate the final answer only | Evaluate every stage |
| **Failure source** | Unknown; requires manual debugging | Isolated to the exact stage; automated root-cause detection |
| **Validation method** | Pure LLM judging | LLM + deterministic validation per stage |
| **Explainability** | Limited; reconstruct intent from a long transcript | Full stage traceability (one topology signal + one readable governed query) |
| **Scaling with new skills** | Central prompt/catalog grows; accuracy & cost decay | Ship a bounded skill bundle; no central prompt grows |

**Obviousness defense.** A CTE alone is well known; text-to-SQL is well known; parallel tool routing is well known; LLM-as-judge is well known. The invention is the **coupled combination**: detecting a cross-skill dependency *at routing time*, selecting the execution topology in that same inference, carrying canonical identity and RLS through the compiled stages, evaluating each stage with combined deterministic + LLM validation, and having all of this coexist with distributed, skill-local ownership under bounded per-skill context. The coupling — not any individual element — is what is non-obvious.

---

## 4. Strong Patent Angle

**Primary angle:**

> *A **stage-aware orchestration and evaluation framework** for a natural-language-to-data pipeline that combines routing-time dependency-topology selection, skill-owned recursive intent decomposition under bounded context, atomic governed dependency compilation (inner CTE + join with canonical-identity join and RLS carried through every stage), and per-stage LLM-plus-deterministic validation off the critical path — enabling **automated root-cause isolation, explainable stage-level evaluation, and reduced token/latency/validation cost** compared to terminal-output planner-loop systems, with flat cost as the skill catalog grows.*

**Two independent-claim candidates:**

1. **Two-level decomposition with distributed exclusion under bounded context** — one router inference selects the relevant domains *and* judges dependency topology; each skill decomposes into sub-questions and prunes "not-mine" candidates against its own bounded metadata slice, so each sub-intent is owned by exactly one skill with no inter-skill negotiation.

2. **Routing-time topology selection with atomic governed dependency compilation and stage-wise evaluation** — the same inference emits a `Nesting` signal; the detected dependency is compiled into a single query (inner CTE + join) carrying the canonical-identity join and RLS predicate through its stages and executed atomically; and each stage is validated by combined deterministic + LLM checks off the critical path for root-cause isolation.

**Why this is a strong case:**

- **Mechanisms, not vague outcomes** — every element names a concrete, inspectable, and therefore claimable mechanism.
- **Every benefit is measurable** — fewer calls, fewer tokens, lower p95, fewer duplicate queries, reduced human validation, flat scaling (all reported via Qual Beat).
- **Coupled-combination framing** forecloses "just a CTE," "just a planner," or "just LLM-as-judge" obviousness attacks.
- **Product-anchored** — live in Account Researcher via the Sales Agent surface (production 1 Jun 2026); generalizes across Microsoft 365 Copilot, Copilot Studio, and Azure AI agent ecosystems.
