# Agentic AI — Must-Read Research Papers of All Time

A curated, topic-sectioned reading list of the foundational and landmark papers behind
modern **agentic AI**: systems where LLMs reason, plan, use tools, remember, reflect, and
collaborate to accomplish goals.

> Legend: ⭐ = absolute must-read / field-defining · 🧱 = foundational building block · 🧪 = key benchmark or evaluation

---

## Table of Contents

1. [Foundations — Transformers & Language Models](#1-foundations--transformers--language-models)
2. [Scaling, Emergence & Foundation Models](#2-scaling-emergence--foundation-models)
3. [Instruction Tuning & Alignment (RLHF)](#3-instruction-tuning--alignment-rlhf)
4. [Reasoning — Chain-of-Thought & Beyond](#4-reasoning--chain-of-thought--beyond)
5. [Tool Use & Function Calling](#5-tool-use--function-calling)
6. [Agent Architectures — Reason + Act](#6-agent-architectures--reason--act)
7. [Planning & Task Decomposition](#7-planning--task-decomposition)
8. [Reflection & Self-Improvement](#8-reflection--self-improvement)
9. [Memory & Long-Horizon Context](#9-memory--long-horizon-context)
10. [Retrieval-Augmented Generation (RAG)](#10-retrieval-augmented-generation-rag)
11. [Multi-Agent Systems](#11-multi-agent-systems)
12. [Code & Software Engineering Agents](#12-code--software-engineering-agents)
13. [Computer-Use, Web & GUI Agents](#13-computer-use-web--gui-agents)
14. [Embodied & Robotics Agents](#14-embodied--robotics-agents)
15. [Reinforcement Learning for Agents](#15-reinforcement-learning-for-agents)
16. [Protocols, Frameworks & Standards](#16-protocols-frameworks--standards)
17. [Evaluation, Benchmarks & Safety](#17-evaluation-benchmarks--safety)

---

## 1. Foundations — Transformers & Language Models

The bedrock every agent is built on.

- [x] ⭐🧱 **Attention Is All You Need** — Vaswani et al., 2017. Introduces the Transformer. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [] 🧱 **BERT: Pre-training of Deep Bidirectional Transformers** — Devlin et al., 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- [x] 🧱 **Improving Language Understanding by Generative Pre-Training (GPT-1)** — Radford et al., 2018. [PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [x] 🧱 **Language Models are Unsupervised Multitask Learners (GPT-2)** — Radford et al., 2019. [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] 🧱 **Sequence to Sequence Learning with Neural Networks** — Sutskever et al., 2014. [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
- [ ] 🧱 **Neural Machine Translation by Jointly Learning to Align and Translate** — Bahdanau et al., 2014. Origin of attention. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

## 2. Scaling, Emergence & Foundation Models

Why bigger models unlock agent-like behavior.

- [ ] ⭐ **Language Models are Few-Shot Learners (GPT-3)** — Brown et al., 2020. In-context learning. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- [ ] ⭐ **Scaling Laws for Neural Language Models** — Kaplan et al., 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- [ ] **Training Compute-Optimal Large Language Models (Chinchilla)** — Hoffmann et al., 2022. [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- [ ] **Emergent Abilities of Large Language Models** — Wei et al., 2022. [arXiv:2206.07682](https://arxiv.org/abs/2206.07682)
- [ ] **On the Opportunities and Risks of Foundation Models** — Bommasani et al., 2021. Coined "foundation model." [arXiv:2108.07258](https://arxiv.org/abs/2108.07258)
- [ ] **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., 2023. [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

## 3. Instruction Tuning & Alignment (RLHF)

Turning raw models into helpful, controllable assistants.

- [ ] ⭐ **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** — Ouyang et al., 2022. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- [ ] **Deep Reinforcement Learning from Human Preferences** — Christiano et al., 2017. Origin of RLHF. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- [ ] **Finetuned Language Models Are Zero-Shot Learners (FLAN)** — Wei et al., 2021. [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
- [ ] **Constitutional AI: Harmlessness from AI Feedback** — Bai et al., 2022. RLAIF. [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- [ ] **Direct Preference Optimization (DPO)** — Rafailov et al., 2023. RLHF without RL. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

## 4. Reasoning — Chain-of-Thought & Beyond

The cognitive core of an agent.

- [ ] ⭐ **Chain-of-Thought Prompting Elicits Reasoning in LLMs** — Wei et al., 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- [ ] **Self-Consistency Improves Chain of Thought Reasoning** — Wang et al., 2022. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- [ ] **Large Language Models are Zero-Shot Reasoners ("Let's think step by step")** — Kojima et al., 2022. [arXiv:2205.11916](https://arxiv.org/abs/2205.11916)
- [ ] ⭐ **Tree of Thoughts: Deliberate Problem Solving with LLMs** — Yao et al., 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- [ ] **Graph of Thoughts** — Besta et al., 2023. [arXiv:2308.09687](https://arxiv.org/abs/2308.09687)
- [ ] **Least-to-Most Prompting Enables Complex Reasoning** — Zhou et al., 2022. [arXiv:2205.10625](https://arxiv.org/abs/2205.10625)
- [ ] **STaR: Bootstrapping Reasoning with Reasoning** — Zelikman et al., 2022. [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)

## 5. Tool Use & Function Calling

Giving models hands to act on the world.

- [ ] ⭐ **Toolformer: Language Models Can Teach Themselves to Use Tools** — Schick et al., 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- [ ] **TALM: Tool Augmented Language Models** — Parisi et al., 2022. [arXiv:2205.12255](https://arxiv.org/abs/2205.12255)
- [ ] **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face** — Shen et al., 2023. [arXiv:2303.17580](https://arxiv.org/abs/2303.17580)
- [ ] **Gorilla: Large Language Model Connected with Massive APIs** — Patil et al., 2023. [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- [ ] 🧪 **ToolLLM: Facilitating LLMs to Master 16000+ Real-World APIs** — Qin et al., 2023. [arXiv:2307.16789](https://arxiv.org/abs/2307.16789)

## 6. Agent Architectures — Reason + Act

The papers that defined "LLM agent."

- [ ] ⭐🧱 **ReAct: Synergizing Reasoning and Acting in Language Models** — Yao et al., 2022. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- [ ] ⭐ **Generative Agents: Interactive Simulacra of Human Behavior** — Park et al., 2023. [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- [ ] **MRKL Systems: Modular Reasoning, Knowledge and Language** — Karpas et al., 2022. [arXiv:2205.00445](https://arxiv.org/abs/2205.00445)
- [ ] **The Rise and Potential of LLM-Based Agents: A Survey** — Xi et al., 2023. [arXiv:2309.07864](https://arxiv.org/abs/2309.07864)
- [ ] **A Survey on Large Language Model Based Autonomous Agents** — Wang et al., 2023. [arXiv:2308.11432](https://arxiv.org/abs/2308.11432)

## 7. Planning & Task Decomposition

- [ ] **Plan-and-Solve Prompting** — Wang et al., 2023. [arXiv:2305.04091](https://arxiv.org/abs/2305.04091)
- [ ] ⭐ **Voyager: An Open-Ended Embodied Agent with LLMs** — Wang et al., 2023. Lifelong learning in Minecraft. [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)
- [ ] **Describe, Explain, Plan and Select (DEPS)** — Wang et al., 2023. [arXiv:2302.01560](https://arxiv.org/abs/2302.01560)
- [ ] **LLM+P: Empowering LLMs with Optimal Planning Proficiency** — Liu et al., 2023. [arXiv:2304.11477](https://arxiv.org/abs/2304.11477)
- [ ] **ADaPT: As-Needed Decomposition and Planning with Language Models** — Prasad et al., 2023. [arXiv:2311.05772](https://arxiv.org/abs/2311.05772)

## 8. Reflection & Self-Improvement

- [ ] ⭐ **Reflexion: Language Agents with Verbal Reinforcement Learning** — Shinn et al., 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- [ ] **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al., 2023. [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
- [ ] **CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing** — Gou et al., 2023. [arXiv:2305.11738](https://arxiv.org/abs/2305.11738)
- [ ] **Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation** — Zelikman et al., 2023. [arXiv:2310.02304](https://arxiv.org/abs/2310.02304)

## 9. Memory & Long-Horizon Context

- [ ] **MemGPT: Towards LLMs as Operating Systems** — Packer et al., 2023. [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- [ ] **Generative Agents (memory stream)** — Park et al., 2023 (see §6). [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- [ ] **MemoryBank: Enhancing LLMs with Long-Term Memory** — Zhong et al., 2023. [arXiv:2305.10250](https://arxiv.org/abs/2305.10250)
- [ ] **Retrieval-Augmented Generation for Long Context** — see §10.

## 10. Retrieval-Augmented Generation (RAG)

Grounding agents in external knowledge.

- [ ] ⭐🧱 **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — Lewis et al., 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- [ ] **Dense Passage Retrieval (DPR)** — Karpukhin et al., 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- [ ] **REALM: Retrieval-Augmented Language Model Pre-Training** — Guu et al., 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
- [ ] **Self-RAG: Learning to Retrieve, Generate, and Critique** — Asai et al., 2023. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- [ ] **Active Retrieval Augmented Generation (FLARE)** — Jiang et al., 2023. [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)

## 11. Multi-Agent Systems

Agents collaborating, debating, and dividing labor.

- [ ] ⭐ **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Wu et al., 2023. [arXiv:2308.08155](https://arxiv.org/abs/2308.08155)
- [ ] ⭐ **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework** — Hong et al., 2023. [arXiv:2308.00352](https://arxiv.org/abs/2308.00352)
- [ ] **CAMEL: Communicative Agents for "Mind" Exploration** — Li et al., 2023. [arXiv:2303.17760](https://arxiv.org/abs/2303.17760)
- [ ] **Improving Factuality and Reasoning via Multiagent Debate** — Du et al., 2023. [arXiv:2305.14325](https://arxiv.org/abs/2305.14325)
- [ ] **ChatDev: Communicative Agents for Software Development** — Qian et al., 2023. [arXiv:2307.07924](https://arxiv.org/abs/2307.07924)
- [ ] **AgentVerse: Facilitating Multi-Agent Collaboration** — Chen et al., 2023. [arXiv:2308.10848](https://arxiv.org/abs/2308.10848)

## 12. Code & Software Engineering Agents

- [ ] ⭐ **Evaluating Large Language Models Trained on Code (Codex / HumanEval)** — Chen et al., 2021. [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- [ ] 🧪 **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** — Jimenez et al., 2023. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- [ ] **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** — Yang et al., 2024. [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
- [ ] **Self-Debugging: Teaching LLMs to Debug their Own Code** — Chen et al., 2023. [arXiv:2304.05128](https://arxiv.org/abs/2304.05128)
- [ ] **CodeAct: Executable Code Actions Elicit Better LLM Agents** — Wang et al., 2024. [arXiv:2402.01030](https://arxiv.org/abs/2402.01030)

## 13. Computer-Use, Web & GUI Agents

- [ ] 🧪 **WebArena: A Realistic Web Environment for Building Autonomous Agents** — Zhou et al., 2023. [arXiv:2307.13854](https://arxiv.org/abs/2307.13854)
- [ ] 🧪 **Mind2Web: Towards a Generalist Agent for the Web** — Deng et al., 2023. [arXiv:2306.06070](https://arxiv.org/abs/2306.06070)
- [ ] **WebGPT: Browser-Assisted Question-Answering with Human Feedback** — Nakano et al., 2021. [arXiv:2112.09332](https://arxiv.org/abs/2112.09332)
- [ ] **CogAgent: A Visual Language Model for GUI Agents** — Hong et al., 2023. [arXiv:2312.08914](https://arxiv.org/abs/2312.08914)
- [ ] **WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models** — He et al., 2024. [arXiv:2401.13919](https://arxiv.org/abs/2401.13919)

## 14. Embodied & Robotics Agents

- [ ] ⭐ **SayCan: Do As I Can, Not As I Say** — Ahn et al., 2022. Grounding language in robotic affordances. [arXiv:2204.01691](https://arxiv.org/abs/2204.01691)
- [ ] **Inner Monologue: Embodied Reasoning through Planning with Language Models** — Huang et al., 2022. [arXiv:2207.05608](https://arxiv.org/abs/2207.05608)
- [ ] **PaLM-E: An Embodied Multimodal Language Model** — Driess et al., 2023. [arXiv:2303.03378](https://arxiv.org/abs/2303.03378)
- [ ] **RT-2: Vision-Language-Action Models** — Brohan et al., 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
- [ ] **Code as Policies: Language Model Programs for Embodied Control** — Liang et al., 2022. [arXiv:2209.07753](https://arxiv.org/abs/2209.07753)

## 15. Reinforcement Learning for Agents

Classic RL and decision-making foundations that inform agent training.

- [ ] ⭐🧱 **Playing Atari with Deep Reinforcement Learning (DQN)** — Mnih et al., 2013. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
- [ ] **Proximal Policy Optimization (PPO)** — Schulman et al., 2017. Powers RLHF. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- [ ] **Mastering the Game of Go without Human Knowledge (AlphaGo Zero)** — Silver et al., 2017. [Nature](https://www.nature.com/articles/nature24270)
- [ ] **Decision Transformer: Reinforcement Learning via Sequence Modeling** — Chen et al., 2021. [arXiv:2106.01345](https://arxiv.org/abs/2106.01345)
- [ ] **A Generalist Agent (Gato)** — Reed et al., 2022. [arXiv:2205.06175](https://arxiv.org/abs/2205.06175)

## 16. Protocols, Frameworks & Standards

Emerging interoperability layers for agent ecosystems.

- [ ] **Model Context Protocol (MCP)** — Anthropic, 2024. [Docs](https://modelcontextprotocol.io/) · [Announcement](https://www.anthropic.com/news/model-context-protocol)
- [ ] **ReAct-style ToolFormer patterns** — see §5 & §6.
- [ ] **OpenAI Function Calling / Assistants API** — OpenAI, 2023. [Docs](https://platform.openai.com/docs/guides/function-calling)
- [ ] **Agent2Agent (A2A) Protocol** — Google, 2025. [Announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)

## 17. Evaluation, Benchmarks & Safety

How we measure — and safeguard — agentic systems.

- [ ] 🧪 **AgentBench: Evaluating LLMs as Agents** — Liu et al., 2023. [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)
- [ ] 🧪 **GAIA: A Benchmark for General AI Assistants** — Mialon et al., 2023. [arXiv:2311.12983](https://arxiv.org/abs/2311.12983)
- [ ] 🧪 **ToolBench / API-Bank** — Li et al., 2023. [arXiv:2304.08244](https://arxiv.org/abs/2304.08244)
- [ ] **Concrete Problems in AI Safety** — Amodei et al., 2016. [arXiv:1606.06565](https://arxiv.org/abs/1606.06565)
- [ ] **Universal and Transferable Adversarial Attacks on Aligned LLMs** — Zou et al., 2023. [arXiv:2307.15043](https://arxiv.org/abs/2307.15043)
- [ ] **Sparks of Artificial General Intelligence (GPT-4 early experiments)** — Bubeck et al., 2023. [arXiv:2303.12712](https://arxiv.org/abs/2303.12712)

---

### How to Use This List

- **New to the field?** Read top-to-bottom through §1 → §6. That path takes you from the
  Transformer to the first true LLM agents (ReAct).
- **Building an agent today?** Focus on §5–§11 and §16.
- **Doing research/evals?** Prioritize the 🧪 benchmark papers and §17.

> Contributions welcome — add papers with author, year, and a stable link (prefer arXiv abs pages).
