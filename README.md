# Feynman-Enhanced Learning Agent (Ongoing Project)

---

## 1. Overview

1. This project is an ongoing effort to build a structured, checkpoint-driven learning agent using LangGraph as the orchestration layer.
2. The system guides learners through sequential, customizable checkpoints.
3. Understanding is verified at each stage before progression.
4. When comprehension falls below a defined threshold, Feynman-style teaching is triggered.
5. The broader objective is to integrate this agent with a self-hosted, fine-tuned language model.
6. This removes third-party API dependency and enables full-stack ownership across:
   1. Training
   2. Serving
   3. Application orchestration

---

## 2. Vision

Our long-term vision is to build a scalable, self-hosted AI tutoring system that:

1. Delivers personalized 1:1 learning experiences.
2. Operates continuously with structured feedback loops.
3. Uses domain-adapted, fine-tuned models instead of external APIs.
4. Combines retrieval, verification, and simplified teaching into a unified learning pipeline.

This repository represents the **agentic application layer** of a modular LLM platform composed of:

1. Infrastructure provisioning  
2. Fine-tuning pipelines  
3. Model serving layer  
4. Agent-based learning applications (this repository)

The strategic objective is to route all inference requests through our own fine-tuned models deployed within controlled infrastructure.

---

## 3. Motivation

This project addresses the following challenges:

1. Limited access to affordable 1:1 tutoring.
2. Lack of structured progression in generative AI tutoring systems.
3. Over-reliance on passive content generation.

The system is designed to:

1. Provide individualized feedback 24/7.
2. Use student-provided notes as primary context.
3. Retrieve supplementary web material when needed.
4. Explain complex topics using simplified, Feynman-style reasoning.
5. Enforce structured learning progression rather than assumption-based advancement.

---

## 4. Core Architecture

The system is built around a **Learning State Graph** powered by LangGraph.

### 4.1 Key Components

1. Learning State Graph  
   1. Orchestrates workflow using conditional routing.
   2. Maintains evolving state across checkpoints.

2. Checkpoint System  
   1. Defines structured learning milestones.
   2. Associates measurable success criteria with each checkpoint.

3. Context Processing  
   1. Chunks learning material.
   2. Embeds content for similarity retrieval.

4. Web Search Integration  
   1. Retrieves supplementary learning content dynamically.
   2. Activates when context is insufficient.

5. Embedding Storage Layer  
   1. Stores embeddings for targeted retrieval.
   2. Ensures only relevant chunks are used during verification.

6. Question Generation  
   1. Produces checkpoint-aligned evaluation questions.

7. Understanding Verification  
   1. Applies a 70% understanding threshold.
   2. Provides structured feedback.
   3. Determines progression or remediation.

8. Feynman Teaching Module  
   1. Simplifies complex concepts.
   2. Uses analogies and key concept reinforcement.
   3. Activates when understanding is insufficient.

---

## 5. Learning Workflow

The agent follows a structured, state-driven learning cycle:

### 5.1 Sequential Process

1. Checkpoint Definition  
   1. Generates structured milestones.
   2. Defines explicit verification criteria.

2. Context Building  
   1. Processes student-provided material.
   2. Retrieves web content if required.

3. Context Validation  
   1. Evaluates whether context satisfies checkpoint requirements.
   2. Triggers additional search if missing information is detected.

4. Embedding Storage  
   1. Stores context chunks.
   2. Enables targeted retrieval during verification.

5. Understanding Verification  
   1. Generates checkpoint-specific questions.
   2. Evaluates responses.
   3. Applies a 70% threshold.
   4. Produces structured feedback.

6. Progressive Learning  
   1. Advances to the next checkpoint when verified.
   2. Triggers Feynman teaching when below threshold.

---

## 6. State Diagram

The following diagram represents the orchestration logic and conditional routing of the system:

<img width="565" height="963" alt="architecture" src="https://github.com/user-attachments/assets/66383cc7-b8b4-4719-bad8-4793de037c04" />


---

## 7. Transition to Fine-Tuned Model

The current implementation demonstrates complete orchestration logic.

The next phase includes:

1. Routing inference to a self-hosted, fine-tuned language model.
2. Eliminating third-party API reliance.
3. Integrating with Kubernetes-based serving infrastructure.
4. Aligning with the broader full-stack LLM ownership strategy.

This transition ensures:

1. Domain adaptation.
2. Cost control.
3. Infrastructure sovereignty.
4. Scalable deployment.

---

## 8. Current Status

1. Learning state graph: Implemented  
2. Checkpoint-driven workflow: Implemented  
3. Retrieval and embedding validation: Implemented  
4. Verification threshold logic: Implemented  
5. Feynman-style remediation: Implemented  
6. Fine-tuned model integration: In progress  

---

## 9. Conclusion

This project demonstrates:

1. State-driven orchestration.
2. Conditional routing with verification thresholds.
3. Retrieval-augmented evaluation.
4. Simplified remediation using structured teaching.
5. A modular architecture aligned with full-stack LLM ownership.

The long-term objective is to evolve this system into a fully self-hosted, domain-adapted AI tutoring platform built on proprietary fine-tuned models and scalable infrastructure.

---

## 10. Requirements

1. Python 3.10+
2. LangGraph
3. LangChain
4. Pydantic
5. Embedding model support
6. Vector similarity utilities
7. (Planned) Fine-tuned LLM inference endpoint integration
