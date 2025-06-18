# AI Infant: A Foundational Cognitive Architecture

This project is an implementation of a synthetic cognitive architecture built from first principles. The goal is not to train a static model on a dataset, but to bootstrap a digital mind that learns about reality through perception, curiosity, comprehension, and reason.

The AI operates under the **Law of Least Action**, utilizing energy-efficient, biologically-inspired cognitive processes.

## Core Principles

- **Grounded Intelligence:** All knowledge is anchored in perception (linguistic or visual). Symbols derive their meaning from associated neural patterns.
- **Emergent Reasoning:** Logic and grammar are not explicitly programmed. They emerge from the statistical properties and physical structure of the AI's neural fabric.
- **Energy Efficiency:** The entire system is designed to run within a strict power budget (~20W), using sparse activations and offline cognitive cycles (like sleep) to manage computationally intensive tasks.
- **Autonomous, Self-Directed Learning:** The AI is driven by an intrinsic curiosity to identify and resolve gaps in its own knowledge, allowing it to learn and research topics autonomously.

## Architecture Overview

The AI is composed of several interconnected "cortices" and modules that communicate through a central `NeuralFabric`:

- **`NeuralFabric`**: The simulated brain matter. A dynamic graph of neurons and synapses where all knowledge is stored as connection patterns.
- **`LanguageCortex`**: The "sense" of reading. Processes text into neural activations.
- **`RelationalCortex`**: The engine of comprehension. Analyzes firing patterns to discover and represent abstract relationships (e.g., Subject-Verb-Object).
- **`LogicCortex`**: The reasoning engine. Performs multi-step deductive inference to synthesize new knowledge from existing facts.
- **`PlanningCortex`**: The seat of agency and motivation. Driven by goals like curiosity and self-correction, it formulates plans to learn and act.
- **`MemoryCore`**: Manages the cognitive cycle of short-term memory, long-term consolidation, dreaming, and forgetting.
- **`External Interfaces`**: `WebBrowser` and `KnowledgeOracle` modules that allow the AI to interact with the web and external LLMs as tools for research.

## How to Run

This project requires Python and several dependencies.

1.  **Clone the repository and set up the environment:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/AI_Infant.git
    cd AI_Infant
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set the API Key (Optional but Recommended):**
    For the AI to be able to ask questions to an external expert (LLM), you need an API key. This project is configured for the DeepSeek API.
    ```bash
    export DEEPSEEK_API_KEY="your_secret_api_key_here"
    ```

4.  **Run the Main Application:**
    The main script guides you through interacting with the AI.
    ```bash
    python main_phase2.py
    ```
    Once running, you can give the AI commands like `research <topic>` to kickstart its autonomous learning process.