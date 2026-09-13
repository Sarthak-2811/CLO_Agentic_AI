# Project Requirements Document (PRD): CLO Waterfall Simulator

## 1. What to Build
The CLO Waterfall Simulator is an autonomous, multi-agent AI structuring engine built on LangGraph. It automates the translation of dense legal covenants (Indentures) into dynamic financial models. The system extracts structured priority-of-payment rules, generates executable Monte Carlo cash flow simulations, evaluates the risk profile against rating agency criteria, and cyclically optimizes the capital structure. The ultimate goal is to solve a constrained optimization problem: maximizing the size of the Senior (AAA) tranche without breaching the $\le 0.01\%$ loss probability constraint under stressed scenarios.

## 2. Targeted Users
* **Structuring Desks (Investment Banks):** To rapidly iterate on deal structures, test cliff effects, and optimize the spread arbitrage before taking a CLO to market.
* **Quantitative Analysts (Quants):** To automate the tedious translation of legal PDFs into programmatic backtesting logic.
* **CLO Collateral Managers:** To simulate how portfolio changes (buying/selling underlying loans) will impact their Overcollateralization (OC) and Interest Coverage (IC) tests.
* **Credit Rating Analysts:** To reverse stress-test existing deals and pinpoint the exact macroeconomic default correlations that break the Mezzanine or Senior tranches.

## 3. Core Features
* **Indenture RAG & Extraction Pipeline:** Ingests 300+ page legal PDFs and reliably extracts tranche sizes, interest rates, OC/IC trigger percentages, fee caps, CCC-bucket penalties, and the sequential priority of payments into a strict JSON schema.
* **Agentic Code Generation & Execution:** A "Quant" node that takes the parsed JSON constraints and dynamically writes a Python simulation script utilizing `numpy`/`pandas`, executing 10,000+ Monte Carlo paths.
* **Rule-Based Rating Agency Critic:** An evaluator node that inspects the simulation outputs (loss probabilities, cash flow shortfalls) against hardcoded rating agency methodologies (e.g., Moody's/S&P).
* **Cyclic Optimization Loop:** A LangGraph conditional router that autonomously adjusts tranche sizing or coverage triggers based on the Critic's feedback, rerunning the simulation until the mathematical maximum efficiency is found.
* **Auditable Execution Trail:** Logs every iteration, showing the exact parameters adjusted between cycles (e.g., "Attempt 2 failed at 68% AAA sizing; Attempt 3 adjusted to 65%").