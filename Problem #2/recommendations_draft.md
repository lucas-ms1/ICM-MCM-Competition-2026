# Problem F: Future of Work & Educational Recommendations (model + evidence draft)

This file is a report-ready, evidence-first narrative companion to the reproducible pipeline. The final submission text is compiled in LaTeX at `Problem #2/reports/main.tex`, and every numeric claim should be traceable to a named artifact under `Problem #2/data/` or `Problem #2/reports/tables/`.

---

## 1) Problem framing & choices

### Careers (one per required category)
- **STEM**: Software Developers (SOC 15-1252)
- **Trade**: Electricians (SOC 47-2111)
- **Arts**: Writers and Authors (SOC 27-3043)

These three span a wide range of task structures: software is computer/tool intensive; electricians are dominated by onsite physical/manual work; writers are writing-intensive creative production.

### Institutions (one per career)
- **San Diego State University (SDSU)**: Computer Science / Software Engineering (San Diego metro)
- **Los Angeles Trade–Technical College (LATTC)**: Electrical Construction & Maintenance (Los Angeles metro)
- **Academy of Art University**: Writing for Film/TV/Digital Media (San Francisco metro)

We interpret “institution-specific” as: recommendations must reflect (i) **national career outlook** from EP and the GenAI scenario model, and (ii) **local labor market conditions** (employment/wages) from OEWS for the institution’s metro area.

---

## 2) Data (what, from where, and how transformed)

### A) BLS OEWS (local context)
- **What it provides**: local employment levels and wages by occupation for metros and states.
- **How used here**: to report institution-local employment and wages for each career and region.
- **Pipeline**: `python build_tables.py` writes `data/oews_institution_local.csv` and career views under `data/careers/*.csv`.

### B) BLS Employment Projections (national baseline)
- **What it provides**: national 2024–2034 employment trajectory, openings, separations.
- **How used here**: compute **baseline annual growth** \(g_\text{base}\) implied by 2024 and 2034 employment.
- **Pipeline**: `python build_tables.py` reads `data/occupation.xlsx` and writes `data/ep_baseline.csv` with `g_baseline`.

### C) O*NET 30.1 (mechanism layer)
- **What it provides**: task/skill/ability descriptors (0–100 “Importance” scale).
- **How used here**: build five interpretable “why” dimensions (Writing, Tool/Tech, Physical, Social, Creativity), convert them to **percentiles across occupations**, then compute Net Risk.
- **Pipeline**: `python build_mechanism_layer_expanded.py` writes `data/mechanism_layer_all.csv` (774 occupations scored) and `data/mechanism_layer_expanded.csv` (raw + percentiles).

### D) Crosswalks and “why 774”
O*NET uses O*NET-SOC specialties (e.g., `15-1252.00`), while BLS tables are SOC-based at the 7-character stem (e.g., `15-1252`). We slice O*NET-SOC to SOC stems for joining and reporting, which collapses multiple specialties into a single SOC occupation. Coverage is transparent and auditable via:
- `reports/tables/mechanism_coverage.tex` (built from O*NET + BLS tables)
- In particular: 894 O*NET-SOC codes collapse to **774 unique SOC stems**, and 750 of those overlap with the SOC codes present in both OEWS national and EP baseline.

---

## 3) Model (equations + interpretation + scenarios)

### A) Mechanism dimensions (percentiles)
For each occupation \(i\), we compute five dimension percentiles \(x_{i,d}\in[0,1]\) across the scored occupation set:
- Writing intensity (example O*NET elements: Writing, Written Expression, Documenting/Recording Information)
- Tool/technology intensity (Working with Computers, Programming, Technology Design)
- Physical/manual intensity (Handling and Moving Objects, Manual Dexterity, Repairing/Maintaining Equipment)
- Social perceptiveness (Social Perceptiveness, Interpersonal Relationships, Working with the Public)
- Creativity/originality (Originality, Thinking Creatively, Fluency of Ideas)

### B) Net Risk index
Define:

\[
\text{Substitution}_i=\frac{x_{i,\text{Writing}}+x_{i,\text{ToolTech}}}{2},\qquad
\text{Defense}_i=\frac{x_{i,\text{Physical}}+x_{i,\text{Social}}+x_{i,\text{Creativity}}}{3}
\]
\[
\text{NetRisk}_i=\text{Substitution}_i-\text{Defense}_i
\]

Interpretation:
- **NetRisk \(>0\)**: more exposed to substitution pressure → growth headwind.
- **NetRisk \(<0\)**: more sheltered / complementary → growth tailwind (relative).

### C) Scenario definition (not “estimated truth”)
Let \(g_\text{base}\) be baseline annual growth from EP and \(E_{2024}\) be the 2024 employment level used as the projection base. For scenario parameter \(s\ge 0\):

\[
g_\text{adj}=g_\text{base}-s\cdot\text{NetRisk}
\qquad\Rightarrow\qquad
E_{2034}=E_{2024}(1+g_\text{adj})^{10}
\]

Scenarios:
- **Baseline**: \(s=0\)
- **Moderate substitution**: \(s=0.015\)
- **High disruption**: \(s=0.03\)

All scenario outputs are written to `data/scenario_summary.csv` by `python run_scenarios.py`.

---

## 4) Results (dashboards + robustness)

### A) National projections for the three careers (traceable)
From `data/scenario_summary.csv` (also rendered to `reports/tables/scenario_summary.tex`):
- **Software Developers**: NetRisk \(=0.380\), 2034 baseline \(=1.916\)M, 2034 high \(=1.711\)M.
- **Electricians**: NetRisk \(=-0.461\), 2034 baseline \(=813\)k, 2034 high \(=931\)k.
- **Writers and Authors**: NetRisk \(=0.170\), 2034 baseline \(=49.5\)k, 2034 high \(=47.1\)k.

### B) Sensitivity / robustness (minimum acceptable)
- **Sensitivity grid**: `reports/tables/sensitivity_grid.tex` shows 2034 employment under \(s\in\{0.01,0.015,0.02,0.03\}\).
- **Sanity check**: `reports/tables/top_exposed_sheltered.tex` lists top 10 most exposed and most sheltered occupations by NetRisk from the full scored set (`data/mechanism_risk_scored.csv`).

These checks show the index behaves intuitively (writing/tool-heavy occupations tend to rank higher risk; physically/manually intensive occupations tend to rank lower risk).

---

## 5) Recommendations (explicitly tied to model outputs)

### A) SDSU (Software Developers)
- **Program size**: Maintain or modest growth (high disruption still yields net employment gain; see `scenario_summary.csv`).
- **What to teach about GenAI**:
  - Shift from boilerplate coding to system design, testing, security, and evaluation.
  - Require audit trails: version control, test suites, provenance notes for AI-generated code.
  - Explicitly address attribution and license/IP risks in code generation workflows.
- **Policy**:
  - Early courses: limited/structured AI to protect learning integrity.
  - Advanced courses: allow AI with disclosure and reproducibility requirements.

### B) LATTC (Electricians)
- **Program size**: Aggressive growth (NetRisk is negative and scenarios preserve/boost demand; see `scenario_summary.csv`).
- **What to teach about GenAI**:
  - Keep hands-on competencies primary (installation cannot be automated by text models).
  - Add modules on AI-assisted diagnostics/documentation/scheduling and code-compliant planning.
  - Emphasize privacy/safety and low-compute, practical tools (business operations, templates).
- **Policy**: Treat GenAI as a productivity tool for small contractors; require accuracy checks against local codes and manufacturer documentation.

### C) Academy of Art University (Writers and Authors)
- **Program size**: Consolidate/specialize (high disruption flips the national outlook to contraction; see `scenario_summary.csv`).
- **What to teach about GenAI**:
  - Move from “content generation” to narrative strategy, editing, and production workflows.
  - Teach AI co-writing with strong provenance: disclosure, citations, and originality standards.
  - Build differentiation around voice, revision quality, and ethics-compliant sourcing.
- **Policy**: Mandatory disclosure in portfolios; grading rubrics should reward originality and documented creative process.

---

## 6) Other factors beyond employment (prompt-required)

Employability is not the only success metric. We explicitly include:
- **Learning integrity** (assessment validity; attribution and plagiarism risks)
- **Equity/access** (students’ access to tools; fairness; accommodation)
- **Sustainability** (energy/water/compute costs; tool selection and usage policies)
- **IP/attribution compliance** (licensing, provenance, creator credit)

A simple defensible extension is an overall program objective:

\[
\mathrm{Score}=w_E\cdot\mathrm{Employability}-w_I\cdot\mathrm{IntegrityRisk}-w_S\cdot\mathrm{SustainabilityCost}
\]

Qualitative implication: increasing \(w_I\) pushes toward stricter disclosure and assessment designs robust to AI; increasing \(w_S\) pushes toward lightweight tools and fewer “always-on” GenAI requirements.

---

## 7) Generalization beyond the three programs

The approach generalizes by (i) swapping in any occupation(s), (ii) recomputing local OEWS context for any institution’s region, and (iii) selecting a scenario parameter grid consistent with decision-makers’ risk tolerance. What does **not** generalize 1:1 is the institutional policy design, which depends on program mission, student constraints, and local labor market.

---

## References (minimum anchors for the final PDF)
- BLS OEWS technical notes (methodology and definitions)
- BLS EP Table 1.10 definitions (occupational separations/openings)
- O*NET 30.1 database landing page + taxonomy + license statement
- One “exposure to LLMs” anchor (e.g., OpenAI occupational exposure)
- One productivity/complementarity anchor (e.g., NBER “Generative AI at Work”)
- One tasks-based economic framing anchor (Autor; Acemoglu–Restrepo; MIT framing)
