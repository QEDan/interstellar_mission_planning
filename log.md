# Mission Planning & Repository Work Log

This log documents environment setup, physics model findings, root cause analysis, and the implementation of the mission plans in `notebooks/Inertia_Ship.ipynb`.

---

## 1. Environment & Dependency Setup

### 1.1 Python Compatibility & `scimath`
* **Issue**: Running `uv run jupyter notebook` originally resulted in `ModuleNotFoundError: No module named 'scimath'`.
* **Root Cause**:
  * The repo lacked a local virtual environment (`.venv`) and `pyproject.toml`, causing `uv` to fall back to the host system interpreter (Anaconda Python 3.13.5).
  * `scimath==5.0.0` uses legacy `setuptools` build logic where `scimath/_version.py` expects the source directory to be on `sys.path`. In Python 3.12+ PEP 517 isolated build environments, this throws `ImportError: Unable to import git_revision`.
  * In Python 3.11 and Python 3.9, `scimath==5.0.0` installs and builds without error.
* **Resolution**:
  1. Created [`.python-version`](file:///home/mazur/linstore/repos/interstellar_mission_planning/.python-version) pinned to `3.11`.
  2. Initialized a local virtual environment: `uv venv --python 3.11`.
  3. Installed dependencies into `.venv`:
     ```bash
     uv pip install --python .venv scimath numpy scipy matplotlib pandas vpython pytest pytest-mock mock jupyter notebook
     ```
  4. Note: `.python-version` and `.venv` are already tracked in `.gitignore`.

### 1.2 Running Commands
* **Run Jupyter Notebook**:
  ```bash
  uv run jupyter notebook
  ```
* **Run Unit Tests**:
  ```bash
  PYTHONPATH=. uv run pytest
  ```
  *(All 79 unit tests across engine, solar sail, starship, and swimmer pass.)*

---

## 2. Physics Model & Codebase Changes

Recent commits to the repository updated several core physics components:
1. **Relativistic Rocket Mechanics (`src/engine.py`)**:
   * Uses relativistic exhaust momentum:
     $$u = \left(\frac{v_e}{c}\right) \ln\left(\frac{m_0}{m_f}\right),\quad \Delta v = c \tanh(u)$$
   * Inversion function `set_target_delta_v` solves for fuel mass using relativistic mass ratios.
2. **Gravitational Corrections (`src/starship.py`, `src/utils.py`)**:
   * Engine burns now account for stellar escape velocity:
     * Inbound: $v_{\text{inbound}} = \sqrt{v^2 + v_{\text{esc}}^2}$
     * Burn $\Delta v$ required to achieve target velocity: $\Delta v = v_{\text{inbound}} - v_{\text{target}}$.
3. **Stellar Luminosity & Radiation Pressure (`src/solar_sail.py`)**:
   * Sailing trajectories resolve destination stellar properties (e.g., Proxima Centauri luminosity $L \approx 0.001567\, L_\odot$).
4. **SWIMMER Plasma Engine (`src/swimmer.py`)**:
   * Propellantless propulsion based on Brisbin (2018), *Spacecraft With Interstellar Medium Momentum Exchange Reactions* ([arXiv:1808.02019](https://arxiv.org/abs/1808.02019)).
   * Interacts with ambient ISM ions via an electrostatic pusher plate ($A_{\text{eff}} \approx 1.54 \times 10^{18}\text{ m}^2$).
   * Generates thrust during acceleration and enhanced braking momentum exchange during deceleration.

---

## 3. Mission Planning Plot Beats & Requirements

The mission plan is designed for a science-fiction generation ship story with the following narrative constraints:
1. **Generation Ship Launch**:
   * Begins with a solar sail phase to accelerate out of the solar system.
   * Followed by a **~50-year fusion engine phase** to reach cruising velocity.
2. **Nominal Mission**:
   * Must reach a nearby star (Alpha Centauri system, $4.244\text{ ly}$).
   * Decelerates into a stable orbit of the destination star.
   * Acceptable for the nominal mission to take thousands of years.
3. **Mission Abort & Return**:
   * Aborted near the beginning of the cruise phase (at ~50 years after launch).
   * **Dry mass of the ship is a fixed quantity** (represents passenger habitats and vital resources; payload mass of $6.6 \times 10^{10}\text{ kg}$ is not deducted).
   * SWIMMER engines are constructed onboard to:
     1. Decelerate outbound motion to zero.
     2. Accelerate back toward Earth.
     3. Decelerate into a stable Earthlike orbit around the Sun (~1 AU).
   * **Return Timeline**: Total elapsed time from original launch should be **within a couple of hundred years** (~150–200 years).

---

## 4. Root Cause: Why the Original Plan Failed

* In the previous version of `Inertia_Ship.ipynb`, Cell 4 executed:
  ```python
  ss.accelerate(fuel_mass=0.6 * initial_fuel_mass, acceleration=fusion_acceleration)
  ```
* Burning $60\%$ of the initial fuel mass ($3.96 \times 10^{10}\text{ kg}$) took **~79.5 years** and left only $40\%$ ($2.64 \times 10^{10}\text{ kg}$) in the tanks.
* After cruising to the destination star, decelerating against the stellar gravity well down to the solar sail capture speed ($402.2\text{ km/s}$) required **$3.11 \times 10^{10}\text{ kg}$** of fuel.
* The burn raised `ValueError: Not enough fuel for this maneuver`, running out of propellant.

---

## 5. New Mission Plan Implementation

The updated plans were implemented in [`notebooks/Inertia_Ship.ipynb`](file:///home/mazur/linstore/repos/interstellar_mission_planning/notebooks/Inertia_Ship.ipynb) and verified by executing all cells in place with `jupyter nbconvert`.

### 5.1 Nominal Generation Ship Mission (Cell 4 & 5)
* **Phase 1: Solar Sail Launch**:
  * Unfurls at $0.01\text{ AU}$, maximum acceleration $3.0\ g$.
  * Reaches $v = 449.5\text{ km/s}$ ($1.50 \times 10^{-3}\ c$) in 2.4 days.
* **Phase 2: ~50-Year Fusion Engine Acceleration**:
  * Target velocity: $v_{\text{target}} = v_{\text{sail}} + a_{\text{fusion}} \times (50\text{ yr}) = 557.8\text{ km/s}$ ($1.86 \times 10^{-3}\ c$).
  * Duration: $50.7\text{ years}$.
  * Consumes $40.6\%$ of fuel, leaving **$3.92 \times 10^{10}\text{ kg}$ ($59.4\%$)** in reserve.
* **Phase 3: Interstellar Cruise**:
  * Cruises $4.16\text{ ly}$ across interstellar space over $2,235\text{ years}$.
  * Onboard electricity generation ($1.5 \times 10^{11}\text{ W}$) consumes only ~0.1% of remaining fuel.
* **Phase 4: Fusion Deceleration**:
  * Decelerates from $557.8\text{ km/s}$ to sail capture velocity $402.2\text{ km/s}$ at destination.
  * Consumes $2.94 \times 10^{10}\text{ kg}$ of fuel, leaving **$9.75 \times 10^{9}\text{ kg}$** of fuel safely in reserve.
* **Phase 5: Solar Sail Deceleration into Destination Orbit**:
  * Sails against destination star radiation pressure to reach $v = 0.0\text{ km/s}$ at $4.36\text{ ly}$.
* **Total Nominal Duration**: **$2,358.2\text{ years}$**.

### 5.2 Naive Mission Abort (Cells 8, 9, 10)
* Models what occurs if the crew attempts to reverse course using fusion rocket propulsion alone.
* Burning $99\%$ of remaining fuel ($3.88 \times 10^{10}\text{ kg}$) over 101 years turns the velocity around to $-339.5\text{ km/s}$, but depletes tanks to $< 1\%$ ($3.9 \times 10^8\text{ kg}$).
* The ship has zero propellant left to brake upon reaching Earth and cannot sustain life support power.

### 5.3 SWIMMER Mission Abort (Cells 11, 12, 13)
* **Configuration**:
  * Dry mass fixed: $m_{\text{payload}} = 6.6 \times 10^{10}\text{ kg}$ intact.
  * Pusher plate: radius $7.0 \times 10^5\text{ km}$, areal density $7.63 \times 10^{-9}\text{ kg/m}^2$, mass $1.17 \times 10^{10}\text{ kg}$.
  * Total ship mass: $1.21 \times 10^{11}\text{ kg}$.
  * Operating power: $P = 7.0 \times 10^{13}\text{ W}$ ($70\text{ TW}$).
* **Flight Profile**:
  1. **Abort Initiation**: At Year $50.7$, distance $0.0852\text{ ly}$ (~$5,388\text{ AU}$), velocity $+557.8\text{ km/s}$.
  2. **SWIMMER Deceleration (Braking Mode, `direction = -1`)**:
     * Duration: $29.23\text{ years}$.
     * Halts outbound motion, reaching aphelion turnaround at **Year $80.0$** at distance **$0.1030\text{ ly}$** (~$6,515\text{ AU}$) with $v = -0.54\text{ km/s}$.
  3. **SWIMMER Inbound Acceleration (`direction = -1`)**:
     * Duration: $55.933\text{ years}$.
     * Accelerates toward Earth, reaching peak return speed of **$-894.6\text{ km/s}$** ($-3.0 \times 10^{-3}\ c$) at **Year $135.9$** at distance $0.0293\text{ ly}$.
  4. **SWIMMER Deceleration into Earth Orbit (`direction = +1`)**:
     * Duration: $34.20\text{ years}$.
     * Brakes against plasma, bringing the ship to rest relative to the Sun.
     * **Arrival Time**: **Year $170.1$** (within ~170 years of launch).
     * **Arrival Position**: **$1.03\text{ AU}$** (Earth-like orbit around the Sun).
     * **Final Velocity**: **$0.15\text{ km/s}$** radial velocity.
     * **Fuel Remaining**: **$3.92 \times 10^{10}\text{ kg}$** ($59.4\%$ of initial fuel) completely intact for station keeping and habitat power.

---

## 6. Verification Status

| Check | Result | Command |
|---|---|---|
| Unit Test Suite | 85 / 85 Passing | `uv run pytest` |
| Notebook Imports | Verified | `from scimath.units...` works via Python 3.9 |
| Notebook Execution | All 15 cells executed cleanly | `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/Inertia_Ship.ipynb` |
| Nominal Mission Fuel | Passed (> 9.7B kg margin) | Cell 6 in notebook |
| SWIMMER Abort Arrival | Year 170.1 at 1.03 AU, v=0.15 km/s | Cell 12 in notebook |

---

## 7. Notebook Structure & Documentation Details

The notebook [`notebooks/Inertia_Ship.ipynb`](file:///home/mazur/linstore/repos/interstellar_mission_planning/notebooks/Inertia_Ship.ipynb) was completely rewritten to present a clean, publication-grade technical story that complements the novel:
1. **Title & Narrative Overview (Cell 0)**: Establishes the generation ship premise, multi-generational mission goals, and the tri-propulsion architecture (Solar Sail, Fusion Drive, SWIMMER).
2. **Physical Principles, Constants & Sources (Cell 1)**: Formally documents:
   - Astrodynamic constants ($G, c, g, M_\odot, L_\odot$) from CODATA and IAU standards.
   - Proxima / Alpha Centauri stellar parameters ($d = 4.244\text{ ly}, L = 0.001567\, L_\odot, M = 0.122\, M_\odot$).
   - Aneutronic $^3\text{He} + {}^2\text{H} \to {}^4\text{He} + \text{p} + 18.354\text{ MeV}$ nuclear reaction and $150\text{ GW}$ closed-loop biosphere power baseline (NIST / Wolfram Alpha).
   - Relativistic rocket mechanics ($v_e = 500\text{ km/s}$) and stellar gravity escape corrections.
   - Solar sail radiation pressure formulations (Spieth & Zubrin 1999; Mallove & Matloff 1989).
   - SWIMMER plasma momentum exchange theory (Brisbin 2018).
3. **Architecture & Parameters (Cells 3 & 4)**: Fixed dry mass justification ($6.6\times 10^{10}\text{ kg}$), 1:1 fuel ratio, and carbon nanotube sail geometry ($R = 6,371\text{ km}$).
4. **Mission Scenario 1 — Nominal Voyage (Cells 5, 6, 7)**: Step-by-step breakdown of the 5 flight phases with full trajectory logs and Matplotlib visualization.
5. **Mission Scenario 2 — The Naive Abort Dilemma (Cells 8, 9, 10)**: Demonstrates the Tsiolkovsky tyranny when attempting to reverse course on reaction mass alone.
6. **Mission Scenario 3 — The SWIMMER Resolution (Cells 11, 12, 13)**: The propellantless return journey, detailing turnaround at $0.1030\text{ ly}$, inbound sprint to $-894.6\text{ km/s}$, and smooth orbital capture at $1.03\text{ AU}$ at Year 170.1.
7. **Comparative Analysis & Summary (Cell 14)**: A synthesis table contrasting all three scenarios on propulsion, max distance, speed, flight time, destination, and fuel margins.
