# ⚙️ Four-Bar Linkage Path Synthesizer

**App link : [4-point-synthesis.streamlit.app](https://4-point-synthesis.streamlit.app/)**

**GitHub link : [github.com/ashike24/fourbar-linkage-synthesizer](https://github.com/ashike24/fourbar-linkage-synthesizer)**

---

## What is a Four-Bar Linkage?

A four-bar linkage is one of the simplest and most widely used mechanisms in mechanical engineering. It appears in everyday objects — car suspension systems, windshield wipers, robotic arms, prosthetic limbs, and industrial machinery. The mechanism consists of four rigid links connected by four revolute (pin) joints.

```
    E ────[EG]──── G ────[GH]──── H ────[HF]──── F
  (fixed)       (moving)        (moving)        (fixed)
                     │
                     I   ← coupler tracer point
```

### The four links

| Link | Name | Description |
|------|------|-------------|
| EG | Input crank | Rotates about fixed pivot E |
| GH | Coupler | Connects the two cranks — tracer point I is on this link |
| HF | Output crank | Rotates about fixed pivot F |
| EF | Ground link | The fixed frame connecting E and F |

### The five key points

| Point | Type | Description |
|-------|------|-------------|
| E | Fixed pivot | One ground support — does not move |
| F | Fixed pivot | Other ground support — does not move |
| G | Moving pivot | Junction of input crank EG and coupler GH |
| H | Moving pivot | Junction of coupler GH and output crank HF |
| I | Tracer point | Attached to coupler GH — traces the coupler curve |

As the input crank EG rotates, point I traces a **coupler curve** — a complex path that can have loops, figure-eights, and many other shapes depending on the link lengths and where I is placed on the coupler.

---

## The Problem This App Solves

### Forward problem (easy)
Given a complete linkage (all link lengths, pivot positions, crank angle) → compute where point I is. This is straightforward geometry.

### Inverse problem (what this app does)
Given 4 desired positions of point I (called A, B, C, D) → find the linkage parameters that make I pass through all four positions. This is called **path synthesis** and is a classical problem in kinematic synthesis of mechanisms.

This is the hard direction. There are infinitely many linkages that can pass through any 4 given points. The app finds one good Grashof-satisfying solution from that infinite family.

---

## Physics and Geometry Behind It

### Forward kinematics — how positions are computed

Given crank angle θ₂ and all linkage parameters:

**Step 1 — Find G (moving pivot on input crank):**
```
G = E + EG × [cos θ₂,  sin θ₂]
```

**Step 2 — Find H (moving pivot on output crank):**

The triangle G–H–F must close. Using the law of cosines:
```
d   = |F − G|                              (distance between G and F)
a   = (GH² − HF² + d²) / (2d)             (foot of altitude)
h   = √(GH² − a²)                         (altitude height)
H   = G + a·(F−G)/d  ±  h·perp(F−G)/d    (two possible branches)
```
If d > GH + HF or d < |GH − HF|, the linkage cannot close — it returns None.

**Step 3 — Find I (coupler tracer point):**
```
θ₃  = atan2(H − G)              (coupler frame angle)
I   = G + R(θ₃) · [px, py]     (rotate offset vector into world frame)
```
where R(θ₃) is a 2×2 rotation matrix and (px, py) is the fixed offset of I from G in the coupler frame.

### Assembly branch

The equation for H has two solutions (the ± in Step 2). These are called the two **assembly branches**. This app always picks the same branch (the minus sign). In a physical linkage, switching branches requires disassembly — this is called a **branch defect**.

### Grashof condition

For a linkage to have at least one link capable of making a full 360° rotation, it must satisfy the Grashof condition:

```
s + l  ≤  p + q
```

where s is the shortest link, l is the longest link, and p, q are the other two (out of EG, GH, HF, EF). If this condition is satisfied, the shortest link can rotate fully. If not, all links merely oscillate — the mechanism rocks back and forth but never completes a full revolution. Since our synthesis requires the crank to visit all 4 positions in one continuous rotation, Grashof must be satisfied.

---

## The Synthesis Method

### Problem formulation

**Inputs (8 values):** x and y coordinates of points A, B, C, D

**Unknowns (9 linkage parameters):**
```
x_E,  y_E  — position of fixed pivot E
x_F,  y_F  — position of fixed pivot F
EG         — input crank length
HF         — output crank length
GH         — coupler link length
px,  py    — offset of tracer point I from G in coupler frame
```

**Hidden variables (4 values):**
```
θ_A,  θ_B,  θ_C,  θ_D  — input crank angle at each prescribed position
```

Each prescribed point gives 2 equations (x and y must match) but introduces 1 hidden variable (the crank angle at that position). Net contribution per point: 2 − 1 = **1 equation**. With 4 points: **4 equations for 9 unknowns** — the system is underdetermined. Infinitely many solutions exist.

### Optimization vector

The code expands to 13 variables by including the crank angles directly:

```
params = [ x_E,  y_E,  x_F,  y_F,  EG,  HF,  GH,  px,  py,
           θ_A,  θ_B,  θ_C,  θ_D ]
```

### Residual function

For each prescribed position k ∈ {A, B, C, D}:
```
res_x_k  =  I_x(θ_k)  −  P_k_x
res_y_k  =  I_y(θ_k)  −  P_k_y
```

This gives **8 residuals** total. The optimizer minimizes:
```
cost  =  Σ  (res_x_k²  +  res_y_k²)   for k = A, B, C, D
```

When cost = 0, the coupler point I passes exactly through all 4 prescribed positions.

### Pipeline

```
User inputs A, B, C, D
        │
        ▼
Compute centroid and spread of input points
        │
        ▼
Generate random starting point (pivot positions + link lengths + crank angles)
        │
        ▼
Run TRF least_squares optimizer
(minimise sum of squared positional errors)
        │
        ▼
Keep best solution found across all restarts
        │
        ▼
Check RMS error  ── high? ──► discard, try next restart
        │ low
        ▼
Check Grashof condition
        │
   ┌────┴────┐
   │         │
 Pass       Fail ──► retry with different seed (up to 8 times)
   │                         │
   ▼                    All retries fail
Show results,                │
animation, GIF          Show detailed
                        failure message
```

### Algorithm: Trust Region Reflective (TRF)

The optimizer used is `scipy.optimize.least_squares` with `method='trf'`.

TRF is a **deterministic gradient-based** algorithm. At each iteration:
1. Computes the Jacobian (gradient of residuals w.r.t. parameters)
2. Defines a **trust region** — a sphere around the current point where the local linear model is reliable
3. Solves a constrained subproblem inside that sphere
4. If the step improved the cost, expand the trust region; otherwise, shrink it and try again
5. Handles **box constraints** natively — parameters stay within their bounds throughout

TRF is not heuristic — every step is a principled mathematical decision. However, it can only find a **local minimum**. This is why multiple random restarts are needed.

### Multistart strategy (heuristic)

To escape local minima, the code runs TRF from **250 different random starting points** per synthesis attempt. Starting points are sampled by:
- Picking a random direction and distance from the centroid of the input points for pivot placement
- Randomizing link lengths proportional to the spread of the input points
- Randomizing crank angles uniformly in [−π, π]

The best result across all restarts (lowest cost) is kept.

### Grashof retry

After each synthesis attempt, the Grashof condition is checked. If it fails, the entire synthesis is repeated with a shifted random seed (seed + 17 × attempt). Up to **8 attempts** are made before reporting failure. This works because the problem is underdetermined — a different starting region may find a different solution family that satisfies Grashof.

---

## Error Calculation

### Residuals

The raw error signal is:
```
residual_k  =  I(θ_k)  −  P_k        (2D vector, one per prescribed point)
```

### Cost (sum of squared residuals)

```
cost  =  0.5 × Σ |residual_k|²
```

This is what the TRF optimizer minimizes directly.

### RMS error (reported to user)

```
RMS  =  √( cost × 2 / 4 )  =  √( mean of squared per-point distances )
```

This gives an average per-point distance in the same units as the input coordinates. A value below 1.0 is considered acceptable. Values below 1×10⁻⁶ indicate near-perfect synthesis.

### Verification

After synthesis, the code independently re-evaluates forward kinematics at each synthesized crank angle and measures:
```
error_k  =  |I(θ_k)  −  P_k|    (Euclidean distance)
```

In verified tests on 24 known ground-truth linkages, all errors were below **1×10⁻¹⁰** — effectively machine precision.

---

## User Inputs

The app accepts only the coordinates of the 4 prescribed positions:

| Input | Type | Description |
|-------|------|-------------|
| x_A, y_A | Float | Coordinates of prescribed position A |
| x_B, y_B | Float | Coordinates of prescribed position B |
| x_C, y_C | Float | Coordinates of prescribed position C |
| x_D, y_D | Float | Coordinates of prescribed position D |

**Any unit system is valid** — millimetres, centimetres, inches, pixels. The output link lengths will be in the same units.

**Default example values:**

| Point | x | y |
|-------|-----|-----|
| A | 200 | 180 |
| B | 370 | 140 |
| C | 490 | 230 |
| D | 310 | 310 |

All solver settings (restarts, retries, FPS, seed) are hidden and pre-tuned for best results.

---

## Output Parameters

The app returns 9 linkage parameters:

| Output | Description |
|--------|-------------|
| x_E | x-coordinate of fixed ground pivot E |
| y_E | y-coordinate of fixed ground pivot E |
| x_F | x-coordinate of fixed ground pivot F |
| y_F | y-coordinate of fixed ground pivot F |
| EG | Length of input crank (E to G) |
| GI | Distance from moving pivot G to tracer point I |
| IH | Distance from tracer point I to moving pivot H |
| HG | Length of coupler link (H to G) |
| HF | Length of output crank (H to F) |

Along with:
- Grashof condition verification with detailed link-by-link breakdown
- Per-point verification errors (should be ~1×10⁻¹⁰ for a good solution)
- Crank angles θ_A, θ_B, θ_C, θ_D at each prescribed position (in radians and degrees)
- Animated GIF of the mechanism in motion

---

## How to Use the App

1. Open [4-point-synthesis.streamlit.app](https://4-point-synthesis.streamlit.app/)
2. In the sidebar, enter the x and y coordinates for each of the 4 points (A, B, C, D)
3. Click the **Synthesize & Animate** button
4. Wait 15–40 seconds while the solver runs
5. If Grashof is satisfied, the app shows:
   - A table of the 9 output parameters
   - Grashof condition verification
   - Per-point error table
   - Crank angles at A, B, C, D
   - An animated GIF of the mechanism
   - A download button for the GIF
6. If Grashof fails, the app explains exactly which links caused the failure and by how much, with suggestions for how to adjust the input points

---

## When the App Will Report Failure

The Grashof condition will typically fail when:
- One point is very far from the other three (e.g., A, B, C clustered near origin and D at (600, 600))
- The four points require an extremely long ground link EF relative to the cranks
- The point geometry forces a non-rotating linkage

**Example of failing input:**

| Point | x | y |
|-------|-----|-----|
| A | 100 | 100 |
| B | 102 | 100 |
| C | 100 | 102 |
| D | 600 | 600 |

Three points are clustered together and one is far away — the optimizer is forced to place E and F far apart, making EF the longest link and violating Grashof.

---

## Run Locally

```bash
git clone https://github.com/ashike24/fourbar-linkage-synthesizer.git
cd fourbar-linkage-synthesizer
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires Python 3.9+.

---

## Project Structure

```
├── app.py            # Complete Streamlit app — kinematics, synthesis,
│                     # Grashof check, animation, UI
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web app framework |
| numpy | Numerical computation |
| scipy | TRF optimizer (least_squares) |
| matplotlib | Animation rendering |
| pillow | GIF encoding |
| pandas | Results tables |

---

## Known Limitations

**Branch defects** — the code always selects the same assembly branch. If two prescribed points lie on different branches, the physical mechanism would need to be disassembled to move between them. Branch defect detection is not yet implemented.

**Circuit defects** — a coupler curve can have multiple separate loops. Points on different loops require the crank to pass through a singular (locked) position, which is physically impossible. Circuit defect detection is not yet implemented.

**Solution uniqueness** — the problem is underdetermined. Many valid linkages pass through the same 4 points. The app returns one Grashof-satisfying solution. A different run may return a different but equally valid mechanism.

**Computation time** — synthesis runs 250 random restarts per attempt with up to 8 Grashof retries. Typical time is 15–40 seconds. In worst case it can take a few minutes.

---

## License

MIT — free to use, modify, and distribute.
