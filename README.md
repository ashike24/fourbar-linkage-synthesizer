# ⚙️ Four-Bar Linkage Path Synthesizer

> **Try it live → [fourbar-linkage-synthesize.streamlit.app](https://fourbar-linkage-synthesize.streamlit.app/)**

Given 4 points in a plane, this app finds a four-bar linkage mechanism whose coupler point passes through all of them, verifies the Grashof condition, and animates the mechanism.

---

## What is a four-bar linkage?

A four-bar linkage is one of the most fundamental mechanisms in mechanical engineering. It has four rigid links connected by revolute (pin) joints:

```
E --[EG]-- G --[GH]-- H --[HF]-- F
                |
                I   (coupler tracer point)
```

- **E and F** are fixed to the ground (they don't move)
- **G and H** are moving joints
- **I** is a point attached to the coupler link GH — it traces a curve as the mechanism rotates

This app solves the inverse problem: given the curve you want (4 points on it), find the linkage that draws it.

---

## The app

**Live at → [fourbar-linkage-synthesize.streamlit.app](https://fourbar-linkage-synthesize.streamlit.app/)**

### How to use it

1. Enter the x, y coordinates of your 4 desired coupler-point positions (A, B, C, D) in the sidebar
2. Adjust solver settings if needed (more restarts = better accuracy, slower)
3. Hit **Synthesize & Animate**
4. The app checks the Grashof condition and either returns the 9 linkage parameters with an animation, or tells you exactly why it failed
5. Download the GIF if you want to keep it

### Inputs and outputs

| | Variables | Count |
|---|---|---|
| Input | x, y coordinates of A, B, C, D | 8 |
| Output | x_E, y_E, x_F, y_F, EG, GI, IH, HG, HF | 9 |

---

## Grashof condition

After synthesis the app checks whether the result satisfies the Grashof condition:

> **Shortest link + Longest link   ≤   Sum of the other two links**

If it fails, the app shows you exactly which links caused it and by how much, and retries automatically with a different random seed (up to a configurable number of attempts) before giving up.

---

## How the synthesis works

### The math

The linkage has 9 unknowns. Each prescribed point contributes 2 equations (x and y) but also 1 hidden variable — the crank angle θ at that position — giving a net of 1 equation per point. With 4 points we have 4 equations for 9 unknowns. The system is underdetermined, meaning infinitely many linkages can pass through the same 4 points. The solver finds one good member of that family.

The full optimization vector is 13 variables:

```
[ x_E, y_E, x_F, y_F, EG, HF, GH, px, py, θ_A, θ_B, θ_C, θ_D ]
```

The 8 residuals are the x and y errors between the computed coupler point I(θ_k) and each prescribed point P_k. scipy.optimize.least_squares (Trust Region Reflective method) minimizes the sum of squared residuals, with many random restarts to avoid local minima.

### Forward kinematics

For a given crank angle θ₂:

1. **G** = E + EG · [cos θ₂, sin θ₂]
2. Solve triangle G–H–F by the law of cosines to find **H**
3. Coupler angle θ₃ = atan2(H − G)
4. **I** = G + R(θ₃) · [px, py]   where R is a 2D rotation matrix

### Accuracy

Verified on 24 known ground-truth linkages. RMS error consistently below 1×10⁻¹⁰ — effectively machine precision.

---

## Run locally

```bash
git clone https://github.com/<your-username>/fourbar-linkage-synthesizer.git
cd fourbar-linkage-synthesizer
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Requires Python 3.9+.

---

## Project structure

```
├── app.py            # Streamlit app — UI, synthesis, Grashof check, animation
├── requirements.txt  # Python dependencies
└── README.md
```

Everything lives in `app.py`. The forward kinematics, optimizer, Grashof checker, animation builder, and Streamlit UI are all in one file for easy deployment.

---

## Known limitations

- **Non-Grashof linkages** — if no Grashof-satisfying solution is found after all retries, the app reports the failure with a detailed explanation. Try placing the points closer together or increasing the retry count.
- **Solution uniqueness** — many different linkages can pass through the same 4 points. The app returns one valid solution. A different random seed may give a different equally valid one.
- **Branch and circuit defects** — the current implementation does not yet check for branch or circuit defects. A synthesised linkage may require disassembly to reach all 4 positions in practice.
- **Computation time** — synthesis takes 10–30 seconds depending on restarts. Reduce restarts for speed, increase for quality.

---

## License

MIT — do whatever you want with it.
