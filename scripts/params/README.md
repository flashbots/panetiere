# Parameter generation for Panetière

`generate_params.py` computes concrete parameters for Panetière and checks them against the conditions of the theorems in the paper. It has two modes:

- **Generate** (default): search for the parameter set that minimizes recipient ingress for a given setting (number of clients ρ, number of servers γ, message size, failure probabilities), check it, and optionally write it as JSON.
- **Check** (`--check-only FILE`): read a parameter set from a JSON file and check it, without searching. Use this to verify parameters chosen elsewhere, e.g. in an implementation.

The other files in this directory:

| File | Role |
|---|---|
| `params.schema.json` | JSON Schema of the parameter files (written by `--json`, read by `--check-only`) |
| `iblt_common.py`, `iblt_empirical.py` | IBLT shape search and simulation, imported by `generate_params.py` |
| `iblt_extrapolate.py`, `stacked_iblt.py` | Separate experiments, not used by `generate_params.py` |
| `estimator/` | The lattice estimator, only needed for `--confirm-with-estimator` |

## Requirements

- Python 3 with `mpmath`, `sympy` and `tabulate` (`pip install mpmath sympy tabulate`).
- `numpy` (optional) makes the IBLT shape search much faster for large ρ.
- `jsonschema` is needed for `--check-only` only (`pip install jsonschema`). Writing JSON does not need it.
- A C compiler (optional) speeds up IBLT simulations (`--empirical`); without one, the script falls back to a slow pure-Python simulator.
- Sage is needed for `--confirm-with-estimator` only. Run the script as `sage -python generate_params.py ...` from this directory so that `estimator` can be imported.

## Generating parameters

```sh
python3 generate_params.py                                  # defaults
python3 generate_params.py --rho 1000 --gamma 16 --msg-bytes 1024 \
    --json params.json
```

The defaults are ρ = 300, γ = 8, n = 2048, λ = 128, 16 KiB messages, ε_KAHE = 100, ε_MSE = 40 and the Newton–Girard (NG) multiset encoding. `--help` lists all options.

The script prints a table of all conditions with their status, the concrete statistical terms (e.g. the KAHE privacy and VC hiding losses), and the final parameters with their sizes. Every condition is re-derived from the final parameters by `check_all()`, independently of the search code, using exact integer arithmetic or rigorous interval arithmetic.

Options worth knowing:

- `--mse IBLT` uses an invertible Bloom lookup table instead of NG. By default the IBLT shape is the smallest one satisfying the provable bound of the paper.
- `--empirical` (IBLT only) also searches for a smaller shape by simulation and uses it if it has fewer cells. Such a shape is not covered by the paper's bound; its condition is then reported as `EMPIRICAL`. Simulation needs about 10 · 2^ε_MSE trials per shape, so it is only feasible up to about ε_MSE = 25. The simulation seed is random unless `--seed` is set; the seed used is printed.
- `--confirm-with-estimator` re-checks every hardness assumption of the final parameters with the lattice estimator. The search itself only uses built-in core-SVP estimates (ports of the estimator's primal uSVP estimate for RLWE and of the Dilithium-style MSIS analysis for SIS).
- `--log-queries` sets the bound Q = 2^log_queries on the adversary's random-oracle queries (default 2^64).
- `--max-q-bits` bounds all moduli searched over (default 64 bits, i.e. one machine word).
- `--capacity C` sets the capacity of the multiset encoding to C ≤ ρ (default C = ρ). Use it when clients without a message send cover traffic (Section `sec:cover` in the paper): ρ is then the number of clients and C the maximum number of senders. All aggregation bounds (KAHE smoothing σ, the key bound ν, the bound on q, the leaf commitment and Chipmunk) stay with respect to ρ, while the encoding (its size, the IBLT shape, ω and the condition t > C) only depends on C. The cost is guaranteed output delivery: it only holds against at most C − h corrupted clients, where h is the number of honest senders, since the adversary can make all corrupted clients send. Choosing C so that the number of senders stays below it is up to the caller.
- `--t-ntt-log K` chooses the plaintext modulus as the smallest prime t > 2^b with t ≡ 1 mod 2^K (default K = 20; `0` takes the smallest prime above 2^b). The paper does not need this, but it lets an implementation use an NTT for polynomial arithmetic over Z_t (e.g., root finding for NG) and build F_{t^ω} as Z_t[Y]/(Y^ω − c). It costs about 0.01 bits per chunk and leaves all other parameters unchanged in practice. The output reports the 2-adic valuation of t − 1.

## Checking a parameter set

```sh
python3 generate_params.py --check-only params.json
```

This validates `params.json` against `params.schema.json`, runs all checks, and prints the condition table and the statistical terms (but not the parameter table). The exit status is:

| Status | Meaning |
|---|---|
| 0 | all checks passed |
| 1 | at least one condition fails (lines marked `FAILS`), or a required tool is missing |
| 2 | the file cannot be read, does not match the schema, or is inconsistent |

All generation options are ignored, since the setting is taken from the file. The only options that apply are `--log-queries`, `--empirical`, `--sim-*`, `--seed` and `--confirm-with-estimator`. With `--empirical`, an IBLT shape that misses the provable bound is checked by simulation instead (status `EMPIRICAL` if it passes).

### The parameter file

A file written by `--json` looks as follows. `generated` and `command` are optional and are not checked, and `setting.capacity` is only present for a capacity C < ρ (`--capacity`); all other fields are required, and unknown fields are rejected.

```json
{
  "generated": "2026-11-05T11:36:18",
  "command": "generate_params.py --rho 100 --msg-bytes 20 --eps-mse 100 --mse IBLT --json params.json",
  "setting": {
    "rho": 100, "gamma": 8, "n": 2048, "secpar": 128,
    "msg_bytes": 20, "eps_kahe": 100, "eps_mse": 100, "mse": "IBLT"
  },
  "kahe": {
    "t": 1073741827, "q": 1979283251201, "mu": 6, "sigma": 6.51, "nu": 949,
    "mse": {"theta": 24, "delta": 28, "omega": 4, "xi": 13, "chunk_bits": 30, "xi_nonce": 7}
  },
  "leaf": {"p": 38149, "beta": 69, "beta_agg": 6900, "kappa": 5, "mu": 1},
  "chipmunk": {"q": 65537, "eta": 128, "kappa": 2, "kappa_prime": 2,
               "beta_agg": 12800, "tau": 3, "xi": 2}
}
```

`params.schema.json` describes every field. In short:

- `setting` is the setting the parameters are for. `setting.mse` (`"IBLT"` or `"NG"`) determines the fields of `kahe.mse`:
  - IBLT: `theta` (rows), `delta` (cells per row), `omega` (key length), `xi` (element length), `chunk_bits`, `xi_nonce`.
  - NG: `rho` (capacity, must equal `setting.capacity`, or `setting.rho` if it is omitted), `omega` (extension degree), `xi`, `chunk_bits`, `xi_nonce`.
- `kahe` holds the KAHE moduli t and q, the number of ring elements µ per ciphertext, the Gaussian parameter σ and the key norm bound ν.
- `leaf` holds the leaf commitment: the modulus p (also the Shamir field and the VC message space R_p), the norm bounds β and β_agg, and the randomness and message lengths κ and µ.
- `chipmunk` holds the Chipmunk vector commitment: the modulus q, the decomposition parameter η (base 2η+1), the limb counts κ and κ', the norm bound β_agg, the tree depth τ and the leaf length ξ.

All values are JSON integers (`5.0` is rejected), except `sigma`, which must be a positive multiple of 0.01. Two consistency conditions cannot be expressed in the schema and are checked by the script: σ is a multiple of 0.01, and `chipmunk.tau` is the smallest τ with 2^τ ≥ γ.

## Verifying parameters used elsewhere

To check whether parameters used somewhere else (an implementation, another paper, a configuration file) are covered by the paper's theorems, write them as a parameter file and run `--check-only`. The steps below are written so that an agent can follow them as well.

1. **Collect the setting.** Determine ρ (the maximum number of clients per session), the MSE capacity C if clients without a message send cover traffic (`setting.capacity`; omit it otherwise), γ (the number of servers), the ring dimension n, the security parameter λ, the message size in bytes, the target failure probabilities 2^-ε_KAHE and 2^-ε_MSE, and the multiset encoding. If the target ε are not stated, use the values the deployment needs; the checks tell you whether the parameters achieve them.
2. **Collect the parameters** of the KAHE scheme, the multiset encoding, the leaf commitment and the Chipmunk vector commitment, and map each one to the field of the same meaning in `params.schema.json`. Mind the conventions:
   - Use the paper's notation, not the implementation's variable names. When in doubt, compare with the field descriptions in the schema and the parameter table printed by a generation run.
   - σ is the parameter of the paper's discrete Gaussian D_σ, which samples each coefficient x ∈ ℤ with probability proportional to exp(−x²/(2σ²)). For σ ≥ 1, σ equals the standard deviation up to a relative error below 10⁻⁶, so an implementation's standard deviation can be used directly. Implementations that use the width s with exp(−πx²/s²) instead need σ = s/√(2π).
   - The script only checks σ that are multiples of 0.01. If an implementation uses another σ, do not round it, since neither direction is conservative (a larger σ helps hardness but hurts correctness); report it instead.
   - Several fields are determined by others: `chipmunk.xi` = `leaf.mu` + 1, `chipmunk.kappa` = ⌈log_{2η+1} q⌉, `chipmunk.kappa_prime` = ⌈log_{2η+1} p⌉, and for NG `kahe.mse.rho` = `setting.capacity` (`setting.rho` without cover traffic). If an implementation's values differ, record them as they are; the checks will flag the inconsistency.
   - Fields with no counterpart in the implementation mean it deviates from the paper's construction. Do not invent values for them; report the deviation instead.
3. **Write the JSON file** and run
   ```sh
   python3 generate_params.py --check-only params.json
   ```
   Fix any schema errors (exit status 2) first; they are listed with their location in the file, e.g. `kahe.mse: 'rho' is a required property`.
4. **Read the result.** Exit status 0 means every condition of the paper holds for these parameters. With status 1, the lines marked `FAILS` name the violated conditions; the source column (`KAHE`, `MSE`, `NG`, `ABC`, `LEAF`, `AVC`) refers to the tags in the comment block at the top of `generate_params.py`, which name the theorem or lemma each condition comes from.
5. **Optionally confirm hardness** with the lattice estimator: `sage -python generate_params.py --check-only params.json --confirm-with-estimator`.

Passing the checks means the parameters satisfy the conditions of the paper's theorems. It does not mean that an implementation is secure: the implementation must also follow the paper's construction, and the checks do not cover components outside these parameters, such as the PKE, the signatures or the PRF. Ask a human before concluding more than that.
