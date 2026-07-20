# Validated SU(2) character-tail / exact-RG prototype

All intervals below are derived from positive rational series; no floating-point
value is used to justify an enclosure.

## Input

- Wilson parameter: beta = 11/5 = 2.2
- Normalized plaquette weight: exp(beta(cos(theta)-1))
- A 2x2x2x2 fine-cell contains 216 elementary plaquettes.

## Initial Peter-Weyl tail

| cutoff n_max | tail tau_N | 216 tau_N |
|---:|---:|---:|
| 6 | [0.002677892949651685, 0.002677892949651686] | [0.578424877124764100, 0.578424877124764101] |
| 8 | [0.000068908927820552, 0.000068908927820553] | [0.014884328409239324, 0.014884328409239325] |
| 10 | [0.000001078814460529, 0.000001078814460530] | [0.000233023923474469, 0.000233023923474470] |
| 12 | [1.1303940764E-8, 1.1303940765E-8] | [0.000002441651205228, 0.000002441651205229] |
| 14 | [8.4628276E-11, 8.4628277E-11] | [1.8279707767E-8, 1.8279707768E-8] |
| 16 | [4.74661E-13, 4.74662E-13] | [1.02526987E-10, 1.02526988E-10] |

The 216 tau_N column is the rigorous telescoping-product bound for replacing all 216 normalized plaquette weights in one 2^4 cell by their n<=N truncations.

## Exact 2D 2x2 RG benchmark

For two-dimensional pure SU(2) gauge theory, normalized Fourier ratios obey the exact recurrence r_n' = r_n^4.

### irrep dimension n=2

- step 0: [0.464479025270420310654070, 0.464479025270420310654071]
- step 1: [0.046544077646609705462415, 0.046544077646609705462416]
- step 2: [0.000004693077365649915651, 0.000004693077365649915652]
- step 3: [4.85E-22, 4.86E-22]

### irrep dimension n=3

- step 0: [0.155492681326508526083508, 0.155492681326508526083509]
- step 1: [0.000584574424138635342548, 0.000584574424138635342549]
- step 2: [1.16777518420E-13, 1.16777518421E-13]
- step 3: [0E-24, 1E-24]

### irrep dimension n=4

- step 0: [0.040408076198124330426319, 0.040408076198124330426320]
- step 1: [0.000002666077058671658747, 0.000002666077058671658748]
- step 2: [5.0E-23, 5.1E-23]
- step 3: [0E-24, 1E-24]

## Casimir-gradient tail

The bound uses ||grad chi_j||_infty <= (2j+1)^2/2.

| cutoff n_max | derivative tail | 6 x tail | 216 x tail |
|---:|---:|---:|---:|
| 6 | [0E-18, 0.009776809377036234] | [0E-18, 0.058660856262217400] | [0E-18, 2.111790825439826386] |
| 8 | [0E-18, 0.000317113428056905] | [0E-18, 0.001902680568341429] | [0E-18, 0.068496500460291409] |
| 10 | [0E-18, 0.000006015764950566] | [0E-18, 0.000036094589703395] | [0E-18, 0.001299405229322208] |
| 12 | [0E-18, 7.4161320899E-8] | [0E-18, 4.44967925394E-7] | [0E-18, 0.000016018845314179] |
| 14 | [0E-18, 6.38962130E-10] | [0E-18, 3.833772775E-9] | [0E-18, 1.38015819894E-7] |
| 16 | [0E-18, 4.054931E-12] | [0E-18, 2.4329586E-11] | [0E-18, 8.75865068E-10] |

The factor 6 applies to differentiation with respect to one fine link in 4D,
because six plaquettes meet that link.  The factor 216 is a deliberately
coarse bound for a variation acting on every plaquette in the 2^4 cell.

## Status

- The initial 4D representation truncation is rigorously enclosed.
- The finite-step RG trajectory is rigorously executed in 2D, where the exact convolution law closes on character coefficients.
- A genuine 4D step still requires interval contraction of the reduced representation/fusion tensor (armillary tensor); the scalar tail estimate alone does not enclose the generated multi-loop interactions.