###############################################################################
# Quantum Computing Tutorial (Python — Non-Executable Teaching Script)
# Author: Jared Edgerton
# Date: date.today()
#
# This script demonstrates (conceptually, without execution in class):
#   1) What a qubit is (state vectors + measurement)
#   2) Basic single-qubit gates (X, H, Z) as matrices
#   3) Multi-qubit states (tensor products)
#   4) Entanglement via CNOT (Bell state)
#   5) A minimal "circuit" representation (gate sequence)
#   6) A non-executed Qiskit example (how this maps to real tooling)
#   7) Key pitfalls (exponential state size, measurement collapse)
#
# Teaching note (important):
# - This file is intentionally written as a "hard-coded" sequential workflow.
# - No user-defined functions.
# - No conditional statements (no if/else).
# - Students should READ this script as an annotated walkthrough.
# - They are NOT expected to run it in Week __ (tooling constraints).
#
# Optional dependencies (NOT required to run this script as reading material):
#   pip install numpy qiskit
###############################################################################

# -----------------------------------------------------------------------------
# Part 0: Why quantum computing looks “weird”
# -----------------------------------------------------------------------------
# Classical bit:
# - A classical bit is either 0 or 1.
#
# Qubit:
# - A qubit is a 2D complex vector (a “state vector”) with unit length:
#
#     |ψ> = α|0> + β|1>
#
# where α and β are complex numbers and:
#
#     |α|^2 + |β|^2 = 1
#
# Measurement:
# - When you measure |ψ> in the computational basis:
#   - you see 0 with probability |α|^2
#   - you see 1 with probability |β|^2
# - After measurement, the state “collapses” to the outcome state.

# -----------------------------------------------------------------------------
# Part 1: State vectors for one qubit
# -----------------------------------------------------------------------------
# Computational basis states:
# - |0> = [1, 0]^T
# - |1> = [0, 1]^T

# (Optional) If you want to actually run matrix math, you’d use numpy:
# import numpy as np

# Hard-coded (conceptual) state vectors:
# ket0 = np.array([[1.0], [0.0]], dtype=complex)
# ket1 = np.array([[0.0], [1.0]], dtype=complex)

# Example superposition:
# - |ψ> = (1/sqrt(2))|0> + (1/sqrt(2))|1>
#
# alpha = 1/sqrt(2)
# beta  = 1/sqrt(2)

# -----------------------------------------------------------------------------
# Part 2: Single-qubit gates (unitary matrices)
# -----------------------------------------------------------------------------
# Gates are matrices that transform state vectors:
#
#     |ψ_out> = U |ψ_in>
#
# Common single-qubit gates:
#
# X (NOT gate): flips |0> <-> |1>
#     X = [[0, 1],
#          [1, 0]]
#
# Z (phase flip): leaves |0> alone, flips sign of |1>
#     Z = [[1,  0],
#          [0, -1]]
#
# H (Hadamard): creates superposition from basis states
#     H = (1/sqrt(2)) * [[ 1,  1],
#                       [ 1, -1]]
#
# Example conceptually:
# - If you apply H to |0>, you get:
#     H|0> = (1/sqrt(2))(|0> + |1>)
# - If you apply H to |1>, you get:
#     H|1> = (1/sqrt(2))(|0> - |1>)

# -----------------------------------------------------------------------------
# Part 3: Two qubits (tensor products)
# -----------------------------------------------------------------------------
# A two-qubit state lives in a 4D complex vector space.
#
# Basis states:
# - |00>, |01>, |10>, |11>
#
# Constructing two-qubit states uses the tensor (Kronecker) product:
#
#   |a b> = |a> ⊗ |b>
#
# For example:
#   |00> = |0> ⊗ |0>
#   |01> = |0> ⊗ |1>
#
# With numpy, tensor product is np.kron(a, b)

# -----------------------------------------------------------------------------
# Part 4: Entanglement with CNOT (controlled NOT)
# -----------------------------------------------------------------------------
# CNOT is a two-qubit gate:
# - The first qubit is the "control"
# - The second qubit is the "target"
# - If control is |1>, apply X to the target
# - If control is |0>, do nothing
#
# CNOT matrix in the |00>,|01>,|10>,|11> basis:
#
#     CNOT = [[1, 0, 0, 0],
#             [0, 1, 0, 0],
#             [0, 0, 0, 1],
#             [0, 0, 1, 0]]
#
# Bell state construction (classic entanglement demo):
# Step 1: Start with |00>
# Step 2: Apply H to the FIRST qubit:
#         (H ⊗ I)|00> = (1/sqrt(2))(|00> + |10>)
# Step 3: Apply CNOT (control=first, target=second):
#         CNOT * (1/sqrt(2))(|00> + |10>) = (1/sqrt(2))(|00> + |11>)
#
# The result:
#     |Φ+> = (1/sqrt(2))(|00> + |11>)
#
# This is entangled: you cannot write it as |a> ⊗ |b>.

# -----------------------------------------------------------------------------
# Part 5: Measurement intuition for entanglement
# -----------------------------------------------------------------------------
# If you measure |Φ+> in the computational basis:
# - You get 00 with probability 1/2
# - You get 11 with probability 1/2
# - You never get 01 or 10
#
# Importantly:
# - If you measure the first qubit and get 0, the second is guaranteed 0.
# - If you measure the first qubit and get 1, the second is guaranteed 1.
#
# This is correlation stronger than classical “independent random bits.”

# -----------------------------------------------------------------------------
# Part 6: A minimal “circuit” view (gate sequence)
# -----------------------------------------------------------------------------
# In real quantum programming, you usually write circuits:
# - a circuit is a sequence of gate operations on qubits
#
# A Bell state circuit:
#   1) H on qubit 0
#   2) CNOT with control qubit 0, target qubit 1
#   3) Measure both qubits
#
# You can represent that as a list of steps:
#
# circuit = [
#   ("H", 0),
#   ("CNOT", 0, 1),
#   ("MEASURE", 0),
#   ("MEASURE", 1),
# ]
#
# This is NOT executable by itself; it’s a conceptual representation.

# -----------------------------------------------------------------------------
# Part 7: Non-executed Qiskit example (how this looks in real code)
# -----------------------------------------------------------------------------
# Qiskit is a common Python library for quantum circuits.
# Your students may not be able to run this in class, but this shows the mapping.
#
# To install (outside class):
#   pip install qiskit qiskit-aer
#
# Example (DO NOT RUN in class):
#
# from qiskit import QuantumCircuit
# from qiskit_aer import AerSimulator
#
# qc = QuantumCircuit(2, 2)     # 2 qubits, 2 classical bits
# qc.h(0)                      # H gate on qubit 0
# qc.cx(0, 1)                  # CNOT: control 0, target 1
# qc.measure(0, 0)             # measure qubit 0 -> classical bit 0
# qc.measure(1, 1)             # measure qubit 1 -> classical bit 1
#
# sim = AerSimulator()
# result = sim.run(qc, shots=1000).result()
# counts = result.get_counts(qc)
# print(counts)
#
# Typical output:
# - about half "00" and half "11"
# - very few or none "01" / "10" (depending on noise)

# -----------------------------------------------------------------------------
# Part 8: What makes quantum computing hard (key pitfalls)
# -----------------------------------------------------------------------------
# 1) Exponential state growth:
# - n qubits require 2^n complex amplitudes.
# - 30 qubits already means ~1 billion amplitudes (too big for many laptops).
#
# 2) Measurement collapse:
# - You cannot “peek” at a quantum state without changing it.
#
# 3) Noise and decoherence:
# - Real hardware introduces errors; ideal math does not.
#
# 4) Algorithm design:
# - Quantum advantage requires careful algorithms (e.g., Grover, Shor).
# - Many problems do not get speedups.

# -----------------------------------------------------------------------------
# Practice Tasks (Students)
# -----------------------------------------------------------------------------
# 1) Conceptual:
# - Explain in words why H|0> produces a 50/50 measurement result.
#
# 2) Gate reasoning:
# - What does X do to (α|0> + β|1>)?
# - What does Z do to (α|0> + β|1>)?
#
# 3) Bell state:
# - Write out the algebra showing how H and CNOT create (|00> + |11>)/sqrt(2).
#
# 4) “Sampling view”:
# - If you run the Bell circuit for 1000 shots, why do you expect ~500 "00" and
#   ~500 "11" (not exactly 500/500)?
#
# 5) Extension (optional, outside class):
# - Install qiskit locally and run the Bell example.
# - Modify the circuit to create the other Bell states (Φ-, Ψ+, Ψ-).
###############################################################################

# -----------------------------------------------------------------------------
# Where to buy quantum computing time (cloud access)
# -----------------------------------------------------------------------------
# If you want to actually run circuits on real quantum hardware (or paid simulators),
# these are the main places people buy compute time / access today:
#
# 1) IBM Quantum Platform (IBM)
#    - Offers a free “open” tier plus paid options (pay-as-you-go, pre-purchased time,
#      annual subscription, and dedicated systems).
#    - Typical workflow: write circuits in Qiskit → submit jobs to IBM backends.
#
# 2) Amazon Braket (AWS)
#    - Managed service with pay-per-task / pay-per-shot pricing (varies by provider).
#    - Provides access to multiple hardware types/providers through one interface.
#
# 3) Azure Quantum (Microsoft)
#    - Marketplace-style access to multiple quantum hardware providers under Azure billing.
#    - Typical workflow: submit jobs through Azure Quantum provider targets.
#
# 4) D-Wave Leap (D-Wave)
#    - Cloud access focused on quantum annealing + hybrid solvers (optimization workloads).
#
# 5) IonQ Quantum Cloud (IonQ direct)
#    - Direct cloud access models (on-demand and reserved-time options).
#
# 6) Quantinuum (direct subscription)
#    - Subscription-based access to Quantinuum trapped-ion systems.
#
# 7) Rigetti Quantum Cloud Services (Rigetti QCS)
#    - Rigetti’s direct cloud access platform (in addition to access via larger clouds).
#
# Teaching note:
# - “Buying time” usually means either (a) per-job/per-shot usage billing, or (b) reserved
#   QPU time blocks/subscriptions. Pricing and availability change, so always check the
#   current pricing pages.
