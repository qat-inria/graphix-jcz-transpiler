"""Tests for transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2026, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from graphix import instruction
from graphix.branch_selector import ConstBranchSelector
from graphix.fundamentals import ANGLE_PI, Axis
from graphix.instruction import CCX, CNOT, CZ
from graphix.measurements import BlochMeasurement, Measurement
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec
from graphix.simulator import DefaultMeasureMethod
from graphix.transpiler import Circuit
from numpy.random import PCG64, Generator

from graphix_jcz_transpiler import (
    circuit_to_causal_flow,
    transpile_jcz,
    transpile_jcz_cf,
)
from graphix_jcz_transpiler.jcz_transpiler import decompose_ccx, normalize_angle

logger = logging.getLogger(__name__)

TEST_BASIC_CIRCUITS = [
    Circuit(1, instr=[instruction.H(0)]),
    Circuit(1, instr=[instruction.S(0)]),
    Circuit(1, instr=[instruction.X(0)]),
    Circuit(1, instr=[instruction.Y(0)]),
    Circuit(1, instr=[instruction.Z(0)]),
    Circuit(1, instr=[instruction.I(0)]),
    Circuit(1, instr=[instruction.RX(0, ANGLE_PI / 4)]),
    Circuit(1, instr=[instruction.RY(0, ANGLE_PI / 4)]),
    Circuit(1, instr=[instruction.RZ(0, ANGLE_PI / 4)]),
    Circuit(2, instr=[instruction.CZ((0, 1))]),
    Circuit(2, instr=[instruction.CNOT(0, 1)]),
    Circuit(3, instr=[instruction.CCX(0, (1, 2))]),
    Circuit(2, instr=[instruction.RZZ(0, 1, ANGLE_PI / 4)]),
]


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation(circuit: Circuit, fx_rng: Generator) -> None:
    """Test circuit transpilation simulation matches direct simulation of the circuit."""
    pattern = transpile_jcz(circuit).pattern
    pattern = pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(rng=fx_rng)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_check_circuit_flow(circuit: Circuit) -> None:
    """Test directly transpiled basic circuits have flow."""
    pattern = transpile_jcz(circuit).pattern
    og = pattern.extract_opengraph()
    f = og.to_bloch().find_causal_flow()
    assert f is not None


@pytest.mark.parametrize("axis", [Axis.X, Axis.Y, Axis.Z])
def test_measure(fx_rng: Generator, axis: Axis) -> None:
    """Test direct circuit transpilation with measurement.

    Circuits transpiled in JCZ give patterns with causal flow.
    This test checks manual measurements work for the `transpile_jcz` function.
    It also checks that measurements have uniform outcomes.
    """
    circuit = Circuit(2)
    circuit.h(1)
    circuit.cnot(0, 1)
    circuit.m(0, axis)
    transpiled = transpile_jcz(circuit)
    # Inferring Pauli measurements simulates the measurement on node 0!
    # transpiled.pattern = transpiled.pattern.infer_pauli_measurements()
    transpiled.pattern.remove_input_nodes()

    def simulate_and_measure() -> int:
        measure_method = DefaultMeasureMethod(results=transpiled.pattern.results)
        state = transpiled.pattern.simulate_pattern(rng=fx_rng, measure_method=measure_method)
        measured = measure_method.measurement_outcome(transpiled.classical_outputs[0])
        assert isinstance(state, Statevec)
        return measured

    nb_shots = 10000
    count = sum(1 for _ in range(nb_shots) if simulate_and_measure())
    assert abs(count - nb_shots / 2) < nb_shots / 20


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_check_circuit_flow_cf(circuit: Circuit) -> None:
    """Test CausalFlow transpiled circuits don't fail."""
    f = circuit_to_causal_flow(circuit)
    assert f is not None


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_cf(circuit: Circuit) -> None:
    """Test causal flow transpilation simulation matches direct simulation of the circuit."""
    bs = ConstBranchSelector(0)
    pattern = transpile_jcz_cf(circuit).pattern
    pattern = pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    state = circuit.simulate_statevector().statevec
    state_mbqc = pattern.simulate_pattern(branch_selector=bs)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_flow_cf(circuit: Circuit) -> None:
    """Test causal flow transpiled circuits match direct JCZ transpilation open graph, correction function and partial order."""
    f_cf = circuit_to_causal_flow(circuit)[0]
    f_dir = transpile_jcz(circuit).pattern.extract_causal_flow()
    assert f_cf.correction_function == f_dir.correction_function
    assert f_cf.partial_order_layers == f_dir.partial_order_layers


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare_direct(circuit: Circuit) -> None:
    """Test comparing direct to causal flow transpilation."""
    bs = ConstBranchSelector(0)
    pattern = transpile_jcz(circuit).pattern.infer_pauli_measurements()
    pattern_cf = transpile_jcz_cf(circuit).pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern_cf.remove_input_nodes()
    pattern_cf.perform_pauli_measurements()
    state_mbqc = pattern.simulate_pattern(branch_selector=bs)
    state_mbqc_cf = pattern_cf.simulate_pattern(branch_selector=bs)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_cf.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_circuit_simulation_compare_graphix(circuit: Circuit) -> None:
    """Test comparing Graphix main transpiler pattern simulation to causal flow transpilation."""
    bs = ConstBranchSelector(0)
    pattern = circuit.transpile().pattern.infer_pauli_measurements()
    pattern_cf = transpile_jcz_cf(circuit).pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern_cf.remove_input_nodes()
    pattern_cf.perform_pauli_measurements()
    state_mbqc = pattern.simulate_pattern(branch_selector=bs)
    state_mbqc_cf = pattern_cf.simulate_pattern(branch_selector=bs)
    assert np.abs(np.dot(state_mbqc.flatten().conjugate(), state_mbqc_cf.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_compare(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation comparing direct and causal flow transpilation."""
    bs = ConstBranchSelector(0)
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 2
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    pattern = transpile_jcz(circuit).pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern = pattern.infer_pauli_measurements()
    pattern.perform_pauli_measurements()
    pattern_og = transpile_jcz_cf(circuit).pattern.infer_pauli_measurements()
    pattern_og.remove_input_nodes()
    pattern_og = pattern_og.infer_pauli_measurements()
    pattern_og.perform_pauli_measurements()
    pattern_gpx = circuit.transpile().pattern.infer_pauli_measurements()
    pattern_gpx.remove_input_nodes()
    pattern_gpx = pattern_gpx.infer_pauli_measurements()
    pattern_gpx.perform_pauli_measurements()
    state = pattern.simulate_pattern(branch_selector=bs)
    state_og = pattern_og.simulate_pattern(branch_selector=bs)
    state_gpx = pattern_gpx.simulate_pattern(branch_selector=bs)
    assert state.isclose(state_og)
    assert state_og.isclose(state_gpx)


@pytest.mark.parametrize("jumps", range(1, 11))
def test_random_circuit_with_m(fx_bg: PCG64, jumps: int) -> None:
    """Test random circuit transpilation comparing direct and causal flow transpilation."""
    bs = ConstBranchSelector(0)
    rng = Generator(fx_bg.jumped(jumps))
    nqubits = 3
    depth = 2
    circuit = rand_circuit(nqubits, depth, rng, use_ccx=True)
    circuit.m(1, Axis.Y)
    pattern = transpile_jcz(circuit).pattern.infer_pauli_measurements()
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern_og = transpile_jcz_cf(circuit).pattern.infer_pauli_measurements()
    pattern_og.remove_input_nodes()
    pattern_og.perform_pauli_measurements()
    pattern_gpx = circuit.transpile().pattern.infer_pauli_measurements()
    pattern_gpx.remove_input_nodes()
    pattern_gpx.perform_pauli_measurements()
    state = pattern.simulate_pattern(backend="tensornetwork", branch_selector=bs)
    state_og = pattern_og.simulate_pattern(backend="tensornetwork", branch_selector=bs)
    state_gpx = pattern_gpx.simulate_pattern(backend="tensornetwork", branch_selector=bs)
    assert np.abs(
        np.dot(
            state.to_statevector().flatten().conjugate(),
            state_og.to_statevector().flatten(),
        )
    ) == pytest.approx(1)
    assert np.abs(
        np.dot(
            state_og.to_statevector().flatten().conjugate(),
            state_gpx.to_statevector().flatten(),
        )
    ) == pytest.approx(1)


def normalize_measurement(m: Measurement) -> Measurement:
    if isinstance(m, BlochMeasurement):
        return BlochMeasurement(normalize_angle(m.angle), m.plane)
    return m


def test_circuit_compare_with_m_early() -> None:
    bs = ConstBranchSelector(0)
    circuit = Circuit(3)
    circuit.m(0, Axis.Y)
    circuit.ry(1, ANGLE_PI / 5)
    circuit.cnot(1, 2)
    pattern_gpx = circuit.transpile().pattern.to_bloch().map(normalize_measurement)
    pattern_gpx.standardize()
    pattern = transpile_jcz(circuit).pattern.to_bloch()
    pattern.standardize()
    state = pattern.simulate_pattern(branch_selector=bs)
    pattern_og = transpile_jcz_cf(circuit).pattern.to_bloch()
    pattern_og.standardize()
    state_og = pattern_og.simulate_pattern(branch_selector=bs)
    state_gpx = pattern_gpx.simulate_pattern(branch_selector=bs)
    assert np.abs(np.dot(state.flatten().conjugate(), state_og.flatten())) == pytest.approx(1)
    assert np.abs(np.dot(state_og.flatten().conjugate(), state_gpx.flatten())) == pytest.approx(1)


def test_circuit_compare_with_m_end() -> None:
    bs = ConstBranchSelector(0)
    circuit = Circuit(2)
    circuit.cz(0, 1)
    circuit.rz(1, ANGLE_PI / 5)
    circuit.m(0, Axis.Z)
    pattern_gpx = circuit.transpile().pattern.to_bloch().map(normalize_measurement)
    pattern_gpx.standardize()
    pattern = transpile_jcz(circuit).pattern.to_bloch()
    pattern.standardize()
    pattern_og = transpile_jcz_cf(circuit).pattern.to_bloch()
    pattern_og.standardize()
    state = pattern.simulate_pattern(branch_selector=bs)
    state_og = pattern_og.simulate_pattern(branch_selector=bs)
    state_gpx = pattern_gpx.simulate_pattern(branch_selector=bs)
    assert np.abs(np.dot(state.flatten().conjugate(), state_og.flatten())) == pytest.approx(1)
    assert np.abs(np.dot(state_og.flatten().conjugate(), state_gpx.flatten())) == pytest.approx(1)


@pytest.mark.parametrize("circuit", TEST_BASIC_CIRCUITS)
def test_pauli_presimulation_sizes_match(circuit: Circuit) -> None:
    """Test causal flow transpilation simulation matches direct simulation of the circuit."""
    pattern = transpile_jcz(circuit).pattern.infer_pauli_measurements()
    pattern_cf = transpile_jcz_cf(circuit).pattern.infer_pauli_measurements()
    pattern_gpx = circuit.transpile().pattern.infer_pauli_measurements()
    print("Circuit: ", circuit, "\n")
    print(
        "Nodes before PP: {JCZ: ",
        pattern.n_node,
        ", JCZ_CF: ",
        pattern_cf.n_node,
        ", Graphix: ",
        pattern_gpx.n_node,
        "}\n",
    )
    pattern.remove_input_nodes()
    pattern.perform_pauli_measurements()
    pattern_cf.remove_input_nodes()
    pattern_cf.perform_pauli_measurements()
    pattern_gpx.remove_input_nodes()
    pattern_gpx.perform_pauli_measurements()
    print(
        "Nodes after PP: {JCZ: ",
        pattern.n_node,
        ", JCZ_CF: ",
        pattern_cf.n_node,
        ", Graphix: ",
        pattern_gpx.n_node,
        "}\n",
    )
    assert pattern.n_node == pattern_cf.n_node
    assert pattern.n_node == pattern_gpx.n_node


def test_cz_ccx() -> None:
    """Test case reported in issue #2.

    https://github.com/qat-inria/graphix-jcz-transpiler/issues/2
    """
    circuit = Circuit(width=3)
    circuit.cz(2, 0)
    circuit.ccx(0, 1, 2)
    ref_state = circuit.simulate_statevector().statevec
    graphix_pattern = circuit.transpile().pattern
    graphix_state = graphix_pattern.simulate_pattern()
    assert graphix_state.isclose(ref_state)
    jcz_pattern = transpile_jcz(circuit).pattern
    jcz_state = jcz_pattern.simulate_pattern()
    assert jcz_state.isclose(ref_state)


def test_ccx_decomposition() -> None:
    circuit = Circuit(width=3)
    circuit.cz(2, 0)
    circuit.ccx(0, 1, 2)
    circuit2 = Circuit(width=3)
    circuit2.cz(2, 0)
    circuit2.extend(decompose_ccx(CCX(controls=(0, 1), target=2)))
    state = circuit.simulate_statevector().statevec
    state2 = circuit2.simulate_statevector().statevec
    assert state.isclose(state2)


def test_cnot_cz() -> None:
    """Test regression about output node reordering."""
    circuit = Circuit(width=3, instr=[CNOT(0, 1), CZ((0, 1))])
    state = circuit.simulate_statevector().statevec
    pattern = transpile_jcz(circuit).pattern
    state_mbqc = pattern.simulate_pattern()
    assert state.isclose(state_mbqc)
