"""Graphix Transpiler from circuit to MBQC patterns via J-∧z decomposition.

Copyright (C) 2025, QAT team (ENS-PSL, Inria, CNRS).
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

import networkx as nx
from graphix import Pattern, command, instruction
from graphix.flow.core import (
    CausalFlow,
    _corrections_to_partial_order_layers,  # noqa: PLC2701
)
from graphix.fundamentals import ANGLE_PI, ParameterizedAngle
from graphix.instruction import InstructionKind
from graphix.measurements import BlochMeasurement, Measurement, PauliMeasurement
from graphix.opengraph import OpenGraph
from graphix.optimization import StandardizedPattern
from graphix.transpiler import (
    Circuit,
    TranspileResult,
)
from typing_extensions import assert_never

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from collections.abc import Set as AbstractSet

    from graphix.parameter import ExpressionOrFloat


class JCZInstructionKind(Enum):
    """Tag for instruction kind."""

    CZ = enum.auto()
    J = enum.auto()


@dataclass
class J:
    """J circuit instruction."""

    target: int
    angle: ExpressionOrFloat
    kind: ClassVar[Literal[JCZInstructionKind.J]] = dataclasses.field(
        default=JCZInstructionKind.J,
        init=False,
    )


JCZInstruction: TypeAlias = (
    instruction.CCX
    | instruction.RZZ
    | instruction.CNOT
    | instruction.SWAP
    | instruction.H
    | instruction.S
    | instruction.X
    | instruction.Y
    | instruction.Z
    | instruction.I
    | instruction.RX
    | instruction.RY
    | instruction.RZ
    | instruction.CZ
    | J
)


def decompose_ccx(
    instr: instruction.CCX,
) -> list[instruction.H | instruction.CNOT | instruction.RZ]:
    """Return a decomposition of the CCX gate into H, CNOT, T and T-dagger gates.

    This decomposition of the Toffoli gate can be found in
    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 182 in the 10th Anniversary Edition).

    Parameters
    ----------
        instr: the CCX instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.target, ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[1], target=instr.target),
        instruction.RZ(instr.target, -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.target),
        instruction.RZ(instr.controls[1], -ANGLE_PI / 4),
        instruction.RZ(instr.target, ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
        instruction.H(instr.target),
        instruction.RZ(instr.controls[1], -ANGLE_PI / 4),
        instruction.CNOT(control=instr.controls[0], target=instr.controls[1]),
        instruction.RZ(instr.controls[0], ANGLE_PI / 4),
        instruction.RZ(instr.controls[1], ANGLE_PI / 2),
    ]


def decompose_rzz(instr: instruction.RZZ) -> list[instruction.CNOT | instruction.RZ]:
    """Return a decomposition of RZZ(α) gate as CNOT(control, target)·Rz(target, α)·CNOT(control, target).

    Parameters
    ----------
        instr: the RZZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(target=instr.target, control=instr.control),
        instruction.RZ(instr.target, instr.angle),
        instruction.CNOT(target=instr.target, control=instr.control),
    ]


def decompose_cnot(instr: instruction.CNOT) -> list[instruction.H | instruction.CZ]:
    """Return a decomposition of the CNOT gate as H·∧z·H.

    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the CNOT instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.H(instr.target),
        instruction.CZ((instr.control, instr.target)),
        instruction.H(instr.target),
    ]


def decompose_swap(instr: instruction.SWAP) -> list[instruction.CNOT]:
    """Return a decomposition of the SWAP gate as CNOT(0, 1)·CNOT(1, 0)·CNOT(0, 1).

    Michael A. Nielsen and Isaac L. Chuang,
    Quantum Computation and Quantum Information,
    Cambridge University Press, 2000
    (p. 23 in the 10th Anniversary Edition).

    Parameters
    ----------
        instr: the SWAP instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
        instruction.CNOT(control=instr.targets[1], target=instr.targets[0]),
        instruction.CNOT(control=instr.targets[0], target=instr.targets[1]),
    ]


def decompose_y(instr: instruction.Y) -> list[instruction.X | instruction.Z]:
    """Return a decomposition of the Y gate as X·Z.

    Parameters
    ----------
        instr: the Y instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return list(reversed([instruction.X(instr.target), instruction.Z(instr.target)]))


def decompose_rx(instr: instruction.RX) -> list[J]:
    """Return a J decomposition of the RX gate.

    The Rx(α) gate is decomposed into J(α)·H (that is to say, J(α)·J(0)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RX instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((instr.angle, 0))]


def decompose_ry(instr: instruction.RY) -> list[J]:
    """Return a J decomposition of the RY gate.

    The Ry(α) gate is decomposed into J(0)·J(π/2)·J(α)·J(-π/2).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, Robust and parsimonious realisations of unitaries in the one-way
    model, 2004.

    Parameters
    ----------
        instr: the RY instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((0, ANGLE_PI / 2, instr.angle, -ANGLE_PI / 2))]


def decompose_rz(instr: instruction.RZ) -> list[J]:
    """Return a J decomposition of the RZ gate.

    The Rz(α) gate is decomposed into H·J(α) (that is to say, J(0)·J(α)).
    Vincent Danos, Elham Kashefi, Prakash Panangaden, The Measurement Calculus, 2007.

    Parameters
    ----------
        instr: the RZ instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    return [J(target=instr.target, angle=angle) for angle in reversed((0, instr.angle))]


def instruction_to_jcz(instr: JCZInstruction) -> Sequence[J | instruction.CZ]:
    """Return a J-∧z decomposition of the instruction.

    Parameters
    ----------
        instr: the instruction to decompose.

    Returns
    -------
        the decomposition.

    """
    # Use == for mypy
    if instr.kind == JCZInstructionKind.J:
        return [instr]
    if instr.kind == InstructionKind.CZ:
        return [instr]
    if instr.kind == InstructionKind.I:
        return []
    if instr.kind == InstructionKind.H:
        return [J(instr.target, 0)]
    if instr.kind == InstructionKind.S:
        return instruction_to_jcz(instruction.RZ(instr.target, ANGLE_PI / 2))
    if instr.kind == InstructionKind.X:
        return instruction_to_jcz(instruction.RX(instr.target, ANGLE_PI))
    if instr.kind == InstructionKind.Y:
        return instruction_list_to_jcz(decompose_y(instr))
    if instr.kind == InstructionKind.Z:
        return instruction_to_jcz(instruction.RZ(instr.target, ANGLE_PI))
    if instr.kind == InstructionKind.RX:
        return decompose_rx(instr)
    if instr.kind == InstructionKind.RY:
        return decompose_ry(instr)
    if instr.kind == InstructionKind.RZ:
        return decompose_rz(instr)
    if instr.kind == InstructionKind.CCX:
        return instruction_list_to_jcz(decompose_ccx(instr))
    if instr.kind == InstructionKind.RZZ:
        return instruction_list_to_jcz(decompose_rzz(instr))
    if instr.kind == InstructionKind.CNOT:
        return instruction_list_to_jcz(decompose_cnot(instr))
    if instr.kind == InstructionKind.SWAP:
        return instruction_list_to_jcz(decompose_swap(instr))
    assert_never(instr.kind)


def instruction_list_to_jcz(
    instrs: Iterable[JCZInstruction],
) -> list[J | instruction.CZ]:
    """Return a J-∧z decomposition of the sequence of instructions.

    Parameters
    ----------
        instrs: the instruction sequence to decompose.

    Returns
    -------
        the decomposition.

    """
    return [jcz_instr for instr in instrs for jcz_instr in instruction_to_jcz(instr)]


class IllformedCircuitError(Exception):
    """Raised if the circuit is ill-formed."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Ill-formed pattern")


class CircuitWithMeasurementError(Exception):
    """Raised if the circuit contains measurements."""

    def __init__(self) -> None:
        """Build the exception."""
        super().__init__("Circuits containing measurements are not supported by the transpiler.")


class InternalInstructionError(Exception):
    """Raised if the circuit contains internal _XC or _ZC instructions."""

    def __init__(self, instr: instruction.Instruction) -> None:
        """Build the exception."""
        super().__init__(f"Internal instruction: {instr}")


def j_commands(current_node: int, next_node: int, angle: ParameterizedAngle) -> list[command.Command]:
    """Return the MBQC pattern commands for a J gate.

    Parameters
    ----------
        current_node: the current node.
        next_node: the next node.
        angle: the angle of the J gate.

    Returns
    -------
        the MBQC pattern commands for a J gate as a list

    """
    return [
        command.N(node=next_node),
        command.E(nodes=(current_node, next_node)),
        command.M(current_node, Measurement.XY(angle)),
        command.X(node=next_node, domain={current_node}),
    ]


def normalize_angle(angle: ParameterizedAngle) -> ParameterizedAngle:
    r"""Return an equivalent angle in range :math:`[0, 2 \cdot \pi)` if ``angle`` is instantiated.

    Parameters
    ----------
    angle: ParameterizedAngle
        An angle.

    Returns
    -------
    ParameterizedAngle
        An equivalent angle in range :math:`[0, 2 \cdot \pi)` if ``angle`` is instantiated.
        If ``angle`` is parameterized, ``angle`` is returned unchanged.
    """
    if isinstance(angle, float):
        return angle % (2 * ANGLE_PI)
    return angle


def transpile_jcz(circuit: Circuit) -> TranspileResult:
    """Transpile a circuit via a J-∧z decomposition.

    Parameters
    ----------
        circuit: the circuit to transpile.

    Returns
    -------
        the result of the transpilation: a pattern and indices for measures.

    Raises
    ------
        IllformedCircuitError: if the circuit has underdefined instructions.

    """
    n_nodes = circuit.width
    indices: list[int | None] = list(range(n_nodes))
    pattern = Pattern(input_nodes=range(n_nodes))
    classical_outputs: dict[int, command.M] = {}
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.M:
            target = indices[instr.target]
            if target is None:
                raise IllformedCircuitError
            classical_outputs[target] = command.M(target, PauliMeasurement(instr.axis))
            indices[instr.target] = None
            continue
        for instr_jcz in instruction_to_jcz(instr):
            if instr_jcz.kind == JCZInstructionKind.J:
                target = indices[instr_jcz.target]
                if target is None:
                    raise IllformedCircuitError
                ancilla = n_nodes
                n_nodes += 1
                pattern.extend(j_commands(target, ancilla, normalize_angle(-instr_jcz.angle)))
                indices[instr_jcz.target] = ancilla
                continue
            if instr_jcz.kind == InstructionKind.CZ:
                t0, t1 = instr_jcz.targets
                i0, i1 = indices[t0], indices[t1]
                if i0 is None or i1 is None:
                    raise IllformedCircuitError
                pattern.extend([command.E(nodes=(i0, i1))])
                continue
            assert_never(instr_jcz.kind)
    pattern.extend(classical_outputs.values())
    pattern.reorder_output_nodes([node for node in indices if node is not None])
    return TranspileResult(pattern, tuple(classical_outputs.keys()))


def circuit_to_causal_flow(
    circuit: Circuit,
) -> tuple[CausalFlow[BlochMeasurement], dict[int, command.M]]:
    """Transpile a circuit via a J-∧z-like decomposition to an open graph.

    Parameters
    ----------
        circuit: the circuit to transpile.

    Returns
    -------
        a causal flow.

    Raises
    ------
        IllformedCircuitError: if the pattern is ill-formed (operation on already measured node)
        CircuitWithMeasurementError: if the circuit contains measurements.

    """
    indices: list[int | None] = list(range(circuit.width))
    n_nodes = circuit.width
    measurements: dict[int, BlochMeasurement] = {}
    classical_outputs: dict[int, command.M] = {}
    inputs = list(range(n_nodes))
    graph: nx.Graph[int] = nx.Graph()
    graph.add_nodes_from(inputs)
    x_corrections: dict[int, AbstractSet[int]] = {}
    for instr in circuit.instruction:
        if instr.kind == InstructionKind.M:
            target = indices[instr.target]
            if target is None:
                raise IllformedCircuitError
            classical_outputs[target] = command.M(target, PauliMeasurement(instr.axis))
            indices[instr.target] = None
            continue
        for instr_jcz in instruction_to_jcz(instr):
            if instr_jcz.kind == JCZInstructionKind.J:
                target = indices[instr_jcz.target]
                if target is None:
                    raise IllformedCircuitError
                graph.add_edge(target, n_nodes)  # Also adds nodes
                measurements[target] = Measurement.XY(normalize_angle(-instr_jcz.angle))
                indices[instr_jcz.target] = n_nodes
                x_corrections[target] = {n_nodes}  # X correction on ancilla
                n_nodes += 1
                continue
            if instr_jcz.kind == InstructionKind.CZ:
                t0, t1 = instr_jcz.targets
                i0, i1 = indices[t0], indices[t1]
                if i0 is None or i1 is None:
                    raise IllformedCircuitError
                # If edge exists, remove it; else, add it
                if graph.has_edge(i0, i1):
                    graph.remove_edge(i0, i1)
                else:
                    graph.add_edge(i0, i1)
                continue
            assert_never(instr_jcz.kind)
    outputs = [i for i in indices if i is not None]
    outputs.extend(classical_outputs.keys())
    og = OpenGraph(
        graph=graph,
        input_nodes=tuple(inputs),
        output_nodes=tuple(outputs),
        measurements=measurements,
    )
    z_corrections: dict[int, AbstractSet[int]] = {}
    for node, correctors in x_corrections.items():
        (corrector,) = correctors
        z_targets = set(graph.neighbors(corrector)) - {node}
        if z_targets:
            z_corrections[node] = z_targets
    partial_order_layers = _corrections_to_partial_order_layers(og, x_corrections, z_corrections)
    return CausalFlow(og, x_corrections, partial_order_layers), classical_outputs


def transpile_jcz_cf(circuit: Circuit) -> TranspileResult:
    """Transpile a circuit via a J-∧z-like decomposition to a pattern.

    Parameters
    ----------
        circuit: the circuit to transpile.

    Returns
    -------
        the result of the transpilation: a pattern.

    """
    f, classical_outputs = circuit_to_causal_flow(circuit)
    pattern = StandardizedPattern.from_pattern(f.to_corrections().to_pattern()).to_space_optimal_pattern()
    pattern.extend(classical_outputs.values())
    return TranspileResult(pattern, tuple(classical_outputs.keys()))
