# Safe Hylo and ZK-WASM Execution Guide

A comprehensive guide to safe hylomorphism patterns and ZK-WASM execution with the Reality SDK.

## Table of Contents

- [Overview](#overview)
- [Safe Hylo Fundamentals](#safe-hylo-fundamentals)
- [ProcessingState Pattern](#processingstate-pattern)
- [SafeStreamCell](#safestreamcell)
- [TypedCoCellWithCell Trait](#typedcocellwithcell-trait)
- [TypedCoCellWithCellAndProof Extension](#typedcocellwithcellandproof-extension)
- [ZK-WASM Integration](#zk-wasm-integration)
- [RealZKWasmExecutor](#realzkwasmexecutor)
- [ZKWasmStateChannelTyped](#zkwasmstatechanneltyped)
- [Custom Routes for ZK-WASM](#custom-routes-for-zk-wasm)
- [Examples](#examples)
- [Running the Examples](#running-the-examples)

---

## Overview

The Reality SDK provides three complementary patterns for building safe, verifiable blockchain computations:

| Pattern | Purpose | Key Feature |
|---------|---------|-------------|
| **Safe Hylo** | Guaranteed termination | WellFounded measure strictly decreases |
| **TypedCoCellWithCell** | Type-safe Cell infrastructure | Combines TypedCoCell with kernel Cell |
| **ZK-WASM** | Cryptographic proofs | ZK-STARK proofs for WASM execution |

These patterns work together to create blockchain nodes where:
1. Every computation terminates (halt-free by construction)
2. Every execution produces cryptographic proof
3. Validators verify proofs, not re-execute computations

---

## Safe Hylo Fundamentals

A **hylomorphism** (hylo) is the composition of an unfold (anamorphism) followed by a fold (catamorphism):

```
hylo(algebra, coalgebra)(input) = algebra(fmap(hylo(algebra, coalgebra))(coalgebra(input)))
```

In the Reality SDK, **safe hylo** guarantees termination via WellFounded:

```scala
// The pattern used throughout SafeL0Cell, SafeBlockConsensusCell, SafeStreamCell
def runSafeHylo[F[_]: Async](input: In): F[Either[CellError, Out]] = {
  for {
    processingState <- safeCoalgebra[F](input)   // Unfold: In → ProcessingState
    result <- safeAlgebra[F](processingState)     // Fold: ProcessingState → Out
  } yield result
}
```

---

## ProcessingState Pattern

The `ProcessingState` sealed trait tracks execution depth for WellFounded termination:

```scala
sealed trait ProcessingState[+A] extends Ω {
  def depth: Int
}

/** Input received - needs processing */
case class Processing[A](value: A) extends ProcessingState[A] {
  def depth: Int = 1
}

/** Processing complete - result available */
case class Completed[A](result: Either[CellError, A]) extends ProcessingState[A] {
  def depth: Int = 0
}
```

### WellFounded Instance

```scala
def wellFoundedProcessingState[A]: WellFounded[ProcessingState[A]] =
  WellFounded.fromMeasure(_.depth)
```

### Termination Guarantee

The depth measure strictly decreases:
- `Processing` has depth 1
- `Completed` has depth 0
- 1 > 0, so termination is guaranteed in one step

---

## SafeStreamCell

`SafeStreamCell` creates TypedCoCellWithCell instances using the safe hylo pattern:

```scala
object SafeStreamCell {

  /**
   * The safe coalgebra - unfolds input into ProcessingState.
   */
  def safeCoalgebra[F[_]: Async, A](input: A): F[ProcessingState[A]] =
    Processing(input).pure[F]

  /**
   * The safe algebra - folds ProcessingState into result.
   */
  def safeAlgebra[F[_]: Async, A](state: ProcessingState[A]): F[Either[CellError, A]] =
    state match {
      case Processing(value) => value.asRight[CellError].pure[F]
      case Completed(result) => result.pure[F]
    }

  /**
   * Create a TypedCoCellWithCellAndProof from a pure transformation.
   */
  def apply[In <: Ω, Out <: Ω](
    cellName: String,
    transform: In => Out
  ): TypedCoCellWithCellAndProof[In, Out]

  /**
   * Create for pair inputs.
   */
  def forPair[A <: Ω, B <: Ω, Out <: Ω](
    cellName: String,
    transform: (A, B) => Out
  ): TypedCoCellWithCellAndProof[ΩPair[A, B], Out]
}
```

### Usage Example

```scala
val safeCalibrate: TypedCoCellWithCellAndProof[RawSensorReading, CalibratedReading] =
  SafeStreamCell("calibrate", (raw: RawSensorReading) =>
    CalibratedReading(
      deviceId = raw.deviceId,
      value = raw.value * 1.05,
      unit = "celsius",
      calibrationFactor = 1.05
    )
  )

// Get termination proof
val proof = safeCalibrate.terminationProof(testInput)
// CellTerminationProof(calibrate, 1 → 0, valid=true)
```

---

## TypedCoCellWithCell Trait

The `TypedCoCellWithCell` trait bridges TypedCoCell with the kernel Cell infrastructure:

```scala
trait TypedCoCellWithCell[In <: Ω, Out <: Ω] extends TypedCoCell[In, Out] {

  /**
   * Create the underlying Cell for a specific effect type.
   * The Cell uses SafeHylo with AlgebraM and CoalgebraM.
   */
  def mkCellF[F[_]: Async]: Ω => Cell[F, StackF, Ω, Ω, Either[CellError, Ω]]

  /**
   * Create HTTP API resources for deploying this cell as a node.
   */
  def mkResources[A <: CliMethod]: (A, SDK[IO]) => Resource[IO, HttpApi[IO]]
}
```

### Key Properties

| Method | Description |
|--------|-------------|
| `mkCellF[F]` | Creates kernel Cell with SafeHylo processing |
| `mkResources` | Creates HTTP API for node deployment |
| `run[F](input)` | Inherited from TypedCoCell - runs the transformation |
| `>>>`, `&&&`, `***` | Inherited composition operators |

---

## TypedCoCellWithCellAndProof Extension

Extends TypedCoCellWithCell with termination proof capabilities:

```scala
trait TypedCoCellWithCellAndProof[In <: Ω, Out <: Ω] extends TypedCoCellWithCell[In, Out] {
  def terminationProof(input: In): CellTerminationProof
}
```

### CellTerminationProof

```scala
case class CellTerminationProof(
  cellName: String,
  inputRank: Rank,
  steps: List[Rank]
) extends Ω {
  def isValid: Boolean = {
    val allRanks = inputRank :: steps
    allRanks.sliding(2).forall {
      case Seq(a, b) => a > b
      case _ => true
    }
  }
  def depth: Int = steps.length
}
```

---

## ZK-WASM Integration

The Reality SDK integrates WASM execution with ZK-STARK proof generation via `RealZKWasmExecutor`.

### Architecture

```
Input → WASM Execution → Output + ZK-STARK Proof
                           ↓
                    Proof verification
                           ↓
                    Block consensus
```

---

## RealZKWasmExecutor

`RealZKWasmExecutor` provides cryptographic proof generation for WASM execution:

```scala
class RealZKWasmExecutor[F[_]: Async] {

  /**
   * Generate a ZK-STARK proof for given inputs.
   */
  def generateProof(
    name: String,
    publicInputs: List[(String, Any)],
    privateInputs: List[(String, Any)]
  ): F[Either[ZKProofError, Path]]

  /**
   * Verify a ZK-STARK proof.
   */
  def verifyProof(
    proofPath: Path,
    publicInputs: List[(String, Any)]
  ): F[Either[ZKProofError, Boolean]]
}
```

### Proof Structure

Generated proofs include:
- Cryptographic commitment to inputs
- Execution trace hash
- Timestamp for freshness
- Metadata for verification

---

## ZKWasmStateChannelTyped

`ZKWasmStateChannelTyped` combines safe hylo with ZK-WASM execution:

```scala
object ZKWasmStateChannelTyped {

  // Processing states with WellFounded measure
  sealed trait ZKWasmProcessingState extends Ω {
    def depth: Int
  }

  case class WasmInput(input: L0OutputState) extends ZKWasmProcessingState {
    def depth: Int = 2  // Needs WASM execution
  }

  case class WasmExecuted(
    input: L0OutputState,
    result: Int,
    proofPath: Option[Path]
  ) extends ZKWasmProcessingState {
    def depth: Int = 1  // Needs consensus
  }

  case class WasmCompleted(result: Either[CellError, StateChannelState]) extends ZKWasmProcessingState {
    def depth: Int = 0  // Done
  }

  /**
   * Create a TypedCoCellWithCell for ZK-WASM execution.
   */
  def apply(staticConfig: Option[L1CoCellConfig] = None): TypedCoCellWithCell[L0OutputState, StateChannelState]
}
```

### Execution Flow

```
L0OutputState → WasmInput(depth=2)
       ↓
   safeCoalgebra
       ↓
   WasmExecuted(depth=1) + ZK proof
       ↓
   safeAlgebra
       ↓
StateChannelState(depth=0)
```

---

## Custom Routes for ZK-WASM

`ZKWasmExecutionRoutes` adds HTTP endpoints for WASM execution:

```scala
class ZKWasmExecutionRoutes[F[_]: Async](
  nodeApi: HttpApi[F]
) extends AdditionalRoutes[F] with Http4sDsl[F] {

  // POST /zk-wasm/execute
  // Executes WASM function with ZK proof generation

  // POST /zk-wasm/verify
  // Verifies a ZK proof

  // GET /zk-wasm/status
  // Returns executor status and available functions
}
```

### Factory Method

```scala
object ZKWasmExecutionRoutes {
  def make[F[_]: Async]: HttpApi[F] => AdditionalRoutes[F] =
    (nodeApi: HttpApi[F]) => new ZKWasmExecutionRoutes[F](nodeApi)
}
```

### Integration with BabelApp

```scala
val zkWasmStateChannelNode: TypedCoCellNode[L0OutputState, StateChannelState] =
  BabelApp.l1Node(
    ZKWasmStateChannelTyped.withConfig(l1Config),
    l1Config,
    MkZKWasmExecutorStateChannel,
    customRoutes = List(ZKWasmExecutionRoutes.make[IO])  // <-- Custom routes
  )
```

---

## Examples

### CombinedSafeHyloExamples

Located at: `modules/combined/src/main/scala/org/reality/combined/examples/topologies/CombinedSafeHyloExamples.scala`

Demonstrates safe hylo versions of all topology patterns:

```scala
// Safe base cells using SafeStreamCell
val safeCalibrate = SafeStreamCell("calibrate", (raw: RawSensorReading) => ...)
val safeValidate = SafeStreamCell("validate", (cal: CalibratedReading) => ...)
val safeComputeStats = SafeStreamCell("computeStats", (reading: CalibratedReading) => ...)
val safeGenerateAlert = SafeStreamCell("generateAlert", (reading: CalibratedReading) => ...)

// Composed topologies use TypedCoCell operators
val safeFullPipeline = safeCalibrate >>> ((safeValidate &&& safeComputeStats) &&& safeGenerateAlert)

val safeDiamondPattern = {
  val safeMerge = SafeStreamCell.forPair("merge", (v: ValidatedReading, s: ReadingStats) => ...)
  safeCalibrate >>> (safeValidate &&& safeComputeStats) >>> tupleToΩPair >>> safeMerge
}
```

### rAppGenius ZK-WASM Example

Located at: `/Users/wyatt/rAppGenius/src/main/scala/com/example/rapp/`

Files:
- `Main.scala` - Main cluster application
- `ZKWasmStateChannelTyped.scala` - State channel with ZK-WASM
- `ZKWasmExecutionRoutes.scala` - Custom HTTP routes

```scala
object Main extends RAppGeniusCluster

open class RAppGeniusCluster extends BabelApp[L0InputState, StateChannelState] {

  val l0Node = BabelApp.l0Node(CombinedL0Typed.withConfig(l0Config), l0Config)

  val zkWasmStateChannelNode = BabelApp.l1Node(
    ZKWasmStateChannelTyped.withConfig(l1Config),
    l1Config,
    MkZKWasmExecutorStateChannel,
    customRoutes = List(ZKWasmExecutionRoutes.make[IO])
  )

  val pipelines = List(CoCellPipeline.from(l0Node) >>> zkWasmStateChannelNode)
}
```

---

## Running the Examples

### CombinedSafeHyloExamples

```bash
sbt "combined/runMain org.reality.combined.examples.topologies.CombinedSafeHyloExamplesRunner"
```

Expected output:
```
======================================================================
COMBINED SAFE HYLO EXAMPLES
Using safeCoalgebra/safeAlgebra pattern (same as SafeL0Cell)
======================================================================

--- Safe Base Cells (with termination proofs) ---
safeCalibrate: Right(CalibratedReading(sensor-001,89.775,celsius,1.05))
  Proof: CellTerminationProof(calibrate, 1 → 0, valid=true)

--- Safe Full Pipeline: Raw → ((Validated, Stats), Alert) ---
Result: Right(((ValidatedReading(...),ReadingStats(...)),Alert(...)))

--- Safe Diamond Pattern ---
       ┌─── validate ───┐
cal ───┤                ├─── merge
       └─── stats ──────┘
Result: Right(Alert(INFO,Valid=true, Sum=89.775,...))
```

### rAppGenius ZK-WASM

```bash
cd /Users/wyatt/rAppGenius
NODE_TYPE=genesis sbt run
```

Test endpoints:
```bash
# Check status
curl http://localhost:9010/zk-wasm/status

# Execute WASM with ZK proof
curl -X POST http://localhost:9010/zk-wasm/execute \
  -H "Content-Type: application/json" \
  -d '{"functionName":"add","param1":42,"param2":58}'

# Verify proof
curl -X POST http://localhost:9010/zk-wasm/verify \
  -H "Content-Type: application/json" \
  -d '{"proofPath":"./proofs/wasm_add_xxx.proof","inputs":[["input_0",42],["input_1",58]]}'
```

Expected responses:
```json
// GET /zk-wasm/status
{
  "status": "ready",
  "wasmPath": "wasm/add_numbers.wasm",
  "proofStoragePath": "./proofs",
  "availableFunctions": ["add", "multiply", "subtract", "sum_of_squares", "fibonacci"]
}

// POST /zk-wasm/execute
{
  "functionName": "add",
  "inputs": [42, 58],
  "result": 100,
  "proofPath": "./proofs/wasm_add_xxx.proof",
  "proofHash": "582db79309ff2eeebaa4fc791abb5c11...",
  "txHash": "wasm-tx-9cbbff4431f15",
  "timestamp": 1767665201519
}

// POST /zk-wasm/verify
{
  "proofPath": "./proofs/wasm_add_xxx.proof",
  "valid": true,
  "message": "Proof verification successful"
}
```

---

## File Structure

### Reality Project

```
modules/combined/src/main/scala/org/reality/combined/
├── cell/
│   ├── SafeL0Cell.scala              # L0 safe hylo implementation
│   ├── SafeBlockConsensusCell.scala  # Block consensus safe hylo
│   └── SafeL2Cell.scala              # L2 safe hylo implementation
├── topos/
│   ├── WellFounded.scala             # WellFounded termination proofs
│   └── TypedCoCell.scala             # TypedCoCell trait and operators
├── examples/
│   ├── TypedCocellExample.scala      # TypedCoCellWithCell trait
│   └── topologies/
│       ├── StreamTopologyTypes.scala       # Domain types
│       ├── CombinedSafeHyloExamples.scala  # Safe hylo versions
│       └── CombinedExamples.scala          # Original examples
├── ZKWasmExecutor.scala              # RealZKWasmExecutor implementation
└── WasmExecutorRoutes.scala          # WASM execution HTTP routes
```

### rAppGenius Project

```
/Users/wyatt/rAppGenius/src/main/scala/com/example/rapp/
├── Main.scala                   # Main cluster application
├── ZKWasmStateChannelTyped.scala    # State channel with ZK-WASM
├── ZKWasmExecutionRoutes.scala  # Custom HTTP routes for /zk-wasm/*
├── SimpleExample.scala          # Simple pipeline examples
└── wasm/
    └── WasmExecutor.scala       # Base WASM executor
```

---

## Summary

| Component | Purpose | Key Pattern |
|-----------|---------|-------------|
| `SafeStreamCell` | Create safe cells | safeCoalgebra → transform → safeAlgebra |
| `TypedCoCellWithCell` | Bridge typed and kernel | mkCellF + mkResources |
| `TypedCoCellWithCellAndProof` | Add termination proofs | terminationProof(input) |
| `RealZKWasmExecutor` | Generate ZK proofs | generateProof / verifyProof |
| `ZKWasmStateChannelTyped` | ZK-WASM state channel | 3-depth processing state |
| `ZKWasmExecutionRoutes` | Custom HTTP endpoints | /zk-wasm/execute, verify, status |

These components enable building blockchain nodes with:
- **Guaranteed termination** via WellFounded measures
- **Type-safe pipelines** via TypedCoCell composition
- **Cryptographic proofs** via ZK-STARK generation
- **Verifiable execution** via proof verification
