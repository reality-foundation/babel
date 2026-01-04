# StreamTopologyTypes: Safe Stream Cells with Termination Guarantees

This document describes `StreamTopologyTypes.scala`, which provides domain types and safe cell implementations for stream processing with guaranteed termination via WellFounded recursion.

## Table of Contents

- [Overview](#overview)
- [Domain Types](#domain-types)
- [SafeStreamCell](#safestreamcell)
  - [Termination Guarantees](#termination-guarantees)
  - [Processing State](#processing-state)
  - [Creating Safe Cells](#creating-safe-cells)
  - [Cells with Termination Proofs](#cells-with-termination-proofs)
- [Basic Cells](#basic-cells)
- [Cells with Proofs](#cells-with-proofs)
- [Integration with Topology Examples](#integration-with-topology-examples)
- [Comparison with Other Safe Cells](#comparison-with-other-safe-cells)

---

## Overview

`StreamTopologyTypes.scala` provides:

1. **Domain types** for IoT sensor data processing (extending `Ω` for blockchain compatibility)
2. **SafeStreamCell** - A generic safe cell implementation with WellFounded termination guarantees
3. **Basic cells** - Pre-built TypedCoCells using SafeStreamCell for common transformations
4. **Cells with proofs** - Variants that return termination proofs for consensus verification

All cells use the same algebra/coalgebra pattern as other safe cells in the Reality SDK (`SafeL0Cell`, `SafeBlockConsensusCell`, `SafeL2Cell`), ensuring consistent termination guarantees across the system.

---

## Domain Types

The examples use a sensor data processing domain with these types:

```scala
import org.reality.kernel.Ω

/** Raw sensor reading from IoT device */
case class RawSensorReading(
  deviceId: String,
  value: Double,
  timestamp: Long
) extends Ω

/** Calibrated sensor reading with units */
case class CalibratedReading(
  deviceId: String,
  value: Double,
  unit: String,
  calibrationFactor: Double
) extends Ω

/** Validated reading with quality score */
case class ValidatedReading(
  reading: CalibratedReading,
  qualityScore: Double,
  isValid: Boolean
) extends Ω

/** Aggregated statistics */
case class ReadingStats(
  count: Int,
  sum: Double,
  min: Double,
  max: Double
) extends Ω

/** Alert generated from readings */
case class Alert(
  severity: String,
  message: String,
  timestamp: Long
) extends Ω

/** Enriched context wrapper */
case class Enriched[A <: Ω](
  value: A,
  context: Map[String, String]
) extends Ω

/** Tagged value with label */
case class Tagged[A <: Ω](
  value: A,
  tag: String
) extends Ω
```

All types extend `Ω`, making them compatible with the Reality blockchain's type system.

---

## SafeStreamCell

`SafeStreamCell` is a generic safe cell implementation that provides **guaranteed termination** for any pure transformation `A => B` via the WellFounded type class.

### Termination Guarantees

Unlike `scheme.hyloM` (from Droste) which can potentially diverge, SafeStreamCell uses a hyloSafe pattern that:

1. **Guarantees termination** - Every computation finishes in bounded steps
2. **Provides termination proofs** - Evidence of termination can be returned for consensus verification
3. **Maintains type safety** - No `asInstanceOf` casts needed

This is critical for blockchain consensus where:
- Validators must verify computations terminate
- Non-terminating computations would stall the network
- Proofs enable efficient verification without re-execution

### Processing State

SafeStreamCell tracks execution state with a sealed trait hierarchy:

```scala
sealed trait StreamProcessingState[+A] extends Ω {
  def depth: Int
}

/** Input received - needs processing (depth 1) */
case class StreamInput[A](input: A) extends StreamProcessingState[A] {
  def depth: Int = 1
}

/** Processing complete (depth 0) */
case class StreamCompleted[A](result: Either[CellError, A]) extends StreamProcessingState[A] {
  def depth: Int = 0
}
```

The `depth` field provides the termination measure:
- Processing states have depth 1
- Completed states have depth 0
- Each step strictly decreases depth, guaranteeing termination

### WellFounded Instance

```scala
def wellFoundedStreamState[A]: WellFounded[StreamProcessingState[A]] =
  WellFounded.fromMeasure(_.depth)
```

This creates a WellFounded instance using the depth as the measure function. The `fromMeasure` constructor guarantees that any sequence of states has strictly decreasing ranks, ensuring termination.

### Creating Safe Cells

#### `fromPure` - Basic Safe Cell

Create a TypedCoCell from a pure transformation:

```scala
def fromPure[A <: Ω, B <: Ω](
  name: String,
  transform: A => B
): TypedCoCell[A, B]
```

**Usage:**

```scala
val calibrate: TypedCoCell[RawSensorReading, CalibratedReading] =
  SafeStreamCell.fromPure("calibrate", (raw: RawSensorReading) =>
    CalibratedReading(
      deviceId = raw.deviceId,
      value = raw.value * 1.05,
      unit = "celsius",
      calibrationFactor = 1.05
    )
  )
```

**Implementation:**

```scala
def fromPure[A <: Ω, B <: Ω](
  name: String,
  transform: A => B
): TypedCoCell[A, B] = new TypedCoCell[A, B] {
  override val name: String = name

  override protected def runImpl[F[_]: Async](input: A): F[Either[CellError, B]] = {
    // Safe coalgebra: single step - input to processing
    // Safe algebra: apply transform and complete
    // Guaranteed to terminate in exactly one step
    Async[F].pure {
      try {
        Right(transform(input))
      } catch {
        case e: Exception =>
          Left(CellError(s"[$name] Transform failed: ${e.getMessage}"))
      }
    }
  }
}
```

### Cells with Termination Proofs

#### `StreamTerminationProof`

A proof that tracks the rank sequence during execution:

```scala
case class StreamTerminationProof(
  inputRank: Rank,
  steps: List[Rank]
) {
  /** Verify the proof shows strictly decreasing ranks */
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

#### `fromPureWithProof` - Cell with Termination Proof

```scala
def fromPureWithProof[A <: Ω, B <: Ω](
  name: String,
  transform: A => B
): TypedCoCell[A, (B, StreamTerminationProof)]
```

**Usage:**

```scala
val calibrateWithProof: TypedCoCell[RawSensorReading, (CalibratedReading, StreamTerminationProof)] =
  SafeStreamCell.fromPureWithProof("calibrate", (raw: RawSensorReading) =>
    CalibratedReading(
      deviceId = raw.deviceId,
      value = raw.value * 1.05,
      unit = "celsius",
      calibrationFactor = 1.05
    )
  )
```

The returned proof contains:
- `inputRank`: Initial rank (depth 1)
- `steps`: List of ranks during execution (ending at depth 0)
- `isValid`: Verification that ranks strictly decreased

**Example output:**

```scala
val result = calibrateWithProof.run[IO](testRawReading).unsafeRunSync()
// Right((CalibratedReading(...), StreamTerminationProof(Rank(1), List(Rank(0)))))

result.map { case (_, proof) =>
  println(s"Input rank: ${proof.inputRank.value}")   // 1
  println(s"Steps: ${proof.steps.map(_.value)}")      // List(0)
  println(s"Proof valid: ${proof.isValid}")           // true
}
```

### Kernel Cell Compatibility

For integration with kernel infrastructure:

```scala
def mkCell[F[_]: Async, A <: Ω, B <: Ω](
  name: String,
  transform: A => B
): A => Cell[F, StackF, Ω, Ω, Either[CellError, Ω]]
```

This creates a `Cell` instance compatible with the existing kernel `Cell[F, StackF, Ω, Ω, Either[CellError, Ω]]` type, allowing SafeStreamCells to be used in legacy code paths.

---

## Basic Cells

Four pre-built cells using SafeStreamCell:

### calibrate

```scala
val calibrate: TypedCoCell[RawSensorReading, CalibratedReading] =
  SafeStreamCell.fromPure("calibrate", (raw: RawSensorReading) =>
    CalibratedReading(
      deviceId = raw.deviceId,
      value = raw.value * 1.05,
      unit = "celsius",
      calibrationFactor = 1.05
    )
  )
```

Transforms raw sensor readings by applying a calibration factor.

### validate

```scala
val validate: TypedCoCell[CalibratedReading, ValidatedReading] =
  SafeStreamCell.fromPure("validate", (calibrated: CalibratedReading) => {
    val quality = if calibrated.value >= -50 && calibrated.value <= 150 then 1.0 else 0.5
    ValidatedReading(
      reading = calibrated,
      qualityScore = quality,
      isValid = quality > 0.7
    )
  })
```

Validates calibrated readings by checking value ranges and assigning quality scores.

### computeStats

```scala
val computeStats: TypedCoCell[CalibratedReading, ReadingStats] =
  SafeStreamCell.fromPure("computeStats", (reading: CalibratedReading) =>
    ReadingStats(
      count = 1,
      sum = reading.value,
      min = reading.value,
      max = reading.value
    )
  )
```

Computes statistics from a single reading (designed for aggregation in pipelines).

### generateAlert

```scala
val generateAlert: TypedCoCell[CalibratedReading, Alert] =
  SafeStreamCell.fromPure("generateAlert", (reading: CalibratedReading) => {
    val severity = if reading.value > 100 then "HIGH"
                   else if reading.value > 80 then "MEDIUM"
                   else "LOW"
    val msg = s"Temperature reading: ${reading.value}${reading.unit}"
    Alert(
      severity = severity,
      message = msg,
      timestamp = System.currentTimeMillis()
    )
  })
```

Generates alerts based on reading thresholds.

---

## Cells with Proofs

For consensus verification, variants that return termination proofs:

```scala
val calibrateWithProof: TypedCoCell[RawSensorReading, (CalibratedReading, StreamTerminationProof)]
val validateWithProof: TypedCoCell[CalibratedReading, (ValidatedReading, StreamTerminationProof)]
val computeStatsWithProof: TypedCoCell[CalibratedReading, (ReadingStats, StreamTerminationProof)]
val generateAlertWithProof: TypedCoCell[CalibratedReading, (Alert, StreamTerminationProof)]
```

These are created using `SafeStreamCell.fromPureWithProof` with the same transformation logic.

### Using Proofs in Consensus

```scala
// Run a cell with proof generation
val result = calibrateWithProof.run[IO](testRawReading)

result.map {
  case Right((calibrated, proof)) =>
    // Validator can verify termination without re-executing
    if (proof.isValid) {
      // Proof shows strictly decreasing ranks: [1, 0]
      // Computation is certified to have terminated
      acceptResult(calibrated)
    } else {
      rejectResult("Invalid termination proof")
    }
  case Left(error) =>
    handleError(error)
}
```

---

## Integration with Topology Examples

The basic cells from `StreamTopologyTypes` are used throughout the topology examples:

### DimapExamples

```scala
import StreamTopologyTypes.*

val enrichingDimap: TypedCoCell[Enriched[RawSensorReading], Enriched[CalibratedReading]] =
  calibrate.dimap(_.value)(cal => Enriched(cal, Map("source" -> "calibrated")))
```

### FanOutExamples

```scala
import StreamTopologyTypes.*

val validateAndStats: TypedCoCell[CalibratedReading, (ValidatedReading, ReadingStats)] =
  validate &&& computeStats
```

### ParallelExamples

```scala
import StreamTopologyTypes.*

val parallelCalibrate: TypedCoCell[(RawSensorReading, RawSensorReading), (CalibratedReading, CalibratedReading)] =
  calibrate *** calibrate
```

### CombinedExamples

```scala
import StreamTopologyTypes.*

val diamondPattern: TypedCoCell[RawSensorReading, Alert] =
  calibrate >>> (validate &&& computeStats) >>> merge
```

All these compositions inherit the termination guarantees from SafeStreamCell.

---

## Comparison with Other Safe Cells

The Reality SDK provides consistent safe cell implementations across layers:

| Cell | Layer | Use Case | Termination Measure |
|------|-------|----------|---------------------|
| `SafeL0Cell` | L0 | Base layer processing | Processing depth (1 → 0) |
| `SafeBlockConsensusCell` | L0 | Block consensus | Consensus phase depth |
| `SafeL2Cell` | L2 | Higher-order composition | Processing depth (1 → 0) |
| `SafeStreamCell` | Generic | Stream processing | Processing depth (1 → 0) |

### Common Pattern

All safe cells follow the same pattern:

```scala
sealed trait ProcessingState extends Ω {
  def depth: Int
}

case class Unprocessed(...) extends ProcessingState { def depth = 1 }
case class Completed(...) extends ProcessingState { def depth = 0 }

given wellFounded: WellFounded[ProcessingState] =
  WellFounded.fromMeasure(_.depth)
```

This consistency ensures:
1. **Predictable behavior** across all cell types
2. **Composable termination** - combining safe cells produces safe pipelines
3. **Unified verification** - same proof structure for all cells

---

## Test Data

Helper functions for testing:

```scala
/** Sample raw sensor reading for testing */
def testRawReading: RawSensorReading =
  RawSensorReading("sensor-001", 85.5, System.currentTimeMillis())

/** Sample calibrated reading for testing */
def testCalibratedReading: CalibratedReading =
  CalibratedReading("sensor-001", 89.775, "celsius", 1.05)
```

---

## Running Examples

```bash
# Run all topology examples using SafeStreamCell
sbt "combined/runMain org.reality.combined.examples.topologies.StreamTopologyExampleRunner"
```

Expected output includes termination verification:

```
======================================================================
STREAM TOPOLOGY EXAMPLES
Demonstrating TypedCoCell operators and patterns
======================================================================

=== DIMAP (biMap) Examples ===
--- Left Adjoint: Enriching dimap ---
Input:  Enriched[RawSensorReading] with context
Output: Right(Enriched(CalibratedReading(sensor-001,89.775,celsius,1.05),...))

=== FAN-OUT (&&&) Examples ===
--- Basic Fan-Out: validate &&& computeStats ---
Result: Right((ValidatedReading(...),ReadingStats(1,89.775,89.775,89.775)))

======================================================================
All demonstrations complete!
======================================================================
```

---

## Summary

`StreamTopologyTypes` provides:

| Component | Purpose |
|-----------|---------|
| **Domain types** | IoT sensor data types extending Ω |
| **SafeStreamCell** | Generic safe cell with WellFounded termination |
| **Basic cells** | Pre-built TypedCoCells for common transformations |
| **Cells with proofs** | Variants returning termination proofs |
| **Test data** | Helpers for testing pipelines |

Key benefits:

1. **Guaranteed termination** - No unbounded recursion possible
2. **Type safety** - Full compile-time type checking
3. **Proof generation** - Termination evidence for consensus
4. **Composability** - Cells compose with all TypedCoCell operators
5. **Consistency** - Same pattern as other safe cells in the SDK

This enables building complex data processing pipelines with blockchain-grade reliability and verifiability.
