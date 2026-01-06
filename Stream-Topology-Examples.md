# Stream Topology Examples

A comprehensive guide to TypedCoCell operators and topology patterns for building data processing pipelines with categorical foundations.

## Table of Contents

- [Overview](#overview)
- [Domain Types](#domain-types)
- [Basic Cells](#basic-cells)
- [Dimap (biMap) Operator](#dimap-bimap-operator)
- [Fan-Out (&&&) Operator](#fan-out--operator)
- [Parallel (***) Operator](#parallel--operator)
- [Wrap Operator](#wrap-operator)
- [Combined Topology Patterns](#combined-topology-patterns)
- [Adjoint Relationships](#adjoint-relationships)
- [Running the Examples](#running-the-examples)
- [Safe Hylo Versions](#safe-hylo-versions)

---

## Overview

The Stream Topology Examples demonstrate how to build complex data processing pipelines using TypedCoCell's categorical operators. Each operator has a precise mathematical meaning rooted in category theory:

| Operator | Signature | Category Theory | Description |
|----------|-----------|-----------------|-------------|
| `>>>` | `A→B >>> B→C = A→C` | Composition | Sequential pipeline |
| `&&&` | `A→B &&& A→C = A→(B,C)` | Product morphism | Fan-out to multiple outputs |
| `***` | `A→B *** C→D = (A,C)→(B,D)` | Bifunctor | Parallel independent processing |
| `dimap` | `(X→A, B→Y) => (A→B) => X→Y` | Profunctor | Transform both input and output |
| `wrap` | `(A→B) => (Either[E,B]→Either[E,C]) => A→C` | Result transformer | Transform cell results |

These operators satisfy the laws of their respective categorical structures, enabling equational reasoning about pipeline behavior.

---

## Domain Types

The examples use a sensor data processing domain defined in `StreamTopologyTypes.scala`:

```scala
package org.reality.combined.examples.topologies

import org.reality.kernel.Ω

// Raw sensor input
case class RawSensorReading(
  deviceId: String,
  value: Double,
  timestamp: Long
) extends Ω

// Calibrated reading after sensor correction
case class CalibratedReading(
  deviceId: String,
  value: Double,
  unit: String,
  calibrationFactor: Double
) extends Ω

// Validated reading with quality assessment
case class ValidatedReading(
  reading: CalibratedReading,
  qualityScore: Double,
  isValid: Boolean
) extends Ω

// Aggregated statistics
case class ReadingStats(
  count: Int,
  sum: Double,
  min: Double,
  max: Double
) extends Ω

// Alert notification
case class Alert(
  severity: String,
  message: String,
  timestamp: Long
) extends Ω

// Wrapper types for dimap examples
case class Enriched[A <: Ω](value: A, context: Map[String, String]) extends Ω
case class Tagged[A <: Ω](value: A, tag: String) extends Ω
```

---

## Basic Cells

Four fundamental cells form the building blocks for all topology examples:

```scala
// calibrate: RawSensorReading → CalibratedReading
val calibrate: TypedCoCell[RawSensorReading, CalibratedReading] =
  TypedCoCell.lift("calibrate") { raw =>
    CalibratedReading(
      deviceId = raw.deviceId,
      value = raw.value * 1.05,  // Apply calibration factor
      unit = "celsius",
      calibrationFactor = 1.05
    )
  }

// validate: CalibratedReading → ValidatedReading
val validate: TypedCoCell[CalibratedReading, ValidatedReading] =
  TypedCoCell.lift("validate") { cal =>
    val quality = if cal.value >= 0 && cal.value <= 100 then 1.0 else 0.5
    ValidatedReading(cal, quality, quality > 0.7)
  }

// computeStats: CalibratedReading → ReadingStats
val computeStats: TypedCoCell[CalibratedReading, ReadingStats] =
  TypedCoCell.lift("computeStats") { cal =>
    ReadingStats(count = 1, sum = cal.value, min = cal.value, max = cal.value)
  }

// generateAlert: CalibratedReading → Alert
val generateAlert: TypedCoCell[CalibratedReading, Alert] =
  TypedCoCell.lift("generateAlert") { cal =>
    val severity = if cal.value > 100 then "HIGH" else if cal.value > 80 then "MEDIUM" else "LOW"
    Alert(severity, s"Temperature reading: ${cal.value}${cal.unit}", System.currentTimeMillis())
  }
```

---

## Dimap (biMap) Operator

The `dimap` operator transforms both the input and output of a cell, making TypedCoCell a **profunctor**.

### Signature

```scala
def dimap[In2, Out2](f: In2 => In)(g: Out => Out2): TypedCoCell[In2, Out2]
```

### Left Adjoint: Contramap (Input Transformation)

Transforms the input type before processing. This is the "constructive" direction.

```scala
// Tagged[RawSensorReading] → CalibratedReading
val contramapExample: TypedCoCell[Tagged[RawSensorReading], CalibratedReading] =
  calibrate.contramap[Tagged[RawSensorReading]](_.value)
```

**Diagram:**
```
Tagged[Raw] ──extract──→ Raw ──calibrate──→ Calibrated
```

### Right Adjoint: Map (Output Transformation)

Transforms the output type after processing. This is the "forgetful" direction.

```scala
// RawSensorReading → Double (extracts just the value)
val mapExample: TypedCoCell[RawSensorReading, Double] =
  calibrate.map(_.value)
```

**Diagram:**
```
Raw ──calibrate──→ Calibrated ──extract──→ Double
```

### Bidirectional: Full Dimap

Transforms both input and output simultaneously.

```scala
// Enriched[RawSensorReading] → Enriched[CalibratedReading]
val enrichingDimap: TypedCoCell[Enriched[RawSensorReading], Enriched[CalibratedReading]] =
  calibrate.dimap[Enriched[RawSensorReading], Enriched[CalibratedReading]](
    _.value  // Extract raw from enriched
  )(cal =>   // Wrap result back in enriched
    Enriched(cal, Map(
      "calibrated_at" -> System.currentTimeMillis().toString,
      "calibration_version" -> "v2.1"
    ))
  )
```

**Diagram:**
```
Enriched[Raw] ──extract──→ Raw ──calibrate──→ Calibrated ──enrich──→ Enriched[Calibrated]
```

### Verification: Roundtrip Property

```scala
// enrich >>> extract = calibrate (up to isomorphism)
val roundTrip: TypedCoCell[RawSensorReading, CalibratedReading] =
  TypedCoCell.lift[RawSensorReading, Enriched[RawSensorReading]]("enrich")(r =>
    Enriched(r, Map())
  ) >>> enrichingDimap >>> TypedCoCell.lift("extract")(_.value)

// This should produce the same result as calibrate directly
```

---

## Fan-Out (&&&) Operator

The `&&&` operator creates a **product morphism** from a shared input, witnessing the universal property of products.

### Signature

```scala
def &&&[Out2](other: TypedCoCell[In, Out2]): TypedCoCell[In, (Out, Out2)]
```

### Left Adjoint: Diagonal (Δ)

The diagonal functor duplicates input, enabling fan-out.

```scala
// Δ: CalibratedReading → (CalibratedReading, CalibratedReading)
val diagonal: TypedCoCell[CalibratedReading, (CalibratedReading, CalibratedReading)] =
  TypedCoCell.diagonal[CalibratedReading]
```

**Categorical meaning:** Δ is the left adjoint to the product functor (×).

### Right Adjoint: Projections (π₁, π₂)

Projections extract components from products.

```scala
// π₁: (ValidatedReading, ReadingStats) → ValidatedReading
val projectFirst: TypedCoCell[(ValidatedReading, ReadingStats), ValidatedReading] =
  TypedCoCell.fst[ValidatedReading, ReadingStats]

// π₂: (ValidatedReading, ReadingStats) → ReadingStats
val projectSecond: TypedCoCell[(ValidatedReading, ReadingStats), ReadingStats] =
  TypedCoCell.snd[ValidatedReading, ReadingStats]
```

### Basic Fan-Out

```scala
// CalibratedReading → (ValidatedReading, ReadingStats)
val validateAndStats: TypedCoCell[CalibratedReading, (ValidatedReading, ReadingStats)] =
  validate &&& computeStats
```

**Diagram:**
```
           ┌─── validate ───→ ValidatedReading ───┐
Calibrated─┤                                      ├─→ (Validated, Stats)
           └─── computeStats ──→ ReadingStats ────┘
```

### Triple Fan-Out

```scala
// Left-associated: ((Validated, Stats), Alert)
val tripleLeft: TypedCoCell[CalibratedReading, ((ValidatedReading, ReadingStats), Alert)] =
  (validate &&& computeStats) &&& generateAlert

// Right-associated: (Validated, (Stats, Alert))
val tripleRight: TypedCoCell[CalibratedReading, (ValidatedReading, (ReadingStats, Alert))] =
  validate &&& (computeStats &&& generateAlert)
```

### Universal Property Verification

The defining property of products: π₁ ∘ ⟨f,g⟩ = f and π₂ ∘ ⟨f,g⟩ = g

```scala
// These should be equal:
val viaFanOut: TypedCoCell[CalibratedReading, ValidatedReading] =
  (validate &&& computeStats) >>> TypedCoCell.fst

val direct: TypedCoCell[CalibratedReading, ValidatedReading] =
  validate

// viaFanOut.run(input) == direct.run(input) for all inputs
```

---

## Parallel (***) Operator

The `***` operator applies two cells in parallel on independent inputs, implementing the **bifunctor** action on products.

### Signature

```scala
def ***[In2, Out2](other: TypedCoCell[In2, Out2]): TypedCoCell[(In, In2), (Out, Out2)]
```

### Key Relationship with Fan-Out

```
Δ >>> (f *** g) = f &&& g
```

This equation shows that fan-out can be decomposed into diagonal followed by parallel.

```scala
// These are equivalent:
val viaParallel: TypedCoCell[CalibratedReading, (ValidatedReading, ReadingStats)] =
  TypedCoCell.diagonal[CalibratedReading] >>> (validate *** computeStats)

val viaFanOut: TypedCoCell[CalibratedReading, (ValidatedReading, ReadingStats)] =
  validate &&& computeStats

// viaParallel.run(input) == viaFanOut.run(input) for all inputs
```

### Basic Parallel

```scala
// (CalibratedReading, CalibratedReading) → (ValidatedReading, ReadingStats)
val parallelValidateAndStats: TypedCoCell[(CalibratedReading, CalibratedReading), (ValidatedReading, ReadingStats)] =
  validate *** computeStats
```

**Diagram:**
```
(Calibrated₁, Calibrated₂) ──→ (validate(Cal₁), computeStats(Cal₂)) ──→ (Validated, Stats)
```

### Heterogeneous Parallel

Different input types processed in parallel:

```scala
// (RawSensorReading, CalibratedReading) → (CalibratedReading, ValidatedReading)
val heterogeneousParallel: TypedCoCell[(RawSensorReading, CalibratedReading), (CalibratedReading, ValidatedReading)] =
  calibrate *** validate
```

### Identity Laws

```scala
// id *** f = first(f): keeps first component unchanged
val parallelWithIdLeft: TypedCoCell[(CalibratedReading, CalibratedReading), (CalibratedReading, ValidatedReading)] =
  TypedCoCell.id[CalibratedReading] *** validate

// f *** id = second(f): keeps second component unchanged
val parallelWithIdRight: TypedCoCell[(CalibratedReading, CalibratedReading), (ValidatedReading, CalibratedReading)] =
  validate *** TypedCoCell.id[CalibratedReading]
```

### Fork-Join Pattern

```scala
val forkJoin: TypedCoCell[(RawSensorReading, RawSensorReading), ReadingStats] = {
  val join: TypedCoCell[(CalibratedReading, CalibratedReading), ReadingStats] =
    TypedCoCell.lift("join") { case (c1, c2) =>
      ReadingStats(
        count = 2,
        sum = c1.value + c2.value,
        min = Math.min(c1.value, c2.value),
        max = Math.max(c1.value, c2.value)
      )
    }

  (calibrate *** calibrate) >>> join
}
```

**Diagram:**
```
        ┌─── calibrate ───┐
(Raw₁, Raw₂)              ├─── join ───→ Stats
        └─── calibrate ───┘
```

---

## Wrap Operator

The `wrap` operator creates derived cells by transforming results, enabling error handling, validation, and output transformation without reimplementing core logic.

### Signature

```scala
def wrap[A, B, C](name: String, underlying: TypedCoCell[A, B])(
  transform: Either[CellError, B] => Either[CellError, C]
): TypedCoCell[A, C]
```

### Output Transformation

```scala
case class ReadingSummary(deviceId: String, value: Double, status: String) extends Ω

val calibrateToSummary: TypedCoCell[RawSensorReading, ReadingSummary] =
  TypedCoCell.wrap("calibrateToSummary", calibrate) {
    case Right(cal) =>
      val status = if cal.value > 100 then "HOT" else if cal.value < 0 then "COLD" else "NORMAL"
      Right(ReadingSummary(cal.deviceId, cal.value, status))
    case Left(err) =>
      Left(err)
  }
```

### Error Recovery

```scala
val calibrateWithDefault: TypedCoCell[RawSensorReading, CalibratedReading] =
  TypedCoCell.wrap("calibrateWithDefault", calibrate) {
    case Right(cal) => Right(cal)
    case Left(_) =>
      // Provide safe default on any error
      Right(CalibratedReading(
        deviceId = "unknown",
        value = 0.0,
        unit = "celsius",
        calibrationFactor = 1.0
      ))
  }
```

### Validation Layer

```scala
val calibrateWithValidation: TypedCoCell[RawSensorReading, CalibratedReading] =
  TypedCoCell.wrap("calibrateWithValidation", calibrate) {
    case Right(cal) if cal.value >= -40 && cal.value <= 200 =>
      Right(cal)
    case Right(cal) =>
      Left(CellError(s"Calibrated value ${cal.value} outside acceptable range [-40, 200]"))
    case Left(err) =>
      Left(err)
  }
```

### Chained Wraps

Wraps can be composed to build up functionality layer by layer:

```scala
val fullyWrappedCalibrate: TypedCoCell[RawSensorReading, ReadingSummary] = {
  // Layer 1: Validation
  val validated = TypedCoCell.wrap("validated", calibrate) {
    case Right(cal) if cal.value >= -40 && cal.value <= 200 => Right(cal)
    case Right(cal) => Left(CellError(s"Out of range: ${cal.value}"))
    case Left(err) => Left(err)
  }

  // Layer 2: Recovery
  val withRecovery = TypedCoCell.wrap("withRecovery", validated) {
    case Right(cal) => Right(cal)
    case Left(_) => Right(CalibratedReading("fallback", 20.0, "celsius", 1.0))
  }

  // Layer 3: Transform to summary
  TypedCoCell.wrap("toSummary", withRecovery) {
    case Right(cal) =>
      val status = if cal.deviceId == "fallback" then "FALLBACK" else "OK"
      Right(ReadingSummary(cal.deviceId, cal.value, status))
    case Left(err) => Left(err)
  }
}
```

### Wrap with Parallel

```scala
val wrappedParallel: TypedCoCell[(RawSensorReading, RawSensorReading), ReadingSummary] = {
  val parallel = calibrate *** calibrate

  TypedCoCell.wrap("mergeParallel", parallel) {
    case Right((cal1, cal2)) =>
      val avgValue = (cal1.value + cal2.value) / 2.0
      val status = if cal1.deviceId == cal2.deviceId then "SAME_DEVICE" else "MULTI_DEVICE"
      Right(ReadingSummary(s"${cal1.deviceId}+${cal2.deviceId}", avgValue, status))
    case Left(err) =>
      Left(err)
  }
}
```

---

## Combined Topology Patterns

### Diamond Pattern

Classic diamond: fan-out then merge.

```scala
val diamondPattern: TypedCoCell[RawSensorReading, Alert] = {
  val merge: TypedCoCell[(ValidatedReading, ReadingStats), Alert] =
    TypedCoCell.lift("merge") { case (v, s) =>
      Alert("INFO", s"Valid=${v.isValid}, Sum=${s.sum}", System.currentTimeMillis())
    }

  calibrate >>> (validate &&& computeStats) >>> merge
}
```

**Diagram:**
```
           ┌─── validate ───┐
calibrate ─┤                ├─── merge ─── Alert
           └─── stats ──────┘
```

### Broadcast Pattern

Single input broadcast to multiple processors:

```scala
val broadcastPattern: TypedCoCell[CalibratedReading, ((ValidatedReading, ReadingStats), Alert)] =
  (validate &&& computeStats) &&& generateAlert
```

**Diagram:**
```
          ┌─── validate ────┐
Calibrated┼─── stats ───────┼─── collect
          └─── alert ───────┘
```

### Scatter-Gather Pattern

```scala
val scatterGather: TypedCoCell[CalibratedReading, ReadingStats] = {
  val path1: TypedCoCell[CalibratedReading, Double] =
    TypedCoCell.lift("path1")(_.value * 2)

  val path2: TypedCoCell[CalibratedReading, Double] =
    TypedCoCell.lift("path2")(_.value + 10)

  val gather: TypedCoCell[(Double, Double), ReadingStats] =
    TypedCoCell.lift("gather") { case (v1, v2) =>
      ReadingStats(2, v1 + v2, Math.min(v1, v2), Math.max(v1, v2))
    }

  (path1 &&& path2) >>> gather
}
```

**Diagram:**
```
          ┌─── path1 (×2) ───┐
Calibrated┤                  ├─── gather ─── Stats
          └─── path2 (+10) ──┘
```

### Nested Diamond

Diamond within diamond:

```scala
val nestedDiamond: TypedCoCell[RawSensorReading, Alert] = {
  // Inner diamond: two stat computations joined
  val statsPath: TypedCoCell[CalibratedReading, ReadingStats] = {
    val stat1 = computeStats
    val stat2 = computeStats.map(s => s.copy(sum = s.sum * 2))
    val joinStats: TypedCoCell[(ReadingStats, ReadingStats), ReadingStats] =
      TypedCoCell.lift("joinStats") { case (s1, s2) =>
        ReadingStats(s1.count + s2.count, s1.sum + s2.sum, s1.min, s2.max)
      }
    (stat1 &&& stat2) >>> joinStats
  }

  // Outer diamond
  val merge: TypedCoCell[(ValidatedReading, ReadingStats), Alert] =
    TypedCoCell.lift("merge") { case (v, s) =>
      Alert("NESTED", s"Valid=${v.isValid}, TotalSum=${s.sum}", System.currentTimeMillis())
    }

  calibrate >>> (validate &&& statsPath) >>> merge
}
```

### Pipeline Stages

Multi-stage sequential processing:

```scala
val pipelineStages: TypedCoCell[RawSensorReading, Alert] = {
  val processValidated: TypedCoCell[ValidatedReading, Alert] =
    TypedCoCell.lift("processValidated") { v =>
      val severity = if v.isValid then "OK" else "INVALID"
      Alert(severity, s"Quality: ${v.qualityScore}", System.currentTimeMillis())
    }

  calibrate >>> validate >>> processValidated
}
```

**Diagram:**
```
Raw ──→ calibrate ──→ validate ──→ process ──→ Alert
```

---

## Adjoint Relationships

The operators form adjoint pairs in category theory:

### Δ ⊣ × (Diagonal ⊣ Product)

```
Hom(Δ(A), (B, C)) ≅ Hom(A, B) × Hom(A, C)
```

- **Left adjoint (Δ):** Duplicates input for fan-out
- **Right adjoint (×):** Products with projections π₁, π₂

**Witnessed by:**
```scala
// Fan-out creates the adjunction
val fanOut: TypedCoCell[A, (B, C)] = f &&& g

// Universal property: π₁ ∘ ⟨f,g⟩ = f
(f &&& g) >>> fst == f

// Universal property: π₂ ∘ ⟨f,g⟩ = g
(f &&& g) >>> snd == g
```

### Profunctor Structure (dimap)

The dimap operation makes TypedCoCell a profunctor in the category of Ω types:

```
dimap: (X → A) × (B → Y) → (A → B) → (X → Y)
```

- **Contravariant in input:** `contramap` transforms inputs
- **Covariant in output:** `map` transforms outputs

**Laws:**
```scala
// Identity
cell.dimap(identity)(identity) == cell

// Composition
cell.dimap(f1 andThen f2)(g1 andThen g2) ==
  cell.dimap(f2)(g1).dimap(f1)(g2)
```

### Parallel *** as Bifunctor

The `***` operator is the bifunctor action on morphisms:

```
*** : (A → B) × (C → D) → (A × C) → (B × D)
```

**Laws:**
```scala
// Identity
id *** id == id

// Composition
(f1 >>> f2) *** (g1 >>> g2) == (f1 *** g1) >>> (f2 *** g2)

// Relationship to fan-out
diagonal >>> (f *** g) == f &&& g
```

---

## Running the Examples

Each example file has a standalone runner:

```bash
# Run all examples
sbt "combined/runMain org.reality.combined.examples.topologies.StreamTopologyExampleRunner"

# Run individual example suites
sbt "combined/runMain org.reality.combined.examples.topologies.DimapExamplesRunner"
sbt "combined/runMain org.reality.combined.examples.topologies.FanOutExamplesRunner"
sbt "combined/runMain org.reality.combined.examples.topologies.ParallelExamplesRunner"
sbt "combined/runMain org.reality.combined.examples.topologies.WrapExamplesRunner"
sbt "combined/runMain org.reality.combined.examples.topologies.CombinedExamplesRunner"
```

### Expected Output

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
Input:  CalibratedReading
Output: (ValidatedReading, ReadingStats)
Result: Right((ValidatedReading(...),ReadingStats(1,89.775,89.775,89.775)))

--- Right Adjoint: Projections π₁, π₂ ---
Demonstrating universal property: π₁ ∘ ⟨f,g⟩ = f
Equal: true

=== PARALLEL (***) Examples ===
--- Key Relationship: Δ >>> (f *** g) = f &&& g ---
Equal: true

=== WRAP Examples ===
--- Validation Wrap: Range Check ---
calibrate with validation: Right(CalibratedReading(sensor-001,89.775,celsius,1.05))
calibrate extreme value (500.0): Left(CellError: Calibrated value 525.0 outside acceptable range [-40, 200])

=== COMBINED Examples ===
--- Diamond Pattern ---
       ┌─── validate ───┐
cal ───┤                ├─── merge
       └─── stats ──────┘
Result: Right(Alert(INFO,Valid=true, Sum=89.775,...))

======================================================================
All demonstrations complete!
======================================================================
```

---

## Safe Hylo Versions

All topology patterns have safe hylo versions in `CombinedSafeHyloExamples.scala` that use:
- `SafeStreamCell` with `safeCoalgebra` / `safeAlgebra` pattern
- `TypedCoCellWithCellAndProof` for termination proofs
- WellFounded measure that strictly decreases

### SafeStreamCell Pattern

```scala
object SafeStreamCell {
  // Coalgebra: unfold input into ProcessingState
  def safeCoalgebra[F[_]: Async, A](input: A): F[ProcessingState[A]] =
    Processing(input).pure[F]

  // Algebra: fold ProcessingState into result
  def safeAlgebra[F[_]: Async, A](state: ProcessingState[A]): F[Either[CellError, A]] =
    state match {
      case Processing(value) => value.asRight[CellError].pure[F]
      case Completed(result) => result.pure[F]
    }

  // Create safe cell from pure transformation
  def apply[In <: Ω, Out <: Ω](
    cellName: String,
    transform: In => Out
  ): TypedCoCellWithCellAndProof[In, Out]
}
```

### Safe Base Cells

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

val safeValidate: TypedCoCellWithCellAndProof[CalibratedReading, ValidatedReading] =
  SafeStreamCell("validate", (calibrated: CalibratedReading) => {
    val quality = if calibrated.value >= -50 && calibrated.value <= 150 then 1.0 else 0.5
    ValidatedReading(
      reading = calibrated,
      qualityScore = quality,
      isValid = quality > 0.7
    )
  })
```

### Safe Composed Topologies

Composed topologies use TypedCoCell operators (`>>>`, `&&&`, `***`), NOT pipelines wrapped in cells:

```scala
// Safe Full Pipeline: Raw → ((Validated, Stats), Alert)
val safeFullPipeline: TypedCoCell[RawSensorReading, ((ValidatedReading, ReadingStats), Alert)] =
  safeCalibrate >>> ((safeValidate &&& safeComputeStats) &&& safeGenerateAlert)

// Safe Diamond Pattern
val safeDiamondPattern: TypedCoCell[RawSensorReading, Alert] = {
  val safeMerge = SafeStreamCell.forPair("merge", (v: ValidatedReading, s: ReadingStats) =>
    Alert("INFO", s"Valid=${v.isValid}, Sum=${s.sum}", System.currentTimeMillis())
  )
  val tupleToΩPair = TypedCoCell.lift("toΩPair") { case (v, s) => ΩPair(v, s) }

  safeCalibrate >>> (safeValidate &&& safeComputeStats) >>> tupleToΩPair >>> safeMerge
}

// Safe Scatter-Gather Pattern
val safeScatterGather: TypedCoCell[CalibratedReading, ReadingStats] = {
  val safePath1 = SafeStreamCell("path1", (c: CalibratedReading) => ΩDouble(c.value * 2))
  val safePath2 = SafeStreamCell("path2", (c: CalibratedReading) => ΩDouble(c.value + 10))
  val safeGather = SafeStreamCell.forPair("gather", (v1: ΩDouble, v2: ΩDouble) =>
    ReadingStats(2, v1.value + v2.value, Math.min(v1.value, v2.value), Math.max(v1.value, v2.value))
  )
  val tupleToΩPair = TypedCoCell.lift("toΩPair") { case (d1, d2) => ΩPair(d1, d2) }

  (safePath1 &&& safePath2) >>> tupleToΩPair >>> safeGather
}
```

### Termination Proofs

Each safe cell provides termination proofs:

```scala
val proof = safeCalibrate.terminationProof(testInput)
// CellTerminationProof(calibrate, 1 → 0, valid=true)

val windowProof = safeWindowedAggregation.terminationProof(ΩPair(testCal, cal2))
// CellTerminationProof(windowAggregate, 1 → 0, valid=true)
```

### ΩPair and ΩDouble Wrappers

For Cell infrastructure compatibility (tuples don't extend Ω):

```scala
case class ΩPair[A, B](first: A, second: B) extends Ω
case class ΩDouble(value: Double) extends Ω
```

### Running Safe Hylo Examples

```bash
sbt "combined/runMain org.reality.combined.examples.topologies.CombinedSafeHyloExamplesRunner"
```

---

## File Structure

```
modules/combined/src/main/scala/org/reality/combined/examples/topologies/
├── StreamTopologyTypes.scala          # Domain types and basic cells
├── DimapExamples.scala                # dimap operator examples
├── FanOutExamples.scala               # &&& operator examples
├── ParallelExamples.scala             # *** operator examples
├── WrapExamples.scala                 # TypedCoCell.wrap examples
├── CombinedExamples.scala             # Combined topology patterns
├── CombinedSafeHyloExamples.scala     # Safe hylo versions with proofs
└── StreamTopologyExample.scala        # Master runner
```

---

## Summary

| Pattern | Operators Used | Use Case |
|---------|---------------|----------|
| Sequential Pipeline | `>>>` | Linear data transformation |
| Fan-Out | `&&&` | One input → multiple outputs |
| Parallel | `***` | Independent inputs processed together |
| Diamond | `&&&` + `>>>` | Fan-out then merge |
| Fork-Join | `***` + `>>>` | Parallel processing with join |
| Scatter-Gather | `&&&` + `>>>` | Multiple paths merged |
| Wrapped | `wrap` | Error handling, validation, transformation |
| Profunctor | `dimap` | Transform both ends of a cell |

These patterns can be freely combined to build arbitrarily complex data processing topologies while maintaining type safety and categorical consistency.
