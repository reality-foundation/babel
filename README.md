# Babel - Reality SDK Documentation

A comprehensive guide to building distributed applications with the Reality SDK using categorical foundations, halt-free computation, and ZK-WASM execution.

## Table of Contents

### Part 1: Getting Started
- [Reality SDK Usage Guide](#reality-sdk-usage-guide)

### Part 2: Core Abstractions
- [CoCell and TypedCoCell Documentation](#cocell-and-typedcocell-documentation)
  - [Cats Effect Compatibility](#cats-effect-compatibility)

### Part 3: Safe Patterns and ZK-WASM
- [Safe Hylo and ZK-WASM Execution Guide](#safe-hylo-and-zk-wasm-execution-guide)
  - [Turing Completeness with Finite Circuits](#turing-completeness-with-finite-circuits)

### Part 4: Theoretical Foundations
- [Halt-Free Computation](#halt-free-computation-in-the-babel)

### Part 5: Practical Examples
- [Stream Topology Examples](#stream-topology-examples)
- [StreamTopologyTypes Reference](#streamtopologytypes-safe-stream-cells-with-termination-guarantees)

---

# Reality SDK Usage Guide

A comprehensive guide to using the Reality SDK for building distributed applications (rApps). This guide references the example implementations in `org.reality.combined.examples`.

## Table of Contents

- [Overview](#overview)
- [Architecture Layers](#architecture-layers)
- [Getting Started](#getting-started)
- [Example Files Reference](#example-files-reference)
- [Untyped CoCell Approach](#untyped-cocell-approach)
- [Typed CoCell Approach](#typed-cocell-approach)
- [BabelApp Framework](#babelapp-framework)
- [Configuration](#configuration)
- [State Types](#state-types)
- [Composition Patterns](#composition-patterns)
- [Running Your Application](#running-your-application)

---

## Overview

The Reality SDK provides two primary approaches for building distributed consensus applications:

| Approach | Type Safety | Framework | Best For |
|----------|-------------|-----------|----------|
| **Untyped (CoCell)** | Runtime | UntypedBabelApp | Existing apps, rapid prototyping |
| **Typed (TypedCoCell)** | Compile-time | BabelApp | New development, type-safe pipelines |

Both approaches support the same underlying infrastructure and can be mixed within a project.

---

## Architecture Layers

Reality applications are structured in layers:

```
┌─────────────────────────────────────┐
│          State Channel (L1)          │  ← Consensus, WASM execution
├─────────────────────────────────────┤
│            Base Layer (L0)           │  ← Core blockchain operations
├─────────────────────────────────────┤
│           Reality SDK/Kernel         │  ← Infrastructure, networking
└─────────────────────────────────────┘
```

- **L0 (Base Layer)**: Handles core blockchain operations, block production, and peer discovery
- **L1 (State Channel)**: Implements application-specific logic, consensus, and optionally WASM execution

---

## Getting Started

### Prerequisites

1. Java 17+
2. SBT 1.9+
3. Scala 3.7.3

### Building the Reality Combined JAR

The reality-combined module bundles all internal modules (kernel, shared, sdk, dagL1, core, aci, etc.) into a single JAR with sources for IDE navigation.

```bash
# In the reality project directory
cd /path/to/reality

# Build both the main JAR and sources JAR
sbt "combined/package; combined/packageSrc"

# Output files:
# modules/combined/target/scala-3.7.3/reality-combined_3-VERSION.jar         (classes)
# modules/combined/target/scala-3.7.3/reality-combined_3-VERSION-sources.jar (sources)
```

### Project Setup

#### 1. Copy JARs to your project

```bash
# Create lib directory if needed
mkdir -p /path/to/your-project/lib

# Copy both JARs
cp modules/combined/target/scala-3.7.3/reality-combined_3-*.jar /path/to/your-project/lib/
```

#### 2. Configure project/Dependencies.scala

Create or update `project/Dependencies.scala` with versions matching reality:

```scala
import sbt._

object Dependencies {

  object V {
    val scala = "3.7.3"

    // Match reality project versions for compatibility
    val apacheDerby = "10.15.2.0"
    val betterFiles = "3.9.2"
    val bitcoinj = "0.17"
    val bouncyCastle = "1.82"
    val breeze = "2.1.0"
    val cats = "2.13.0"
    val catsEffect = "3.6.3"
    val catsRetry = "4.0.0"
    val circe = "0.14.15"
    val ciris = "3.11.0"
    val comcast = "3.7.0"
    val decline = "2.5.0"
    val doobie = "1.0.0-RC10"
    val droste = "0.10.0"
    val flyway = "11.14.0"
    val fs2 = "3.12.2"
    val fs2Data = "1.12.0"
    val http4s = "0.23.32"
    val http4sJwtAuth = "2.0.11"
    val iron = "3.2.0"
    val izumi = "3.0.6"
    val javaIpfsHttpClient = "v1.4.1"
    val jawnVersion = "1.6.0"
    val jawnFs2Version = "2.4.0"
    val kittens = "3.5.0"
    val log4cats = "2.7.1"
    val logback = "1.5.19"
    val logstashLogbackEncoder = "8.1"
    val mapref = "0.2.0-M2"
    val micrometer = "1.15.4"
    val monocle = "3.3.0"
    val pureconfig = "0.17.9"
    val wasmtime = "0.19.0"
    val weaver = "0.10.1"
    val web3j = "4.13.0"
  }

  object Libraries {
    // Core libraries
    val cats = "org.typelevel" %% "cats-core" % V.cats
    val catsEffect = "org.typelevel" %% "cats-effect" % V.catsEffect
    // ... (see full Dependencies.scala in examples)
  }
}
```

#### 3. Configure build.sbt

```scala
import Dependencies._

ThisBuild / organization := "com.example"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := V.scala

lazy val root = (project in file("."))
  .settings(
    name := "my-rapp",
    libraryDependencies ++= Seq(
      Libraries.cats,
      Libraries.catsEffect,
      // ... add required libraries
    ),
    testFrameworks += new TestFramework("weaver.framework.CatsEffect")
  )
```

### IntelliJ IDEA Setup

After copying the JARs to `lib/`, configure IntelliJ to use the sources:

1. **Reload the SBT project** - Right-click `build.sbt` → Reload
2. **Attach sources** (if not automatically detected):
   - Open **Project Structure** (⌘;)
   - Go to **Libraries** or **Modules → Dependencies**
   - Find the `reality-combined` JAR entry
   - Click **Edit** or the **+** button
   - Under **Sources**, add the `-sources.jar` file from `lib/`
   - Click **OK** and **Apply**

**Note:** Due to Scala 3's `.tasty` files, ⌘+B may initially show decompiled interfaces. Navigate to the package in the Project panel to access actual source files.

### Minimal Example

```scala
import org.reality.combined.examples._

// Typed approach (recommended for new projects)
object MyApp extends TypedCocellBabelExample

// Or untyped approach
object MyApp extends ComposeExamples
```

---

## Example Files Reference

All examples are in `org.reality.combined.examples`:

| File | Description | Approach |
|------|-------------|----------|
| `TypedCocellExample.scala` | State types and base typed cells | Typed |
| `TypedCocellBabelExample.scala` | Single-pipeline BabelApp example | Typed |
| `ComposeMultipleTypedCocellBabelExample.scala` | Multi-pipeline BabelApp example | Typed |
| `HigherOrderTypedCoCellBabelExample.scala` | 3-layer architecture with L2 | Typed |
| `ConfiguredBabelExample.scala` | Programmatic configuration helpers | Typed |
| `CombinedMonoidExample.scala` | Untyped CoCell definitions | Untyped |
| `ComposeExamples.scala` | Simple untyped composition | Untyped |
| `ComposeMultipleExample.scala` | Multi-node untyped composition | Untyped |

---

## Untyped CoCell Approach

The untyped approach uses `CoCell` and `Portal` for maximum flexibility.

### CombinedMonoidExample.scala

Defines the core untyped CoCell implementations:

```scala
// L0 CoCell - base layer node
case class CombinedL0(staticConfig: Option[L0CoCellConfig] = None) extends CoCell {
  override val config: Option[CoCellConfig] = staticConfig
  override val name: String = staticConfig.map(_.name).getOrElse("CombinedL0")

  def mkCell[F[_]: Async](
    l1OutputQueue: Queue[F, Signed[NETBlock]],
    stateChannelOutputQueue: Queue[F, StateChannelOutput]
  ): Ω => Cell[F, StackF, Ω, Ω, Either[CellError, Ω]] = data => {
    val left = L0Cell.mkCell(l1OutputQueue, stateChannelOutputQueue)
    val right = EmptyCellObj.mkCell
    Cell.cellMonoid[F, StackF].combine(left(data), right(data))
  }
}

// L1 CoCell - state channel node
case class CombinedStateChannel(staticConfig: Option[L1CoCellConfig] = None) extends CoCell {
  override val config: Option[CoCellConfig] = staticConfig
  val stateChannel: MkStateChannel = L1

  override def mkResources[A <: CliMethod]: (A, SDK[IO]) => Resource[IO, HttpApi[IO]] =
    L1HttpApi.mkResources(_: A, _: SDK[IO])

  def mkCell[F[_]: {Async, SecurityProvider, Random}](
    ctx: BlockConsensusContext[F]
  ): Ω => Cell[F, StackF, Ω, Ω, Either[CellError, Ω]] =
    BlockConsensusCell.mkCell[F](ctx)
}
```

### ComposeExamples.scala

Simple two-node composition using the `>>>` operator:

```scala
open class ComposeExamples extends UntypedBabelApp {
  val combinedL0: CombinedL0 = CombinedL0()
  val combinedStateChannel: CombinedStateChannel = CombinedStateChannel()

  // Compose L0 >>> L1
  val composed: CoCell = combinedL0 >>> combinedStateChannel

  val cellProgram = (args: List[String]) =>
    for {
      cellInstances <- composed.setup(args)
    } yield cellInstances :: Nil
}
```

---

## Typed CoCell Approach

The typed approach uses `TypedCoCell` and `BabelApp` for compile-time type safety.

### TypedCocellExample.scala

Defines the typed state representations and base typed cells:

#### State Types

```scala
/** L0 layer input state */
case class L0InputState(
  data: Ω,
  metadata: Map[String, String] = Map.empty
) extends Ω

/** L0 layer output state */
case class L0OutputState(
  processedData: Ω,
  l0Hash: Option[String] = None,
  metadata: Map[String, String] = Map.empty
) extends Ω

/** State channel output */
case class StateChannelState(
  finalData: Ω,
  stateChannelHash: Option[String] = None,
  consensusReached: Boolean = false,
  metadata: Map[String, String] = Map.empty
) extends Ω
```

#### Typed L0 Cell

```scala
object CombinedL0Typed {
  def apply(staticConfig: Option[L0CoCellConfig] = None): TypedCoCell[L0InputState, L0OutputState] =
    new TypedCoCell[L0InputState, L0OutputState] {
      override val name: String = staticConfig.map(_.name).getOrElse("CombinedL0Typed")

      override def run[F[_]: Async](input: L0InputState): F[Either[CellError, L0OutputState]] =
        Async[F].pure {
          Right(L0OutputState(
            processedData = input.data,
            l0Hash = Some(s"l0-${input.data.hashCode.toHexString}"),
            metadata = input.metadata + ("layer" -> "L0")
          ))
        }
    }

  def withConfig(cfg: L0CoCellConfig): TypedCoCell[L0InputState, L0OutputState] =
    apply(Some(cfg))
}
```

---

## BabelApp Framework

BabelApp provides the runtime infrastructure for typed pipelines.

### TypedCocellBabelExample.scala

Single-pipeline BabelApp application:

```scala
open class TypedCocellBabelExample extends BabelApp[L0InputState, StateChannelState] {

  /** L0 layer node */
  val l0Node: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(
      CombinedL0Typed.withConfig(StaticConfigs.SingleNodeGenesis.l0Config),
      Some(StaticConfigs.SingleNodeGenesis.l0Config)
    )

  /** L1/StateChannel layer node */
  val stateChannelNode: TypedCoCellNode[L0OutputState, StateChannelState] =
    BabelApp.l1Node(
      CombinedStateChannelTyped.withConfig(StaticConfigs.SingleNodeGenesis.l1Config),
      L1,
      Some(StaticConfigs.SingleNodeGenesis.l1Config)
    )

  /** Type-safe pipeline */
  val pipelines: List[CoCellPipeline[L0InputState, StateChannelState]] =
    List(CoCellPipeline.from(l0Node) >>> stateChannelNode)
}
```

---

## Configuration

### StaticConfigs

Pre-defined configurations for common scenarios:

```scala
import org.reality.combined.StaticConfigs

// Single node (genesis + initial validator)
StaticConfigs.SingleNodeGenesis.l0Config   // L0 genesis: ports 9000-9002
StaticConfigs.SingleNodeGenesis.l1Config   // L1 initial: ports 9010-9012
StaticConfigs.SingleNodeGenesis.l0Config2  // L0 validator: ports 9020-9022
StaticConfigs.SingleNodeGenesis.l1Config2  // L1 validator: ports 9030-9032
```

### Port Allocation Convention

| Layer | Node | Public | P2P | CLI |
|-------|------|--------|-----|-----|
| L0 | Genesis | 9000 | 9001 | 9002 |
| L1 | Initial | 9010 | 9011 | 9012 |
| L0 | Validator | 9020 | 9021 | 9022 |
| L1 | Validator | 9030 | 9031 | 9032 |

---

## Composition Patterns

### Sequential (`>>>`)

Chain cells where output of one feeds into the next:

```scala
val pipeline: TypedCoCell[A, D] = cellAB >>> cellBC >>> cellCD
```

### Parallel (`***`)

Process tuple components concurrently:

```scala
val parallel: TypedCoCell[(A, B), (C, D)] = cellAC *** cellBD
```

### Fan-out (`&&&`)

Same input to multiple cells:

```scala
val fanOut: TypedCoCell[A, (B, C)] = cellAB &&& cellAC
```

---

## Running Your Application

### With Environment Variables

```bash
export NODE_TYPE=genesis  # or validator1, validator2
sbt run
```

### With Docker

```yaml
services:
  genesis:
    environment:
      - NODE_TYPE=genesis
    ports:
      - "9000-9002:9000-9002"
      - "9010-9012:9010-9012"

  validator1:
    environment:
      - NODE_TYPE=validator1
    depends_on:
      - genesis
```

---

# CoCell and TypedCoCell Documentation

A comprehensive guide to the CoCell abstraction layer for building distributed consensus systems with categorical foundations.

## Overview

The CoCell system provides two complementary abstractions for building blockchain state transformations:

| Abstraction | Type Safety | Use Case |
|-------------|-------------|----------|
| **CoCell** | Runtime (dynamic) | Existing apps, rapid deployment |
| **TypedCoCell** | Compile-time (static) | New development, type-safe pipelines |

Both share the same categorical structure and composition operators (`>>>`), but differ in when type compatibility is verified.

---

## Cats Effect Compatibility

The product and coproduct types used throughout the SDK are fully compatible with Cats Effect. They're built on standard category theory constructs that Cats/Cats Effect natively supports.

### Products (×)

```scala
import cats.syntax.all._
import cats.effect.{Async, IO}

// Tuples are products - Cats Effect works with them directly
def processProduct[F[_]: Async](a: Int, b: String): F[(Int, String)] =
  (computeA[F](a), computeB[F](b)).tupled  // Applicative.tupled

// The &&& operator produces products
val pipeline: TypedCoCell[Input, (OutputA, OutputB)] =
  cellA &&& cellB  // Fan-out to product

// mapN for combining products
(cellA.run[IO](input), cellB.run[IO](input)).mapN { (resultA, resultB) =>
  // combine results
}
```

### Coproducts (+)

```scala
import cats.data.EitherT

// Either is the canonical coproduct - full Cats Effect support
def handleCoproduct[F[_]: Async](input: Either[ErrorA, ValueB]): F[Result] =
  input.fold(
    err => handleError[F](err),
    value => processValue[F](value)
  )

// EitherT for effectful coproducts
val computation: EitherT[IO, CellError, Result] =
  EitherT(myCell.run[IO](input))

// Chain operations with EitherT
val pipeline: EitherT[IO, CellError, FinalResult] = for {
  a <- EitherT(cellA.run[IO](input))
  b <- EitherT(cellB.run[IO](a))
  c <- EitherT(cellC.run[IO](b))
} yield c
```

### Type Class Support

| Construct | Scala Type | Cats Support |
|-----------|------------|--------------|
| Product (×) | `(A, B)`, `Tuple2` | `Applicative.product`, `tupled`, `mapN` |
| Coproduct (+) | `Either[A, B]` | `MonadError`, `EitherT`, `Bifunctor` |
| Exponential (→) | `A => F[B]` | `Kleisli`, `FunctionK` |
| Identity | `A` | `Id`, `Applicative.pure` |

### Kleisli Integration

The `CoCellArrow` converts directly to Kleisli for integration with Cats Effect pipelines:

```scala
import cats.data.Kleisli

val arrow: CoCellArrow[IO, Input, Output] = CoCellArrow("process", processInput)

// Convert to Kleisli for composition with other Cats Effect code
val kleisli: Kleisli[IO, Input, Either[CellError, Output]] = arrow.toKleisli

// Compose Kleisli arrows
val combined = kleisli.andThen(otherKleisli)
```

### Parallel Execution

```scala
import cats.syntax.parallel._

// Run cells in parallel using Cats Effect
val parallelResults: IO[(Either[CellError, A], Either[CellError, B])] =
  (cellA.run[IO](inputA), cellB.run[IO](inputB)).parTupled

// The *** operator leverages this internally
val parallelPipeline: TypedCoCell[(In1, In2), (Out1, Out2)] =
  cellA *** cellB  // Parallel product
```

---

## TypedCoCell (Typed)

The `TypedCoCell[In, Out]` trait provides compile-time type safety for state transformations.

### Core Trait

```scala
trait TypedCoCell[In, Out] { self =>
  def name: String
  def run[F[_]: Async](input: In): F[Either[CellError, Out]]

  // Sequential composition
  def >>>[Out2](other: TypedCoCell[Out, Out2]): TypedCoCell[In, Out2]

  // Parallel composition
  def ***[In2, Out2](other: TypedCoCell[In2, Out2]): TypedCoCell[(In, In2), (Out, Out2)]

  // Fan-out composition
  def &&&[Out2](other: TypedCoCell[In, Out2]): TypedCoCell[In, (Out, Out2)]

  // Functor operations
  def map[Out2](f: Out => Out2): TypedCoCell[In, Out2]
  def contramap[In2](f: In2 => In): TypedCoCell[In2, Out]
  def dimap[In2, Out2](f: In2 => In)(g: Out => Out2): TypedCoCell[In2, Out2]
}
```

### Companion Object Constructors

```scala
// Lift a pure function
TypedCoCell.lift[A, B](name: String)(f: A => B): TypedCoCell[A, B]

// Lift an effectful function
TypedCoCell.liftF[A, B](name: String)(f: A => IO[B]): TypedCoCell[A, B]

// Categorical primitives
TypedCoCell.id[A]: TypedCoCell[A, A]
TypedCoCell.fst[A, B]: TypedCoCell[(A, B), A]
TypedCoCell.snd[A, B]: TypedCoCell[(A, B), B]
TypedCoCell.diagonal[A]: TypedCoCell[A, (A, A)]
```

---

## TypedCoCellWithCell (Bridge Trait)

Bridges the high-level TypedCoCell abstraction with the low-level kernel Cell infrastructure:

```scala
trait TypedCoCellWithCell[In <: Ω, Out <: Ω] extends TypedCoCell[In, Out] {
  def cell: Cell[IO, StackF, In, Out, Either[CellError, Out]]
  def toCell: Cell[IO, StackF, Ω, Ω, Either[CellError, Ω]] = cell.asInstanceOf[...]
}
```

---

## SafeBlockConsensusCell

Extends the safe hylo pattern to block consensus operations:

```scala
sealed trait ZKWasmProcessingState extends Ω {
  def depth: Int
}

case class WasmInput(input: L0OutputState) extends ZKWasmProcessingState {
  def depth: Int = 2  // Initial state
}

case class WasmExecuted(input: L0OutputState, result: Int, proofPath: Option[Path])
    extends ZKWasmProcessingState {
  def depth: Int = 1  // After WASM execution
}

case class WasmCompleted(result: Either[CellError, StateChannelState])
    extends ZKWasmProcessingState {
  def depth: Int = 0  // Terminal state
}
```

---

## Arrow Implementations

The `CoCellArrow[F[_], A, B]` provides a reusable arrow abstraction:

```scala
final case class CoCellArrow[F[_], A, B](
  name: String,
  run: A => F[Either[CellError, B]]
) {
  def apply(a: A)(using F: Async[F]): F[Either[CellError, B]]
  def map[C](f: B => C)(using F: Functor[F]): CoCellArrow[F, A, C]
  def toKleisli(using F: Functor[F]): Kleisli[F, A, Either[CellError, B]]
}
```

---

## Pipeline Composition

### Sequential Composition (`>>>`)

```scala
val pipeline: TypedCoCell[A, D] = cell1 >>> cell2 >>> cell3
// A → cell1 → B → cell2 → C → cell3 → D
```

### Parallel Composition (`***`)

```scala
val parallel: TypedCoCell[(A, C), (B, D)] = cell1 *** cell2
// (A, C) → (cell1(A), cell2(C)) → (B, D)
```

### Fan-out Composition (`&&&`)

```scala
val fanOut: TypedCoCell[A, (B, C)] = cell1 &&& cell2
// A → (cell1(A), cell2(A)) → (B, C)
```

---

## Topos Type System

The CoCell system is built on topos-theoretic foundations:

### Products (`×`)

```scala
type ×[A, B] = (A, B)

def assocL[A, B, C]: ((A, B), C) => (A, (B, C))
def assocR[A, B, C]: (A, (B, C)) => ((A, B), C)
def swap[A, B]: (A, B) => (B, A)
```

### Coproducts (`⊕`)

```scala
sealed trait ⊕[+A, +B]
case class Inl[A](value: A) extends ⊕[A, Nothing]
case class Inr[B](value: B) extends ⊕[Nothing, B]

def fold[A, B, C](left: A => C, right: B => C): ⊕[A, B] => C
```

### Conditional Routing

```scala
TypedCoCell.conditional[A, C](
  predicate: A => Boolean,
  ifTrue: TypedCoCell[A, C],
  ifFalse: TypedCoCell[A, C]
): TypedCoCell[A, C]
```

---

## Dynamic Endpoints

TypedCoCell nodes can dynamically add HTTP endpoints at runtime:

```scala
val l0CustomRoutes: HttpApi[IO] => AdditionalRoutes[IO] = { api =>
  new AdditionalRoutes[IO] with Http4sDsl[IO] {
    private val l0InfoRoute: HttpRoutes[IO] = HttpRoutes.of[IO] {
      case GET -> Root / "l0" / "info" =>
        Ok(Json.obj(
          "layer" -> Json.fromString("L0"),
          "nodeId" -> Json.fromString(api.selfId.toString)
        ))
    }

    override val publicRoutes: HttpRoutes[IO] = l0InfoRoute
    override val p2pRoutes: HttpRoutes[IO] = HttpRoutes.empty[IO]
  }
}

val l0Node = BabelApp.l0Node(
  cell,
  Some(config),
  customRoutes = List(l0CustomRoutes)
)
```

---

# Safe Hylo and ZK-WASM Execution Guide

A comprehensive guide to safe hylomorphism patterns and ZK-WASM execution with the Reality SDK.

## Overview

The Reality SDK provides three complementary patterns for building safe, verifiable blockchain computations:

| Pattern | Purpose | Key Feature |
|---------|---------|-------------|
| **Safe Hylo** | Guaranteed termination | WellFounded measure strictly decreases |
| **TypedCoCellWithCell** | Type-safe Cell infrastructure | Combines TypedCoCell with kernel Cell |
| **ZK-WASM** | Cryptographic proofs | ZK-STARK proofs for WASM execution |

---

## Safe Hylo Fundamentals

A **hylomorphism** (hylo) is the composition of an unfold (anamorphism) followed by a fold (catamorphism):

```scala
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

case class Processing[A](value: A) extends ProcessingState[A] {
  def depth: Int = 1
}

case class Completed[A](result: Either[CellError, A]) extends ProcessingState[A] {
  def depth: Int = 0
}
```

---

## SafeStreamCell

Creates TypedCoCellWithCell instances using the safe hylo pattern:

```scala
object SafeStreamCell {
  def safeCoalgebra[F[_]: Async, A](input: A): F[ProcessingState[A]] =
    Processing(input).pure[F]

  def safeAlgebra[F[_]: Async, A](state: ProcessingState[A]): F[Either[CellError, A]] =
    state match {
      case Processing(value) => value.asRight[CellError].pure[F]
      case Completed(result) => result.pure[F]
    }

  def apply[In <: Ω, Out <: Ω](
    cellName: String,
    transform: In => Out
  ): TypedCoCellWithCellAndProof[In, Out]
}
```

### Usage

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

## ZK-WASM Integration

### RealZKWasmExecutor

```scala
class RealZKWasmExecutor[F[_]: Async] {
  def generateProof(
    name: String,
    publicInputs: List[(String, Any)],
    privateInputs: List[(String, Any)]
  ): F[Either[ZKProofError, Path]]

  def verifyProof(
    proofPath: Path,
    publicInputs: List[(String, Any)]
  ): F[Either[ZKProofError, Boolean]]
}
```

### ZKWasmExecutionRoutes

```scala
class ZKWasmExecutionRoutes[F[_]: Async](nodeApi: HttpApi[F])
    extends AdditionalRoutes[F] {

  // POST /zk-wasm/execute - Execute WASM with ZK proof
  // POST /zk-wasm/verify - Verify a ZK proof
  // GET /zk-wasm/status - Executor status
}
```

### Testing Endpoints

```bash
# Check status
curl http://localhost:9010/zk-wasm/status

# Execute WASM with ZK proof
curl -X POST -H "Content-Type: application/json" \
  -d '{"function": "add", "a": 42, "b": 58}' \
  http://localhost:9010/zk-wasm/execute

# Verify proof
curl -X POST -H "Content-Type: application/json" \
  -d '{"proofPath": "/tmp/proof.proof", "a": 42, "b": 58, "expectedResult": 100}' \
  http://localhost:9010/zk-wasm/verify
```

---

### Turing Completeness with Finite Circuits

ZK-WASM can prove Turing-complete code execution despite using finite circuits. Here's how the apparent paradox is resolved:

#### Key Concepts

| Concept | What It Means |
|---------|---------------|
| **Language Turing-completeness** | WASM can express any computable function |
| **Circuit finiteness** | Each proof covers a bounded number of steps |
| **Practical unboundedness** | Proof composition chains proofs for longer computation |

#### How It Works

**1. Universal Circuit Structure**

The ZK circuit is designed to prove *any* WASM instruction sequence up to N steps. The circuit structure doesn't change based on the program - it's a universal WASM interpreter in arithmetic circuit form.

**2. Step-Bounded Proofs**

A single proof covers execution up to a fixed cycle limit (e.g., 2²⁰ steps). If your computation finishes within that bound, you get a valid proof.

**3. Continuation/Folding for Unbounded Computation**

For computations exceeding the step limit, IVC (Incrementally Verifiable Computation) schemes allow chaining:

```
Proof₁: Steps 0 → N (produces state S₁)
Proof₂: Steps N → 2N starting from S₁ (produces state S₂)
Proof₃: Steps 2N → 3N starting from S₂ ...
```

Each proof is constant-size, and they can be folded/aggregated into a single succinct proof.

**4. The Reality SDK Approach**

The SDK uses `SafeHylo` and categorical recursion schemes that are *structurally terminating* - they guarantee completion within bounds by construction, avoiding the need for unbounded proof chains in most cases.

#### Practical Implications

```scala
// This is provable - bounded recursion via catamorphism
val safeSum = SafeHylo.cata[List[Int], Int](
  _ => 0,                    // nil case
  (head, acc) => head + acc  // cons case
)

// This would require continuation proofs if it exceeds step limit
def unboundedLoop(n: BigInt): BigInt =
  if n == 1 then 1
  else unboundedLoop(if n % 2 == 0 then n/2 else 3*n + 1)
```

ZK-WASM hosts Turing-complete code, proves finite execution segments, and achieves unbounded computation through proof composition when needed.

---

# Halt-Free Computation in the Babel

The Babel provides a **stratified approach to computation** where the halt-free property applies to specific subsets of the language.

## The Computational Hierarchy

| Layer | Computational Class | Termination |
|-------|---------------------|-------------|
| Basic combinators | Sub-Turing (dataflow) | **Guaranteed** |
| Bounded hylomorphisms | Primitive recursive | **Guaranteed** |
| Unbounded hylomorphisms | Turing complete | **Not guaranteed** |
| zkSNARK-compilable | Finite circuits | **Guaranteed** |

---

## Why Halt-Free?

The DSL combinators **preserve termination structurally**:

| Combinator | Termination Property |
|------------|---------------------|
| `f >>> g` | If f and g halt, composition halts |
| `f &&& g` | If f and g halt, fan-out halts |
| `f *** g` | If f and g halt, product halts |
| `fanIn(f, g)` | If both branches halt, case analysis halts |
| `conditional(p, f, g)` | If predicate and branches halt, routing halts |

---

## Categorical Semantics

In category theory, a morphism `A → B` **is** a total function by definition:

```
CoCell programs ≅ Morphisms in a symmetric monoidal category with coproducts
```

| Categorical Concept | CoCell Equivalent | Property |
|---------------------|-------------------|----------|
| Morphism `A → B` | `TypedCoCell[A, B]` | Total by definition |
| Composition `g ∘ f` | `f >>> g` | Preserves totality |
| Product `A × B` | `(A, B)` tuple | Finite structure |
| Coproduct `A + B` | `A ⊕ B` | Finite branching |

---

## The `lift` Boundary

```scala
TypedCoCell.lift("transform")(f: A => B)
```

The `lift` combinator is the boundary between the halt-free DSL and the host language. The DSL *assumes* lifted functions are total.

### Safe Operations for `lift`

| Category | Examples | Why Safe |
|----------|----------|----------|
| Pure arithmetic | `+`, `-`, `*`, `/` | Primitive operations terminate |
| Pattern matching | `case class` destructuring | Finite case analysis |
| Primitive recursive | `fold`, `map` on finite structures | Bounded iteration |

---

## Hylomorphisms and Turing Completeness

The Babel includes hylomorphisms which provide **full Turing completeness**:

```scala
def hylo[F[_]: Functor, A, B](
  coalgebra: A => F[A],
  algebra: F[B] => B
): A => B
```

### When Hylomorphisms Terminate

| Coalgebra Type | Termination |
|----------------|-------------|
| Well-founded (has base case) | **Terminates** |
| Productive (infinite but lazy) | May not terminate eagerly |
| Divergent (no base case) | **Does not terminate** |

### Bounded Hylomorphisms

For zkSNARK compilation:

```scala
val safe: TypedCoCell[Input, Output] =
  hylo(algebra)(coalgebra).bounded(maxDepth = 1000)
```

---

## The zkSNARK Connection

```
CoCell pipeline → Arithmetic circuit → zkSNARK proof
     ↓                   ↓                  ↓
 Halt-free DSL      Finite gates       Witness exists
```

When a CoCell pipeline produces a zkSNARK proof, the proof's existence **is** the termination certificate.

---

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Full Babel (with unbounded hylomorphisms)                  │
│  Turing complete • May not terminate • Off-chain use        │
├─────────────────────────────────────────────────────────────┤
│  Bounded Hylomorphisms                                      │
│  Primitive recursive • Always terminates • Complex algos    │
├─────────────────────────────────────────────────────────────┤
│  Basic Combinators (>>>, &&&, ***, conditional)             │
│  Sub-Turing • Always terminates • State transitions         │
├─────────────────────────────────────────────────────────────┤
│  zkSNARK-Compilable Subset                                  │
│  Finite circuits • Always terminates • Consensus-critical   │
└─────────────────────────────────────────────────────────────┘
```

---

# Stream Topology Examples

A comprehensive guide to TypedCoCell operators and topology patterns for building data processing pipelines.

## Operator Overview

| Operator | Signature | Category Theory | Description |
|----------|-----------|-----------------|-------------|
| `>>>` | `A→B >>> B→C = A→C` | Composition | Sequential pipeline |
| `&&&` | `A→B &&& A→C = A→(B,C)` | Product morphism | Fan-out to multiple outputs |
| `***` | `A→B *** C→D = (A,C)→(B,D)` | Bifunctor | Parallel independent processing |
| `dimap` | `(X→A, B→Y) => (A→B) => X→Y` | Profunctor | Transform both input and output |
| `wrap` | `(A→B) => (Either[E,B]→Either[E,C]) => A→C` | Result transformer | Transform cell results |

---

## Domain Types

```scala
case class RawSensorReading(deviceId: String, value: Double, timestamp: Long) extends Ω
case class CalibratedReading(deviceId: String, value: Double, unit: String, calibrationFactor: Double) extends Ω
case class ValidatedReading(reading: CalibratedReading, qualityScore: Double, isValid: Boolean) extends Ω
case class ReadingStats(count: Int, sum: Double, min: Double, max: Double) extends Ω
case class Alert(severity: String, message: String, timestamp: Long) extends Ω
```

---

## Basic Cells

```scala
val calibrate: TypedCoCell[RawSensorReading, CalibratedReading] =
  TypedCoCell.lift("calibrate") { raw =>
    CalibratedReading(raw.deviceId, raw.value * 1.05, "celsius", 1.05)
  }

val validate: TypedCoCell[CalibratedReading, ValidatedReading] =
  TypedCoCell.lift("validate") { cal =>
    val quality = if cal.value >= 0 && cal.value <= 100 then 1.0 else 0.5
    ValidatedReading(cal, quality, quality > 0.7)
  }

val computeStats: TypedCoCell[CalibratedReading, ReadingStats] =
  TypedCoCell.lift("computeStats") { cal =>
    ReadingStats(1, cal.value, cal.value, cal.value)
  }
```

---

## Combined Topology Patterns

### Diamond Pattern

```scala
val diamondPattern: TypedCoCell[RawSensorReading, Alert] = {
  val merge: TypedCoCell[(ValidatedReading, ReadingStats), Alert] =
    TypedCoCell.lift("merge") { case (v, s) =>
      Alert("INFO", s"Valid=${v.isValid}, Sum=${s.sum}", System.currentTimeMillis())
    }

  calibrate >>> (validate &&& computeStats) >>> merge
}
```

```
           ┌─── validate ───┐
calibrate ─┤                ├─── merge ─── Alert
           └─── stats ──────┘
```

### Scatter-Gather Pattern

```scala
val scatterGather: TypedCoCell[CalibratedReading, ReadingStats] = {
  val path1 = TypedCoCell.lift[CalibratedReading, Double]("path1")(_.value * 2)
  val path2 = TypedCoCell.lift[CalibratedReading, Double]("path2")(_.value + 10)
  val gather = TypedCoCell.lift[(Double, Double), ReadingStats]("gather") { case (v1, v2) =>
    ReadingStats(2, v1 + v2, Math.min(v1, v2), Math.max(v1, v2))
  }

  (path1 &&& path2) >>> gather
}
```

### Fork-Join Pattern

```scala
val forkJoin: TypedCoCell[(RawSensorReading, RawSensorReading), ReadingStats] = {
  val join = TypedCoCell.lift[(CalibratedReading, CalibratedReading), ReadingStats]("join") {
    case (c1, c2) => ReadingStats(2, c1.value + c2.value, Math.min(c1.value, c2.value), Math.max(c1.value, c2.value))
  }
  (calibrate *** calibrate) >>> join
}
```

---

## Running Examples

```bash
sbt "combined/runMain org.reality.combined.examples.topologies.StreamTopologyExampleRunner"
```

---

# StreamTopologyTypes: Safe Stream Cells with Termination Guarantees

## SafeStreamCell

Creates TypedCoCells with **guaranteed termination** via WellFounded recursion:

```scala
object SafeStreamCell {
  def fromPure[A <: Ω, B <: Ω](name: String, transform: A => B): TypedCoCell[A, B]
  def fromPureWithProof[A <: Ω, B <: Ω](name: String, transform: A => B): TypedCoCell[A, (B, StreamTerminationProof)]
}
```

### Termination Guarantees

| Property | Guarantee |
|----------|-----------|
| Termination | Every computation finishes in bounded steps |
| Proofs | Evidence of termination for consensus verification |
| Type Safety | No `asInstanceOf` casts needed |

### Processing State

```scala
sealed trait StreamProcessingState[+A] extends Ω {
  def depth: Int
}

case class StreamInput[A](input: A) extends StreamProcessingState[A] {
  def depth: Int = 1
}

case class StreamCompleted[A](result: Either[CellError, A]) extends StreamProcessingState[A] {
  def depth: Int = 0
}
```

---

## Safe Base Cells

```scala
val safeCalibrate = SafeStreamCell("calibrate", (raw: RawSensorReading) =>
  CalibratedReading(raw.deviceId, raw.value * 1.05, "celsius", 1.05)
)

val safeValidate = SafeStreamCell("validate", (cal: CalibratedReading) => {
  val quality = if cal.value >= -50 && cal.value <= 150 then 1.0 else 0.5
  ValidatedReading(cal, quality, quality > 0.7)
})
```

---

## Safe Composed Topologies

```scala
val safeFullPipeline: TypedCoCell[RawSensorReading, ((ValidatedReading, ReadingStats), Alert)] =
  safeCalibrate >>> ((safeValidate &&& safeComputeStats) &&& safeGenerateAlert)

val safeDiamondPattern: TypedCoCell[RawSensorReading, Alert] = {
  val safeMerge = SafeStreamCell.forPair("merge", (v: ValidatedReading, s: ReadingStats) =>
    Alert("INFO", s"Valid=${v.isValid}, Sum=${s.sum}", System.currentTimeMillis())
  )
  safeCalibrate >>> (safeValidate &&& safeComputeStats) >>> tupleToΩPair >>> safeMerge
}
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
| Wrapped | `wrap` | Error handling, validation |
| Profunctor | `dimap` | Transform both ends of a cell |

These patterns enable building complex data processing topologies with blockchain-grade reliability, type safety, and verifiable termination.
