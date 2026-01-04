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

1. Add the `reality-combined` JAR to your project's `lib/` directory
2. Configure your `build.sbt` with required dependencies
3. Choose your approach: Untyped (UntypedBabelApp) or Typed (BabelApp)

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
| [`TypedCocellExample.scala`](#typedcocellexamplescala) | State types and base typed cells | Typed |
| [`TypedCocellBabelExample.scala`](#typedcocellbabelexamplescala) | Single-pipeline BabelApp example | Typed |
| [`ComposeMultipleTypedCocellBabelExample.scala`](#composemultipletypedcocellbabelexamplescala) | Multi-pipeline BabelApp example | Typed |
| [`HigherOrderTypedCoCellBabelExample.scala`](#higherordertypedcocellbabelexamplescala) | 3-layer architecture with L2 | Typed |
| [`ConfiguredBabelExample.scala`](#configuredbabelexamplescala) | Programmatic configuration helpers | Typed |
| [`CombinedMonoidExample.scala`](#combinedmonoidexamplescala) | Untyped CoCell definitions | Untyped |
| [`ComposeExamples.scala`](#composeexamplesscala) | Simple untyped composition | Untyped |
| [`ComposeMultipleExample.scala`](#composemultipleexamplescala) | Multi-node untyped composition | Untyped |

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

**Key features:**
- `CombinedL0` / `CombinedL0_2` - L0 layer implementations
- `CombinedStateChannel` / `CombinedStateChannel_2` - L1 layer implementations
- Static configuration support via `withConfig()`
- CLI argument parsing via `argsToStartUp()`

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

### ComposeMultipleExample.scala

Four-node composition (2 L0/L1 pairs) with static configuration:

```scala
open class ComposeMultipleExample extends UntypedBabelApp {
  // First L0/L1 pair (ports 9000-9012)
  val combinedL0 = CombinedL0.withConfig(StaticConfigs.TwoNodePairs.l0Config)
  val combinedStateChannel = CombinedStateChannel.withConfig(StaticConfigs.TwoNodePairs.l1Config)

  // Second L0/L1 pair (ports 9020-9032)
  val combinedL0_2 = CombinedL0_2.withConfig(StaticConfigs.TwoNodePairs.l0Config2)
  val combinedStateChannel_2 = CombinedStateChannel_2.withConfig(StaticConfigs.TwoNodePairs.l1Config2)

  // Compose all 4 CoCells
  val composedX2: CoCell = combinedL0 >>> combinedStateChannel >>> combinedL0_2 >>> combinedStateChannel_2

  val cellProgram = (args: List[String]) =>
    for {
      cellInstances <- composedX2.setup(args)
    } yield cellInstances :: Nil
}
```

### Monoid Composition (`++` / `<~>`)

The monoid operator creates a list of independent CoCells:

```scala
open class CombinedMonoidExample extends UntypedBabelApp {
  val combinedL0: CombinedL0 = CombinedL0()
  val combinedStateChannel: CombinedStateChannel = CombinedStateChannel()

  // Monoid composition: creates List[CoCell] - each runs independently
  val mergedCells: Seq[CoCell] = combinedL0 ++ combinedStateChannel

  val cellProgram = (args: List[String]) =>
    for {
      cellInstances <- mergedCells.map(_.setup(args)).sequence
    } yield cellInstances
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

#### Typed State Channel Cell

```scala
object CombinedStateChannelTyped {
  def apply(staticConfig: Option[L1CoCellConfig] = None): TypedCoCell[L0OutputState, StateChannelState] =
    new TypedCoCell[L0OutputState, StateChannelState] {
      override val name: String = staticConfig.map(_.name).getOrElse("CombinedStateChannelTyped")

      override def run[F[_]: Async](input: L0OutputState): F[Either[CellError, StateChannelState]] =
        Async[F].pure {
          Right(StateChannelState(
            finalData = input.processedData,
            stateChannelHash = Some(s"sc-${input.l0Hash.getOrElse("unknown")}"),
            consensusReached = true,
            metadata = input.metadata + ("layer" -> "StateChannel")
          ))
        }
    }

  def withConfig(cfg: L1CoCellConfig): TypedCoCell[L0OutputState, StateChannelState] =
    apply(Some(cfg))
}
```

#### Default Pipeline

```scala
object TypedCocellExample {
  def defaultPipeline: TypedCoCell[L0InputState, StateChannelState] =
    CombinedL0Typed() >>> CombinedStateChannelTyped()
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

  /** Execute with tracing */
  def executeTypedPipelineWithTrace(input: L0InputState): IO[Either[CellError, StateChannelState]] =
    for {
      _ <- IO.println(s"[BabelTypedCocellExample] Input: $input")
      result <- execute(input)
      _ <- result.fold(
        error => IO.println(s"Error: ${error.reason}"),
        state => IO.println(s"Success: $state")
      )
    } yield result
}
```

### ComposeMultipleTypedCocellBabelExample.scala

Multi-pipeline BabelApp application (2 L0/L1 pairs):

```scala
open class ComposeMultipleTypedCocellBabelExample extends BabelApp[L0InputState, StateChannelState] {

  // First L0/L1 pair (genesis)
  val l0Node: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(
      CombinedL0Typed.withConfig(StaticConfigs.SingleNodeGenesis.l0Config),
      Some(StaticConfigs.SingleNodeGenesis.l0Config)
    )

  val stateChannelNode: TypedCoCellNode[L0OutputState, StateChannelState] =
    BabelApp.l1Node(
      CombinedStateChannelTyped.withConfig(StaticConfigs.SingleNodeGenesis.l1Config),
      L1,
      Some(StaticConfigs.SingleNodeGenesis.l1Config)
    )

  // Second L0/L1 pair (validator)
  val l0Node2: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(
      CombinedL0Typed.withConfig(StaticConfigs.SingleNodeGenesis.l0Config2),
      Some(StaticConfigs.SingleNodeGenesis.l0Config2)
    )

  val stateChannelNode2: TypedCoCellNode[L0OutputState, StateChannelState] =
    BabelApp.l1Node(
      CombinedStateChannelTyped.withConfig(StaticConfigs.SingleNodeGenesis.l1Config2),
      L1,
      Some(StaticConfigs.SingleNodeGenesis.l1Config2)
    )

  // Two pipelines
  val pipeline1 = CoCellPipeline.from(l0Node) >>> stateChannelNode
  val pipeline2 = CoCellPipeline.from(l0Node2) >>> stateChannelNode2

  val pipelines: List[CoCellPipeline[L0InputState, StateChannelState]] =
    List(pipeline1, pipeline2)
}
```

**Network topology:**
- First pair (genesis): L0=9000-9002, L1=9010-9012
- Second pair (validator): L0=9020-9022, L1=9030-9032
- Both L1 nodes point to first L0 for peer discovery

### HigherOrderTypedCoCellBabelExample.scala

Three-layer architecture with L2 data ingestion layer:

```scala
/**
 * Architecture:
 *   L2 (accepts Ω) ──► L0OutputState ──► L1 (StateChannel)
 *         │                                     │
 *         │         L0 (base layer)             │
 *         │      L0InputState → L0OutputState   │
 *         │              │                      │
 *         └──────────────┴──────────────────────┘
 */
open class HigherOrderTypedCoCellBabelExample extends BabelApp[L0InputState, Ω] {

  // L0 layer - Base consensus
  val l0Node: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(CombinedL0Typed.withConfig(...), Some(...))

  // L1 layer - State Channel
  val l1Node: TypedCoCellNode[L0OutputState, StateChannelState] =
    BabelApp.l1Node(CombinedStateChannelTyped.withConfig(...), L1, Some(...))

  // L2 layer - Accepts any Ω, outputs L0OutputState
  val l2Node: TypedCoCellNode[Ω, L0OutputState] =
    BabelApp.node(L2TypedCoCell())

  // Primary pipeline: L0 → L1 → Ω
  val primaryPipeline: CoCellPipeline[L0InputState, Ω] =
    CoCellPipeline.from(l0Node) >>> l1Node >>> BabelApp.node(
      TypedCoCell.lift[StateChannelState, Ω]("StateChannelToOmega")(identity)
    )

  // L2 → L1 pipeline: Ω → L0OutputState → StateChannelState → Ω
  val l2ToL1Pipeline: CoCellPipeline[Ω, Ω] =
    CoCellPipeline.from(l2Node) >>> l1Node >>> BabelApp.node(...)

  val pipelines = List(primaryPipeline)

  // Execute L2 pipeline with arbitrary Ω input
  def executeL2Pipeline(input: Ω): IO[Either[CellError, Ω]] =
    l2ToL1Pipeline.run[IO](input)

  // Start periodic L2 data sender
  def startPeriodicL2Sender(
    interval: FiniteDuration,
    dataGenerator: () => Ω
  ): IO[FiberIO[Unit]] = ...
}
```

**L2 Cell with Algebra/CoAlgebra:**

```scala
class L2Cell[F[_]: Async](
  data: Ω,
  l1InputQueue: Queue[F, L0OutputState]
) extends Cell[F, StackF, L2CoalgebraCommand, Ω, Either[CellError, Ω]](
  data,
  scheme.hyloM(
    // Algebra: Process commands and send to L1
    AlgebraM[F, StackF, Either[CellError, Ω]] {
      case Done(Right(L2AlgebraCommand.SendToL1(outputState))) =>
        l1InputQueue.offer(outputState) >> ...
      // ...
    },
    // CoAlgebra: Accept any Ω and produce processing command
    CoalgebraM[F, StackF, Ω] {
      case input: L2Input => processInput(input)
      case other => processInput(L2Input(other))  // Accept ANY Ω
    }
  ),
  // Input converter
  { case input => L2CoalgebraCommand.ProcessInput(L2Input(input)) }
)
```

**Key features:**
- L2 accepts arbitrary `Ω` input
- L2 outputs `L0OutputState` (compatible with L1 input type)
- Cell CoAlgebra accepts every Ω
- Cell Algebra sends transactions to L1 via queue
- Periodic data sender for continuous L2→L1 injection

**Adjunction Structure (Cohomological Duality):**

The two pipelines form an adjunction representing dual cohomological perspectives:

```scala
// Left Adjoint (Cochain direction): L0 → L1 → Ω
// State derivation - consensus state lifts to applications
val leftAdjoint = CoCellPipeline.from(l0Node) >>> l1Node >>> ...

// Right Adjoint (Chain direction): Ω → L1 → Ω
// Transaction submission - user data descends to consensus
val rightAdjoint = CoCellPipeline.from(l2Node) >>> l1Node >>> ...
```

| Direction | Operator | Semantics | Cohomological View |
|-----------|----------|-----------|-------------------|
| L0 → L1 → L2 | Coboundary δ | State derivation | Cochain complex C^n |
| L2 → L1 → L0 | Boundary ∂ | Transaction submission | Chain complex C_n |

The adjunction laws encode blockchain invariants:
- **Unit (η)**: Submit then derive = transaction inclusion proof
- **Counit (ε)**: Derive then submit = idempotent re-submission

**Periodic Right Adjoint Sender:**

```scala
// Start periodic data injection (lazy initialization)
val fiber: IO[FiberIO[Unit]] = periodicL2SenderFiber

// Or with custom parameters
def startPeriodicRightAdjointSender(
  interval: FiniteDuration,
  dataGenerator: () => Ω
): IO[FiberIO[Unit]]

// Cancel to stop
fiber.flatMap(_.cancel)
```

---

### ConfiguredBabelExample.scala

Programmatic configuration for deployments:

```scala
object ConfiguredBabelExample {

  /** Create single-pipeline BabelApp instance */
  def withConfig(
    l0Config: L0CoCellConfig,
    l1Config: L1CoCellConfig
  ): BabelApp[L0InputState, StateChannelState] = new BabelApp[L0InputState, StateChannelState] {

    val l0Node = BabelApp.l0Node(CombinedL0Typed.withConfig(l0Config), Some(l0Config))
    val stateChannelNode = BabelApp.l1Node(
      CombinedStateChannelTyped.withConfig(l1Config),
      L1,
      Some(l1Config)
    )

    val pipelines = List(CoCellPipeline.from(l0Node) >>> stateChannelNode)
  }

  /** Create two-pipeline BabelApp instance */
  def withTwoPipelines(
    l0Config: L0CoCellConfig,
    l1Config: L1CoCellConfig,
    l0Config2: L0CoCellConfig,
    l1Config2: L1CoCellConfig
  ): BabelApp[L0InputState, StateChannelState] = // ...

  /** Create pipeline from configuration */
  def createPipeline(
    l0Config: L0CoCellConfig,
    l1Config: L1CoCellConfig
  ): CoCellPipeline[L0InputState, StateChannelState] = {
    val l0Node = BabelApp.l0Node(CombinedL0Typed.withConfig(l0Config), Some(l0Config))
    val scNode = BabelApp.l1Node(CombinedStateChannelTyped.withConfig(l1Config), L1, Some(l1Config))
    CoCellPipeline.from(l0Node) >>> scNode
  }

  /** Create multiple pipelines from config list */
  def createPipelines(
    configs: List[(L0CoCellConfig, L1CoCellConfig)]
  ): List[CoCellPipeline[L0InputState, StateChannelState]] =
    configs.map { case (l0, l1) => createPipeline(l0, l1) }
}
```

**Use cases:**
- Docker/Kubernetes deployments
- Programmatic cluster creation
- Test environments
- Multi-node clusters with pre-defined topology

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

// Two-node pairs
StaticConfigs.TwoNodePairs.l0Config
StaticConfigs.TwoNodePairs.l1Config
StaticConfigs.TwoNodePairs.l0Config2
StaticConfigs.TwoNodePairs.l1Config2

// Validator configurations
StaticConfigs.Validators.validator1L0Config
StaticConfigs.Validators.validator1L1Config
```

### L0CoCellConfig

```scala
case class L0CoCellConfig(
  name: String,
  keyConfig: KeyConfig,
  environment: AppEnvironment,
  httpConfig: HttpConfig,
  dbConfig: L0DBConfig,
  snapshotConfig: SnapshotConfig,
  l0Peer: L0Peer,
  seedlistPath: Option[Path] = None,
  collateralAmount: Option[Amount] = Some(Amount.empty),
  genesisPath: Option[Path] = None,
  isGenesis: Boolean = false
) extends CoCellConfig
```

### L1CoCellConfig

```scala
case class L1CoCellConfig(
  name: String,
  keyConfig: KeyConfig,
  environment: AppEnvironment,
  httpConfig: HttpConfig,
  dbConfig: L1DBConfig,
  snapshotConfig: SnapshotConfig,
  l0Peer: L0Peer,
  aciDBPath: Path,
  seedlistPath: Option[Path] = None,
  collateralAmount: Option[Amount] = Some(Amount.empty),
  isInitialValidator: Boolean = false
) extends CoCellConfig
```

### CoCellConfigGenerator

Dynamic port allocation and configuration generation:

```scala
import org.reality.combined.CoCellConfigGenerator

// Auto-allocate ports
val (publicPort, p2pPort, cliPort) = CoCellConfigGenerator.allocatePorts()

// Reset port counter
CoCellConfigGenerator.resetPortCounter(9000)

// Generate configurations
val l0Config = CoCellConfigGenerator.generateL0GenesisConfig(...)
val l1Config = CoCellConfigGenerator.generateL1InitialValidatorConfig(...)
```

---

## State Types

### Type Flow

```
L0InputState  ──►  L0OutputState  ──►  StateChannelState
    │                   │                    │
    │ L0 Processing     │ L1 Processing      │ Final State
    └───────────────────┴────────────────────┘
```

### Extending State Types

Create your own state types by extending `Ω`:

```scala
// Custom input state
case class MyInputState(
  userId: String,
  action: String,
  payload: Json,
  metadata: Map[String, String] = Map.empty
) extends Ω

// Custom intermediate state
case class MyProcessedState(
  result: Json,
  proofHash: Option[String] = None,
  metadata: Map[String, String] = Map.empty
) extends Ω

// Custom output state
case class MyFinalState(
  output: Json,
  blockHash: String,
  consensusReached: Boolean,
  metadata: Map[String, String] = Map.empty
) extends Ω
```

---

## Composition Patterns

### Sequential (`>>>`)

Chain cells where output of one feeds into the next:

```scala
// Untyped
val pipeline: CoCell = l0 >>> l1 >>> l2

// Typed (compile-time verified!)
val pipeline: TypedCoCell[A, D] = cellAB >>> cellBC >>> cellCD
```

### Parallel (`***`)

Process tuple components concurrently:

```scala
val parallel: TypedCoCell[(A, B), (C, D)] = cellAC *** cellBD
// Input (a, b) → (cellAC(a), cellBD(b)) → (c, d)
```

### Fan-out (`&&&`)

Same input to multiple cells:

```scala
val fanOut: TypedCoCell[A, (B, C)] = cellAB &&& cellAC
// Input a → (cellAB(a), cellAC(a)) → (b, c)
```

### Monoid (`++` / `<~>`)

Independent cells running in parallel:

```scala
val independent: Seq[CoCell] = cell1 ++ cell2 ++ cell3
// Each cell runs independently, results collected
```

---

## Running Your Application

### As Main Object

```scala
// Typed approach
object MyApp extends TypedCocellBabelExample

// Untyped approach
object MyApp extends ComposeExamples
```

### With Environment Variables

```bash
# Set node type
export NODE_TYPE=genesis  # or validator1, validator2

# Run
sbt run
```

### With Docker

```yaml
# docker-compose.yml
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

### Programmatic Execution

```scala
val app = ConfiguredBabelExample.withConfig(l0Config, l1Config)

// Run the application
app.main(Array("--command", "run-genesis"))

// Or execute a pipeline directly
val result: IO[Either[CellError, StateChannelState]] =
  app.execute(L0InputState(myData))
```

---

## Quick Reference

### Choosing Your Approach

| Scenario | Recommended Approach | Example File |
|----------|---------------------|--------------|
| New project | Typed + BabelApp | `TypedCocellBabelExample.scala` |
| Existing app migration | Untyped + UntypedBabelApp | `ComposeExamples.scala` |
| Multi-node local dev | Typed + BabelApp | `ComposeMultipleTypedCocellBabelExample.scala` |
| Docker deployment | Typed + ConfiguredBabelApp | `ConfiguredBabelExample.scala` |
| Rapid prototyping | Untyped + Monoid | `CombinedMonoidExample.scala` |

### Common Imports

```scala
// Typed approach
import org.reality.combined.{BabelApp, TypedCoCellNode, StaticConfigs}
import org.reality.combined.topos.{TypedCoCell, CoCellPipeline}
import org.reality.combined.examples._

// Untyped approach
import org.reality.combined.{CoCell, UntypedBabelApp, StaticConfigs}
import org.reality.combined.CoCellOps._
import org.reality.combined.examples._
```

### Port Allocation Convention

| Layer | Node | Public | P2P | CLI |
|-------|------|--------|-----|-----|
| L0 | Genesis | 9000 | 9001 | 9002 |
| L1 | Initial | 9010 | 9011 | 9012 |
| L0 | Validator | 9020 | 9021 | 9022 |
| L1 | Validator | 9030 | 9031 | 9032 |
