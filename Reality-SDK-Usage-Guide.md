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
    // Helper functions
    def circe(artifact: String): ModuleID = "io.circe" %% s"circe-$artifact" % V.circe
    def ciris(artifact: String): ModuleID = "is.cir" %% artifact % V.ciris
    def decline(artifact: String = ""): ModuleID =
      "com.monovore" %% { if (artifact.isEmpty) "decline" else s"decline-$artifact" } % V.decline
    def doobie(artifact: String): ModuleID =
      ("org.tpolecat" %% s"doobie-$artifact" % V.doobie).exclude("org.slf4j", "slf4j-api")
    def droste(artifact: String): ModuleID = "io.higherkindness" %% s"droste-$artifact" % V.droste
    def fs2(artifact: String): ModuleID = "co.fs2" %% s"fs2-$artifact" % V.fs2
    def fs2Data(artifact: String): ModuleID = "org.gnieh" %% s"fs2-data-$artifact" % V.fs2Data
    def http4s(artifact: String): ModuleID = "org.http4s" %% s"http4s-$artifact" % V.http4s
    def bouncyCastle(artifact: String): ModuleID = "org.bouncycastle" % artifact % V.bouncyCastle
    def jawn(artifact: String): ModuleID = "org.typelevel" %% artifact % V.jawnVersion

    // Core libraries
    val cats = "org.typelevel" %% "cats-core" % V.cats
    val catsEffect = "org.typelevel" %% "cats-effect" % V.catsEffect
    val catsRetry = "com.github.cb372" %% "cats-retry" % V.catsRetry

    // JSON
    val circeCore = circe("core")
    val circeGeneric = circe("generic")
    val circeParser = circe("parser")

    // HTTP
    val http4sCore = http4s("core")
    val http4sDsl = http4s("dsl")
    val http4sServer = http4s("ember-server")
    val http4sClient = http4s("ember-client")
    val http4sCirce = http4s("circe")
    val http4sJwtAuth = "dev.profunktor" %% "http4s-jwt-auth" % V.http4sJwtAuth

    // Streaming
    val fs2Core = fs2("core")
    val fs2IO = fs2("io")
    val fs2ReactiveStreams = fs2("reactive-streams")
    val fs2DataCsv = fs2Data("csv")
    val fs2DataCsvGeneric = fs2Data("csv-generic")

    // Database
    val doobieCore = doobie("core")
    val doobieHikari = doobie("hikari")
    val flyway = "org.flywaydb" % "flyway-core" % V.flyway
    val apacheDerby = "org.apache.derby" % "derby" % V.apacheDerby

    // Recursion schemes
    val drosteCore = droste("core")

    // Optics
    val monocleCore = "dev.optics" %% "monocle-core" % V.monocle
    val monocleMacro = "dev.optics" %% "monocle-macro" % V.monocle

    // Configuration
    val cirisCore = ciris("ciris")
    val declineCore = decline()
    val declineEffect = decline("effect")
    val pureconfig = "com.github.pureconfig" %% "pureconfig-core" % V.pureconfig

    // Refinement types
    val iron = "io.github.iltotore" %% "iron" % V.iron
    val ironCats = "io.github.iltotore" %% "iron-cats" % V.iron
    val ironCirce = "io.github.iltotore" %% "iron-circe" % V.iron
    val ironDecline = "io.github.iltotore" %% "iron-decline" % V.iron

    // Type class derivation
    val kittens = "org.typelevel" %% "kittens" % V.kittens
    val izumi = "dev.zio" %% "izumi-reflect" % V.izumi

    // Networking
    val comcast = "com.comcast" %% "ip4s-core" % V.comcast

    // Cryptography
    val bc = bouncyCastle("bcprov-jdk18on")
    val bcExtensions = bouncyCastle("bcpkix-jdk18on")
    val bitcoinj = ("org.bitcoinj" % "bitcoinj-core" % V.bitcoinj)
      .exclude("org.bouncycastle", "bcprov-jdk15to18")

    // Numerical
    val breeze = "org.scalanlp" %% "breeze" % V.breeze

    // Blockchain/Web3
    val web3j = "org.web3j" % "core" % V.web3j

    // IPFS
    val javaIpfsHttpClient = "com.github.ipfs" % "java-ipfs-http-client" % V.javaIpfsHttpClient

    // WASM
    val wasmtime = "io.github.kawamuray.wasmtime" % "wasmtime-java" % V.wasmtime

    // JSON parsing
    val jawnParser = jawn("jawn-parser")
    val jawnAst = jawn("jawn-ast")
    val jawnFs2 = "org.typelevel" %% "jawn-fs2" % V.jawnFs2Version

    // Utilities
    val betterFiles = "com.github.pathikrit" %% "better-files" % V.betterFiles
    val mapref = "io.chrisdavenport" %% "mapref" % V.mapref
    val micrometerPrometheusRegistry = "io.micrometer" % "micrometer-registry-prometheus" % V.micrometer

    // Logging
    val log4cats = "org.typelevel" %% "log4cats-slf4j" % V.log4cats
    val logback = "ch.qos.logback" % "logback-classic" % V.logback
    val logstashLogbackEncoder = "net.logstash.logback" % "logstash-logback-encoder" % V.logstashLogbackEncoder

    // Testing
    val weaverCats = "org.typelevel" %% "weaver-cats" % V.weaver
    val weaverDiscipline = "org.typelevel" %% "weaver-discipline" % V.weaver
    val weaverScalaCheck = "org.typelevel" %% "weaver-scalacheck" % V.weaver
    val catsEffectTestkit = "org.typelevel" %% "cats-effect-testkit" % V.catsEffect
  }
}
```

#### 3. Configure build.sbt

```scala
import Dependencies._

ThisBuild / organization := "com.example"
ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := V.scala

ThisBuild / javacOptions ++= Seq("-source", "17", "-target", "17")

ThisBuild / resolvers ++= Seq(
  "jitpack".at("https://jitpack.io")
)

lazy val root = (project in file("."))
  .settings(
    name := "my-rapp",

    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-language:implicitConversions",
      "-language:higherKinds",
      "-Xmax-inlines", "64",
      "-source:future"
    ),

    Compile / mainClass := Some("com.example.Main"),

    libraryDependencies ++= Seq(
      // Core
      Libraries.cats,
      Libraries.catsEffect,
      Libraries.catsRetry,

      // JSON
      Libraries.circeCore,
      Libraries.circeGeneric,
      Libraries.circeParser,

      // HTTP
      Libraries.http4sCore,
      Libraries.http4sDsl,
      Libraries.http4sServer,
      Libraries.http4sClient,
      Libraries.http4sCirce,
      Libraries.http4sJwtAuth,

      // Streaming
      Libraries.fs2Core,
      Libraries.fs2IO,
      Libraries.fs2ReactiveStreams,
      Libraries.fs2DataCsv,
      Libraries.fs2DataCsvGeneric,

      // Database
      Libraries.doobieCore,
      Libraries.doobieHikari,
      Libraries.flyway,
      Libraries.apacheDerby,

      // Recursion schemes
      Libraries.drosteCore,

      // Optics
      Libraries.monocleCore,
      Libraries.monocleMacro,

      // Configuration
      Libraries.cirisCore,
      Libraries.declineCore,
      Libraries.declineEffect,
      Libraries.pureconfig,

      // Refinement types
      Libraries.iron,
      Libraries.ironCats,
      Libraries.ironCirce,
      Libraries.ironDecline,

      // Type class derivation
      Libraries.kittens,
      Libraries.izumi,

      // Networking
      Libraries.comcast,

      // Cryptography
      Libraries.bc,
      Libraries.bcExtensions,
      Libraries.bitcoinj,

      // Numerical
      Libraries.breeze,

      // Blockchain/Web3
      Libraries.web3j,

      // IPFS
      Libraries.javaIpfsHttpClient,

      // WASM
      Libraries.wasmtime,

      // JSON parsing
      Libraries.jawnParser,
      Libraries.jawnAst,
      Libraries.jawnFs2,

      // Utilities
      Libraries.betterFiles,
      Libraries.mapref,
      Libraries.micrometerPrometheusRegistry,

      // Logging
      Libraries.log4cats,
      Libraries.logback % Runtime,
      Libraries.logstashLogbackEncoder % Runtime,

      // Testing
      Libraries.weaverCats % Test,
      Libraries.catsEffectTestkit % Test
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
