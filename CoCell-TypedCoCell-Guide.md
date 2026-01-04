# CoCell and TypedCoCell Documentation

A comprehensive guide to the CoCell abstraction layer for building distributed consensus systems with categorical foundations.

## Table of Contents

- [Overview](#overview)
- [CoCell (Untyped)](#cocell-untyped)
- [TypedCoCell (Typed)](#typedcocell-typed)
- [Arrow Implementations](#arrow-implementations)
- [Pipeline Composition](#pipeline-composition)
- [Configuration](#configuration)
- [BabelApp Integration](#babelapp-integration)
- [Topos Type System](#topos-type-system)
- [Theoretical Foundations: Different Arrows as Different Cohomology Theories](#theoretical-foundations-different-arrows-as-different-cohomology-theories)
- [Halt-Free Computation](#halt-free-computation)
- [Usage Examples](#usage-examples)
- [Dynamic Endpoints](#dynamic-endpoints)
- [Best Practices](#best-practices)

---

## Overview

The CoCell system provides two complementary abstractions for building blockchain state transformations:

| Abstraction | Type Safety | Use Case |
|-------------|-------------|----------|
| **CoCell** | Runtime (dynamic) | Existing apps, rapid deployment, heterogeneous composition |
| **TypedCoCell** | Compile-time (static) | New development, type-safe pipelines, verified composition |

Both share the same categorical structure and composition operators (`>>>`), but differ in when type compatibility is verified.

---

## CoCell (Untyped)

The untyped `CoCell` trait provides maximum flexibility for developers with existing deployment tools or applications that don't require compile-time type safety.

### Core Trait

```scala
trait CoCell extends Ω { self =>
  val name: String = self.getClass.getName
  val coCell: CoCell = self
  val stateChannels: List[MkStateChannel] = Nil
  val config: Option[CoCellConfig] = None

  // Lifecycle
  def setup(args: List[String]): IO[CoCell.Context[CoCell]]
  def setupFromConfig(cfg: CoCellConfig): IO[CoCell.Context[CoCell]]

  // Resource creation
  def mkResources[A <: CliMethod]: (A, SDK[IO]) => Resource[IO, HttpApi[IO]]
  def mkCell[F[_]: Async]: Ω => Cell[F, StackF, Ω, Ω, Either[CellError, Ω]]

  // Composition
  def join[F[_]](otherCell: CoCell): CoCell
  def toList: List[CoCell] = this :: Nil

  // Comonadic structure
  def ohmToCoHom: Ω => CoHomF[Ω]
  def mixHomCoHom[Z]: CoHomF[Z] => CoHomF[Z]
  def coFlatMap[B](f: CoHomF[Ω] => B): CoHomF[B]
  val coMonad: Comonad[CoHomF]
  def <~>[Z](other: CoHomF[Z]): CoHomF[Z]

  // HTTP endpoints
  def getEndpoints[F[_]]: List[(List[HttpRoutes[F]], List[HttpRoutes[F]])]
}
```

### Key Inner Types

| Type | Description |
|------|-------------|
| `CoHomF[Z]` | Comonad functor wrapping cell state |
| `TypedCoCellF[A, B]` | Typed morphism representation |
| `TypedCoCellCons[A, B]` | Constructor for typed cell chains |
| `TypedCoCellLast[A, B]` | Terminal cell in a chain |
| `Context[A]` | Runtime context containing cell and resources |

### When to Use CoCell

- Migrating existing applications to the Reality network
- Rapid prototyping without type constraints
- Heterogeneous composition of cells with different state types
- Runtime-determined cell composition

---

## TypedCoCell (Typed)

The `TypedCoCell[In, Out]` trait provides compile-time type safety for state transformations, ensuring that pipelines only compose when their types align.

### Core Trait

```scala
trait TypedCoCell[In, Out] { self =>
  def name: String
  def run[F[_]: Async](input: In): F[Either[CellError, Out]]

  // Sequential composition
  def >>>[Out2](other: TypedCoCell[Out, Out2]): TypedCoCell[In, Out2]
  def andThen[Out2](other: TypedCoCell[Out, Out2]): TypedCoCell[In, Out2]
  def compose[In2](other: TypedCoCell[In2, In]): TypedCoCell[In2, Out]

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

#### Lifting Functions

```scala
// Lift a pure function
TypedCoCell.lift[A, B](name: String)(f: A => B): TypedCoCell[A, B]

// Lift an effectful function
TypedCoCell.liftF[A, B](name: String)(f: A => IO[B]): TypedCoCell[A, B]

// Lift a fallible function
TypedCoCell.liftEither[A, B](name: String)(f: A => Either[CellError, B]): TypedCoCell[A, B]
```

#### Categorical Primitives

```scala
// Identity morphism
TypedCoCell.id[A]: TypedCoCell[A, A]

// Projections (products)
TypedCoCell.fst[A, B]: TypedCoCell[(A, B), A]  // π₁
TypedCoCell.snd[A, B]: TypedCoCell[(A, B), B]  // π₂

// Injections (coproducts)
TypedCoCell.inl[A, B]: TypedCoCell[A, Either[A, B]]  // ι₁
TypedCoCell.inr[A, B]: TypedCoCell[B, Either[A, B]]  // ι₂

// Diagonal and terminal
TypedCoCell.diagonal[A]: TypedCoCell[A, (A, A)]  // Δ: A → A × A
TypedCoCell.terminal[A]: TypedCoCell[A, Unit]   // !: A → 𝟙
```

#### Exponentials (Currying)

```scala
// Curry: (A × B → C) to (A → C^B)
TypedCoCell.curry[A, B, C](f: TypedCoCell[(A, B), C]): TypedCoCell[A, TypedCoCell[B, C]]

// Uncurry: (A → C^B) to (A × B → C)
TypedCoCell.uncurry[A, B, C](f: TypedCoCell[A, TypedCoCell[B, C]]): TypedCoCell[(A, B), C]

// Eval: C^B × B → C
TypedCoCell.eval[B, C]: TypedCoCell[(TypedCoCell[B, C], B), C]
```

### When to Use TypedCoCell

- New feature development requiring type safety
- Complex multi-stage pipelines
- Systems requiring compile-time verification
- Integration with typed state channel logic

---

## Arrow Implementations

The `CoCellArrow[F[_], A, B]` provides a reusable arrow abstraction with proper category and arrow instances.

### CoCellArrow Definition

```scala
final case class CoCellArrow[F[_], A, B](
  name: String,
  run: A => F[Either[CellError, B]]
) {
  def apply(a: A)(using F: Async[F]): F[Either[CellError, B]]
  def map[C](f: B => C)(using F: Functor[F]): CoCellArrow[F, A, C]
  def flatMap[C](f: B => CoCellArrow[F, A, C])(using F: Monad[F]): CoCellArrow[F, A, C]
  def toKleisli(using F: Functor[F]): Kleisli[F, A, Either[CellError, B]]
  def toKleisliT(using F: Functor[F]): Kleisli[EitherT[F, CellError, *], A, B]
  def toCell(input: A)(using F: Async[F], ev: A <:< Ω): Cell[F, StackF, Ω, Ω, Either[CellError, B]]
}
```

### Category Instance

```scala
def coCellCategory[F[_]](using F: Monad[F]): Category[CoCellArrow[F, *, *]]
```

Satisfies the category laws:
- **Left identity:** `id >>> f ≡ f`
- **Right identity:** `f >>> id ≡ f`
- **Associativity:** `(f >>> g) >>> h ≡ f >>> (g >>> h)`

### Arrow Instance

```scala
def coCellArrow[F[_]](using F: Monad[F]): Arrow[CoCellArrow[F, *, *]]
```

Provides:
- `lift[A, B](f: A => B)` — Lift pure function to arrow
- `first[A, B, C](fa)` — `A → B` becomes `(A, C) → (B, C)`
- `second[A, B, C](fa)` — `A → B` becomes `(C, A) → (C, B)`
- `split[A, B, C, D](f, g)` — Parallel composition `(***)`: `(A, C) → (B, D)`
- `merge[A, B, C](f, g)` — Fan-out `(&&&)`: `A → (B, C)`

### Arrow Constructors

```scala
CoCellArrow.id[F, A]: CoCellArrow[F, A, A]
CoCellArrow.lift[F, A, B](name)(f): CoCellArrow[F, A, B]
CoCellArrow.liftF[F, A, B](name)(f): CoCellArrow[F, A, B]
CoCellArrow.liftEither[F, A, B](name)(f): CoCellArrow[F, A, B]
CoCellArrow.liftEitherF[F, A, B](name)(f): CoCellArrow[F, A, B]
CoCellArrow.fail[F, A, B](error): CoCellArrow[F, A, B]
CoCellArrow.fromCoCell[F](coCell): CoCellArrow[F, Ω, Ω]
```

---

## Pipeline Composition

### Sequential Composition (`>>>`)

Data flows through cells in sequence. Errors short-circuit the pipeline.

```scala
val pipeline: TypedCoCell[A, D] = cell1 >>> cell2 >>> cell3
// A → cell1 → B → cell2 → C → cell3 → D
```

### Parallel Composition (`***`)

Both cells run concurrently on their respective tuple components.

```scala
val parallel: TypedCoCell[(A, C), (B, D)] = cell1 *** cell2
// (A, C) → (cell1(A), cell2(C)) → (B, D)
```

### Fan-out Composition (`&&&`)

The same input is processed by both cells, with results combined.

```scala
val fanOut: TypedCoCell[A, (B, C)] = cell1 &&& cell2
// A → (cell1(A), cell2(A)) → (B, C)
```

### Fan-in Composition

Handle coproducts by providing handlers for each case.

```scala
val fanIn: TypedCoCell[Either[A, B], C] = TypedCoCell.fanIn(leftHandler, rightHandler)
```

### Kleisli Composition (`>=>`)

Monadic composition for dependent cell chains.

```scala
val kleisli: TypedCoCell[A, C] = cell1 >=> (b => cell2(b))
```

---

## Configuration

### CoCellConfig Sealed Trait

```scala
sealed trait CoCellConfig {
  def name: String
  def keyConfig: KeyConfig
  def environment: AppEnvironment
  def httpConfig: HttpConfig
  def snapshotConfig: SnapshotConfig
  def seedlistPath: Option[Path]
  def collateralAmount: Option[Amount]
}
```

### L0CoCellConfig

Configuration for L0 (base layer) cells:

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

Configuration for L1 (state channel) cells:

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

Dynamic configuration generation for multi-node clusters:

```scala
object CoCellConfigGenerator {
  def allocatePorts(): (Port, Port, Port)
  def resetPortCounter(startPort: Int = 9000): Unit
  def defaultHttpConfig(...): HttpConfig
  def defaultL0DBConfig(): L0DBConfig
  def defaultL1DBConfig(): L1DBConfig
  def generateL0GenesisConfig(...): L0CoCellConfig
  def generateL0ValidatorConfig(...): L0CoCellConfig
  def generateL1InitialValidatorConfig(...): L1CoCellConfig
  def generateL1ValidatorConfig(...): L1CoCellConfig
}
```

### StaticConfigs

Pre-defined configurations for common scenarios:

| Config | Description |
|--------|-------------|
| `StaticConfigs.SingleNodeGenesis` | Single-node with L0 + L1 pair |
| `StaticConfigs.TwoNodePairs` | Two L0/L1 pairs for testing |
| `StaticConfigs.Validators` | Validator node configurations |

---

## BabelApp Integration

BabelApp provides the bridge between TypedCoCell and the runtime system.

### TypedCoCellNode

Extends TypedCoCell with node lifecycle management:

```scala
trait TypedCoCellNode[In <: Ω, Out <: Ω] extends TypedCoCell[In, Out] {
  def nodeName: String = name
  def config: Option[CoCellConfig] = None
  def stateChannels: List[MkStateChannel] = Nil

  def mkResources[A <: CliMethod]: (A, SDK[IO]) => Resource[IO, HttpApi[IO]]
  def argsToStartUp(args: List[String]): StartUp
  def setup(args: List[String]): IO[TypedCoCellContext[In, Out]]
  def setupFromConfig(cfg: CoCellConfig): IO[TypedCoCellContext[In, Out]]
}
```

### TypedCoCellContext

Runtime context for a running node:

```scala
case class TypedCoCellContext[In <: Ω, Out <: Ω](
  cell: TypedCoCellNode[In, Out],
  resource: (Resource[IO, CoCellInternals], SignallingRef[IO, Unit])
) {
  def nodeResource: Resource[IO, NodeInternals]
  def coCellResource: Resource[IO, CoCellInternals]
  def restartSignal: SignallingRef[IO, Unit]
  def runCell(input: In): IO[Either[CellError, Out]]
}
```

### CoCellPipeline

Type-safe pipeline construction:

```scala
val pipeline: CoCellPipeline[L0InputState, StateChannelState] =
  CoCellPipeline.from(l0Node) >>> wasmStateChannelNode
```

---

## Topos Type System

The CoCell system is built on topos-theoretic foundations, providing categorical primitives as first-class types.

### Terminal and Initial Objects

```scala
Ω𝟙  // Terminal object (unit type)
Ω𝟘  // Initial object with absurd: Ω𝟘 => A
```

### Products (`×`)

```scala
type ×[A, B] = (A, B)
type ⊗[A, B] = (A, B)

// Operations
def assocL[A, B, C]: ((A, B), C) => (A, (B, C))
def assocR[A, B, C]: (A, (B, C)) => ((A, B), C)
def swap[A, B]: (A, B) => (B, A)
def bimap[A, B, C, D](f: A => C, g: B => D): (A, B) => (C, D)
```

### Coproducts (`⊕`)

```scala
sealed trait ⊕[+A, +B]
case class Inl[A](value: A) extends ⊕[A, Nothing]
case class Inr[B](value: B) extends ⊕[Nothing, B]

// Operations
def fold[A, B, C](left: A => C, right: B => C): ⊕[A, B] => C
def swap[A, B]: ⊕[A, B] => ⊕[B, A]
def bimap[A, B, C, D](f: A => C, g: B => D): ⊕[A, B] => ⊕[C, D]
```

### Coproduct-based Conditional Routing

The Reality SDK provides proper categorical coproducts for conditional routing. Unlike simulated conditionals that run both branches and select, coproduct-based routing executes only the chosen path.

#### The Route Morphism

```scala
// Route: A → A ⊕ A (injection based on predicate)
TypedCoCell.route[A](predicate: A => Boolean): TypedCoCell[A, A ⊕ A]

// Route with transformation: A → B ⊕ C
TypedCoCell.routeWith[A, B, C](
  predicate: A => Boolean,
  ifTrue: A => B,
  ifFalse: A => C
): TypedCoCell[A, B ⊕ C]
```

#### The Conditional Combinator

The `conditional` combinator decomposes as: `route(p) >>> fanIn(f, g)`

```scala
// Conditional: only ONE path executes
TypedCoCell.conditional[A, C](
  predicate: A => Boolean,
  ifTrue: TypedCoCell[A, C],
  ifFalse: TypedCoCell[A, C]
): TypedCoCell[A, C]

// Divergent conditional: different output types
TypedCoCell.conditionalDiverge[A, B, C](
  predicate: A => Boolean,
  ifTrue: TypedCoCell[A, B],
  ifFalse: TypedCoCell[A, C]
): TypedCoCell[A, B ⊕ C]
```

#### Categorical Laws

The conditional combinator satisfies important categorical laws:

```scala
// Idempotency: choosing the same path both ways is identity
conditional(p, f, f) ≅ f

// Left identity: true predicate always takes left path
conditional(_ => true, f, g) ≅ f

// Right identity: false predicate always takes right path
conditional(_ => false, f, g) ≅ g
```

#### Example Usage

```scala
case class SensorReading(deviceId: String, value: Double) extends Ω
case class Alert(level: String, message: String) extends Ω

val highPath: TypedCoCell[SensorReading, Alert] =
  TypedCoCell.lift("highPath") { r =>
    Alert("HIGH", s"High reading: ${r.value}")
  }

val lowPath: TypedCoCell[SensorReading, Alert] =
  TypedCoCell.lift("lowPath") { r =>
    Alert("LOW", s"Normal reading: ${r.value}")
  }

// Proper coproduct routing: only ONE path executes
val routing: TypedCoCell[SensorReading, Alert] =
  TypedCoCell.conditional(
    predicate = _.value > 30.0,
    ifTrue = highPath,
    ifFalse = lowPath
  )
```

### Cohomological Routing with Proofs

For blockchain consensus, validators must verify which execution path was taken. The SDK provides traced routing that generates cryptographic proofs.

#### Branch Proofs

```scala
// BranchPath: which injection was used (ι₁ or ι₂)
sealed trait BranchPath extends Ω
case object LeftPath extends BranchPath   // ι₁
case object RightPath extends BranchPath  // ι₂

// BranchProof: witness of routing decision
sealed trait BranchProof[+A <: Ω] extends Ω {
  def path: BranchPath
  def witness: A
  def inputHash: String  // For efficient verification
}
```

#### Traced Routing

```scala
// Traced route: routing with proof
TypedCoCell.tracedRoute[A](
  predicate: A => Boolean
): TypedCoCell[A, TracedCoproduct[A, A]]

// Traced conditional: returns result with proof
TypedCoCell.tracedConditional[A, C](
  predicate: A => Boolean,
  ifTrue: TypedCoCell[A, C],
  ifFalse: TypedCoCell[A, C]
): TypedCoCell[A, TracedResult[C, A]]
```

#### TracedResult

The `TracedResult` type bundles a computation result with its routing proof:

```scala
final case class TracedResult[A <: Ω, W <: Ω](
  result: A,
  proof: BranchProof[W]
) extends Ω {
  def map[B <: Ω](f: A => B): TracedResult[B, W]
  def path: BranchPath = proof.path
  def inputHash: String = proof.inputHash
}
```

#### TracedCoproduct

For preserving proofs through pipelines:

```scala
sealed trait TracedCoproduct[+A <: Ω, +B <: Ω] extends Ω {
  def value: A ⊕ B           // The underlying coproduct
  def proof: BranchProof[Ω]  // Proof of injection
  def untraced: A ⊕ B        // Discard proof
}

case class TracedInl[A <: Ω, B <: Ω](a: A, inputWitness: Ω)
    extends TracedCoproduct[A, B]
case class TracedInr[A <: Ω, B <: Ω](b: B, inputWitness: Ω)
    extends TracedCoproduct[A, B]
```

#### Traced Fan-In

Process a traced coproduct while preserving the proof:

```scala
TypedCoCell.tracedFanIn[A, B, C](
  left: TypedCoCell[A, C],
  right: TypedCoCell[B, C]
): TypedCoCell[TracedCoproduct[A, B], (C, BranchProof[Ω])]
```

#### Cohomological Interpretation

In blockchain cohomology:

| Concept | Meaning |
|---------|---------|
| **Route** | Branching coboundary - creates cochains |
| **BranchProof** | Cohomological witness - records the decision |
| **TracedCoproduct** | Enriched coproduct - carries its proof |
| **RoutingTrace** | Cochain complex - accumulates proofs |

The proofs form a "merkle-like" structure for verifying execution paths:

```scala
final case class RoutingTrace(proofs: List[BranchProof[Ω]]) extends Ω {
  lazy val traceHash: String = proofs.map(_.inputHash).mkString("-")
}
```

#### Pipeline Integration

Use `tracedConditionalRoute` on CoCellPipeline:

```scala
val pipeline: CoCellPipeline[Input, TracedResult[Alert, SensorReading]] =
  CoCellPipeline.from(calibrateNode)
    .tracedConditionalRoute(
      predicate = _.value > 30.0,
      ifTrue = highAlertCell,
      ifFalse = lowAlertCell
    )
```

This creates an auditable execution trace where validators can verify:
- Which branch was taken
- What input determined the routing
- The hash of the routing decision

### Exponential Objects

```scala
case class Exp[A, B](apply: A => B) {
  def andThen[C](other: Exp[B, C]): Exp[A, C]
  def compose[Z](other: Exp[Z, A]): Exp[Z, B]
}

object Exp {
  def pure[A, B](f: A => B): Exp[A, B]
  def id[A]: Exp[A, A]
}
```

### Subobject Classifier

```scala
sealed trait Ωₜ extends Ω {
  def &&(other: Ωₜ): Ωₜ
  def ||(other: Ωₜ): Ωₜ
  def unary_! : Ωₜ
  def ==>(other: Ωₜ): Ωₜ
}

case object ΩTrue extends Ωₜ
case object ΩFalse extends Ωₜ
```

### Partial Map Classifier

```scala
sealed trait ΩMaybe[+A] extends Ω {
  def fold[B](ifNone: => B)(ifSome: A => B): B
  def map[B](f: A => B): ΩMaybe[B]
  def flatMap[B](f: A => ΩMaybe[B]): ΩMaybe[B]
  def getOrElse[B >: A](default: => B): B
}

case object ΩNone extends ΩMaybe[Nothing]
case class ΩSome[A](value: A) extends ΩMaybe[A]
```

### Natural Numbers Object

```scala
sealed trait ΩNat extends Ω {
  def succ: ΩNat = ΩSucc(this)
  def +(other: ΩNat): ΩNat
  def *(other: ΩNat): ΩNat
  def fold[A](zero: A)(succ: A => A): A  // Primitive recursion
}

case object ΩZero extends ΩNat
case class ΩSucc(n: ΩNat) extends ΩNat
```

---

## Theoretical Foundations: Different Arrows as Different Cohomology Theories

In category theory, an **Arrow** defines how morphisms compose. Having different arrows for CoCell vs TypedCoCell means they represent **distinct cohomology theories** over the same underlying blockchain state space.

### The Grading Distinction

```
TypedCoCell[In, Out]  →  Graded cochain complex
                         δ: C^n(A→B) ⊗ C^{n+1}(B→C) → C^{n+2}(A→C)

CoCell                →  Ungraded/dynamically-graded complex
                         δ: C^n ⊗ C^{n+1} → C^{n+2}
                         (type compatibility checked at runtime)
```

The type parameters `[In, Out]` define an **explicit grading** on the cochain complex. The `>>>` operator is the coboundary map, and types enforce that cochains only compose when their boundaries align.

### Cohomological Interpretation

| Aspect | TypedCoCell | CoCell |
|--------|-------------|--------|
| Grading | Compile-time (types) | Runtime (dynamic) |
| Coboundary | `δ` is a typed functor | `δ` is a partial function |
| Cohomology class | Statically-verified | Dynamically-verified |
| Obstruction detection | Compile error | Runtime error |

### What This Means Practically

The two arrows represent **dual perspectives** on the same blockchain transformations:

1. **TypedCoCell arrow**: A **refined cohomology** where type alignment is a cocycle condition. Invalid compositions are *obstructions* caught at compile time.

2. **CoCell arrow**: A **coarse cohomology** where the cocycle condition is deferred. This is analogous to working with singular chains before taking homology—you have more flexibility but less structure.

They're related by a **forgetful functor**: every TypedCoCell can be viewed as a CoCell (erasing type info), but not vice versa. This functor induces a map between cohomology theories.

### The Deeper Point

Having both is *cohomologically natural*. Just as de Rham, Čech, and singular cohomology all describe the same topological invariants with different computational trade-offs, CoCell and TypedCoCell describe the same blockchain state transformations with different verification trade-offs.

- **CoCell**: The "flexible chain complex" for rapid deployment—analogous to working with singular chains where boundary conditions are checked operationally
- **TypedCoCell**: The "refined complex" where cohomological consistency is a compile-time invariant—analogous to de Rham cohomology where differential forms carry intrinsic type information

This duality allows developers to choose their verification strategy: pay upfront with types (TypedCoCell) or defer to runtime (CoCell), while maintaining the same categorical semantics.

---

## Halt-Free Computation

The Babel is a **language for halt-free computation**. Unlike general-purpose programming languages where the halting problem applies, the Babel defines a computational model where termination is guaranteed by construction.

### Why Halt-Free?

The DSL combinators **preserve termination structurally**:

| Combinator | Termination Property |
|------------|---------------------|
| `f >>> g` | If f and g halt, composition halts |
| `f &&& g` | If f and g halt, fan-out halts |
| `f *** g` | If f and g halt, product halts |
| `fanIn(f, g)` | If both branches halt, case analysis halts |
| `conditional(p, f, g)` | If predicate and branches halt, routing halts |
| `route(p)` | Predicate application, always halts |

The DSL describes a **dataflow graph**, and finite dataflow graphs are inherently halt-free—they're not Turing-complete.

### Categorical Semantics = Totality

In category theory, a morphism `A → B` **is** a total function. There's no representation for "this morphism might not return." The Babel is essentially:

```
CoCell programs ≅ Morphisms in a symmetric monoidal category with coproducts
```

Non-termination doesn't exist in this semantic model. Every `TypedCoCell[A, B]` is a morphism that, by definition, produces a `B` for every `A`.

### The `lift` Boundary

```scala
// The DSL is halt-free; the embedded code is contractually total
TypedCoCell.lift("pure")(f: A => B)
```

The `lift` combinator is the boundary between the halt-free DSL and the host language. The DSL *assumes* lifted functions are total—this is a **contract** between the developer and the framework.

For guaranteed totality, restrict lifted functions to:
- Pure arithmetic operations
- Pattern matching on finite algebraic data types
- Primitive recursive functions
- Operations that compile to finite arithmetic circuits

### Comparison to Other Halt-Free Languages

| Language | Domain | Halt-Free By |
|----------|--------|--------------|
| SQL | Queries | Relational algebra is finite |
| Regular expressions | Pattern matching | Finite automata |
| Arithmetic circuits | zkSNARKs | Fixed gate count |
| Datalog | Logic queries | Stratified negation |
| **Babel** | State transforms | Categorical morphisms |

### The zkSNARK Connection

The halt-free property connects directly to zkSNARK proof generation:

```
CoCell pipeline → Arithmetic circuit → zkSNARK proof
     ↓                   ↓                  ↓
 Halt-free DSL      Finite gates       Witness exists
```

A zkSNARK circuit is **finite by construction**—you cannot have an infinite arithmetic circuit. When a CoCell pipeline produces a zkSNARK proof, the proof's existence **is** the termination certificate.

This creates a powerful guarantee for blockchain consensus:
- The proof witnesses that computation completed
- Validators verify the proof, not re-execute the computation
- The halt-free DSL ensures the proof can always be generated

### Implications for Distributed Systems

In distributed consensus, halt-freedom provides critical guarantees:

1. **Deterministic Execution**: Every node executing the same CoCell pipeline on the same input will terminate with the same result
2. **Bounded Resources**: Pipeline execution has predictable resource consumption
3. **Consensus Safety**: No node can be stalled by non-terminating computation
4. **Proof Generation**: zkSNARK proofs can always be generated for valid executions

### Not Solving the Halting Problem

The Babel doesn't *solve* the halting problem—that's provably impossible. Instead, it **sidesteps** the problem by defining a computational model where:

- Programs are morphisms (total by definition)
- Control flow is coproducts (finite branching)
- Data flow is products (parallel composition)
- The structure guarantees termination

The halting problem asks "will this program halt?" The Babel answers "that question doesn't apply here—these programs are halt-free by construction."

---

## Usage Examples

### Example 1: Simple TypedCoCell Pipeline

```scala
import org.reality.combined.topos.TypedCoCell
import org.reality.kernel.{Ω, CellError}

// Define state types
case class L0InputState(data: String) extends Ω
case class L0OutputState(processed: String, hash: String) extends Ω
case class FinalState(result: String) extends Ω

// Define cells
val l0Cell: TypedCoCell[L0InputState, L0OutputState] =
  TypedCoCell.lift("L0Process") { input =>
    L0OutputState(
      processed = input.data.toUpperCase,
      hash = input.data.hashCode.toHexString
    )
  }

val finalCell: TypedCoCell[L0OutputState, FinalState] =
  TypedCoCell.lift("Finalize") { input =>
    FinalState(s"${input.processed}-${input.hash}")
  }

// Compose pipeline (type-safe!)
val pipeline: TypedCoCell[L0InputState, FinalState] = l0Cell >>> finalCell

// Run
val result: IO[Either[CellError, FinalState]] =
  pipeline.run[IO](L0InputState("hello"))
```

### Example 2: Parallel Processing

```scala
val processA: TypedCoCell[Input, OutputA] = ???
val processB: TypedCoCell[Input, OutputB] = ???

// Fan-out: same input to both cells
val fanOut: TypedCoCell[Input, (OutputA, OutputB)] = processA &&& processB

// Or with different inputs (parallel)
val parallel: TypedCoCell[(InputA, InputB), (OutputA, OutputB)] =
  cellA *** cellB
```

### Example 3: CoCell with Static Configuration

```scala
import org.reality.combined.{CoCell, L0CoCellConfig, StaticConfigs}

case class MyL0Cell(staticConfig: Option[L0CoCellConfig] = None) extends CoCell {
  override val config: Option[CoCellConfig] = staticConfig
  override val name: String = staticConfig.map(_.name).getOrElse("MyL0Cell")

  // ... cell implementation
}

object MyL0Cell {
  def withConfig(cfg: L0CoCellConfig): MyL0Cell = MyL0Cell(Some(cfg))
}

// Usage
val cell = MyL0Cell.withConfig(StaticConfigs.SingleNodeGenesis.l0Config)
val context = cell.setup(args).unsafeRunSync()
```

### Example 4: BabelApp Multi-Node Cluster

```scala
import org.reality.combined.{BabelApp, TypedCoCellNode, CoCellPipeline}

object MyCluster extends BabelApp[L0InputState, FinalState] {

  val l0Node: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(MyL0Typed(), Some(l0Config))

  val l1Node: TypedCoCellNode[L0OutputState, FinalState] =
    BabelApp.l1Node(MyL1Typed(), MkMyStateChannel, Some(l1Config))

  val pipeline: CoCellPipeline[L0InputState, FinalState] =
    CoCellPipeline.from(l0Node) >>> l1Node

  val pipelines: List[CoCellPipeline[L0InputState, FinalState]] =
    List(pipeline)
}
```

### Example 5: Arrow Composition

```scala
import org.reality.combined.CoCellArrow

val toStringArrow: CoCellArrow[IO, Int, String] =
  CoCellArrow.lift[IO, Int, String]("toString")(_.toString)

val lengthArrow: CoCellArrow[IO, String, Int] =
  CoCellArrow.lift[IO, String, Int]("length")(_.length)

// Sequential
val composed: CoCellArrow[IO, Int, Int] = toStringArrow >>> lengthArrow

// Parallel
val parallel: CoCellArrow[IO, (Int, Int), (String, String)] =
  toStringArrow *** toStringArrow

// Fan-out
val fanOut: CoCellArrow[IO, Int, (String, Int)] =
  toStringArrow &&& lengthArrow.compose(toStringArrow)
```

---

## Error Handling

All cell executions return `F[Either[CellError, Out]]`. Errors propagate through composition:

```scala
// If cell1 fails, cell2 and cell3 never run
val pipeline = cell1 >>> cell2 >>> cell3

// Handle errors
pipeline.run[IO](input).flatMap {
  case Right(output) => IO.println(s"Success: $output")
  case Left(error)   => IO.println(s"Error: ${error.message}")
}
```

---

## Dynamic Endpoints

TypedCoCell nodes can dynamically add HTTP endpoints at runtime. This allows TypedCoCells to expose custom REST APIs alongside the standard node endpoints on both L0 and L1 layers.

### Required Imports

```scala
import cats.effect.{Async, IO}
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.semigroupk.*
import io.circe.Json
import org.http4s.HttpRoutes
import org.http4s.circe.*
import org.http4s.circe.CirceEntityCodec.circeEntityDecoder
import org.http4s.dsl.Http4sDsl
import org.bouncycastle.jce.provider.BouncyCastleProvider
import org.reality.modules.{AdditionalRoutes, HttpApi}
import org.reality.security.SecurityProvider
```

### Method Signatures

Both `BabelApp.l0Node` and `BabelApp.l1Node` accept an optional `customRoutes` parameter:

```scala
// L0 node with optional custom routes
def l0Node[In <: Ω, Out <: Ω](
  cell: TypedCoCell[In, Out],
  nodeConfig: Option[L0CoCellConfig] = None,
  customRoutes: List[HttpApi[IO] => AdditionalRoutes[IO]] = Nil
): TypedCoCellNode[In, Out]

// L1 node with optional custom routes
def l1Node[In <: Ω, Out <: Ω](
  cell: TypedCoCell[In, Out],
  stateChannel: MkStateChannel,
  nodeConfig: Option[L1CoCellConfig] = None,
  customRoutes: List[HttpApi[IO] => AdditionalRoutes[IO]] = Nil
): TypedCoCellNode[In, Out]
```

### Complete Working Example

Here is a complete example showing custom endpoints on both L0 and L1:

```scala
package org.reality.combined.examples

import cats.effect.IO
import cats.syntax.flatMap.*
import cats.syntax.functor.*
import cats.syntax.semigroupk.*
import io.circe.Json
import org.http4s.HttpRoutes
import org.http4s.circe.*
import org.http4s.circe.CirceEntityCodec.circeEntityDecoder
import org.http4s.dsl.Http4sDsl
import org.bouncycastle.jce.provider.BouncyCastleProvider

import org.reality.combined.{BabelApp, StaticConfigs, TypedCoCellNode}
import org.reality.combined.topos.CoCellPipeline
import org.reality.dag.l1.L1
import org.reality.modules.{AdditionalRoutes, HttpApi}
import org.reality.security.SecurityProvider

/**
 * Example: TypedCoCell cluster with custom HTTP endpoints on both L0 and L1.
 *
 * L0 Custom Endpoints (port 9000):
 * - GET  /l0/info     - Returns L0 node information
 * - GET  /l0/metrics  - Returns custom L0 metrics
 *
 * L1 Custom Endpoints (port 9010):
 * - GET  /l1/info     - Returns L1 node information
 * - POST /l1/submit   - Accepts data submissions
 */
open class TypedCocellWithCustomEndpoints extends BabelApp[L0InputState, StateChannelState] {

  // SecurityProvider is required for route creation
  given SecurityProvider[IO] = new SecurityProvider[IO] {
    val provider: BouncyCastleProvider = new BouncyCastleProvider()
  }

  /**
   * L0 Custom Routes
   * Endpoints available on L0 public HTTP port (default 9000)
   */
  val l0CustomRoutes: HttpApi[IO] => AdditionalRoutes[IO] = { api =>
    new AdditionalRoutes[IO] with Http4sDsl[IO] {

      // GET /l0/info - Returns L0 node information
      private val l0InfoRoute: HttpRoutes[IO] = HttpRoutes.of[IO] {
        case GET -> Root / "l0" / "info" =>
          Ok(Json.obj(
            "layer" -> Json.fromString("L0"),
            "nodeId" -> Json.fromString(api.selfId.toString),
            "version" -> Json.fromString(api.nodeVersion),
            "type" -> Json.fromString("Genesis Node"),
            "description" -> Json.fromString("L0 base layer with custom endpoints")
          ))
      }

      // GET /l0/metrics - Returns custom L0 metrics
      private val l0MetricsRoute: HttpRoutes[IO] = HttpRoutes.of[IO] {
        case GET -> Root / "l0" / "metrics" =>
          Ok(Json.obj(
            "layer" -> Json.fromString("L0"),
            "timestamp" -> Json.fromLong(System.currentTimeMillis()),
            "customMetric1" -> Json.fromInt(42),
            "customMetric2" -> Json.fromDouble(3.14).getOrElse(Json.Null),
            "status" -> Json.fromString("operational")
          ))
      }

      override val publicRoutes: HttpRoutes[IO] = l0InfoRoute <+> l0MetricsRoute
      override val p2pRoutes: HttpRoutes[IO] = HttpRoutes.empty[IO]
    }
  }

  /**
   * L1 Custom Routes
   * Endpoints available on L1 public HTTP port (default 9010)
   */
  val l1CustomRoutes: HttpApi[IO] => AdditionalRoutes[IO] = { api =>
    new AdditionalRoutes[IO] with Http4sDsl[IO] {

      // GET /l1/info - Returns L1 node information
      private val l1InfoRoute: HttpRoutes[IO] = HttpRoutes.of[IO] {
        case GET -> Root / "l1" / "info" =>
          Ok(Json.obj(
            "layer" -> Json.fromString("L1"),
            "nodeId" -> Json.fromString(api.selfId.toString),
            "version" -> Json.fromString(api.nodeVersion),
            "type" -> Json.fromString("State Channel Node"),
            "description" -> Json.fromString("L1 state channel with custom endpoints")
          ))
      }

      // POST /l1/submit - Accepts data submissions
      private val l1SubmitRoute: HttpRoutes[IO] = HttpRoutes.of[IO] {
        case req @ POST -> Root / "l1" / "submit" =>
          for {
            body <- req.as[Json]
            payload = body.hcursor.downField("payload").as[String].getOrElse("empty")
            response <- Ok(Json.obj(
              "layer" -> Json.fromString("L1"),
              "received" -> Json.fromString(payload),
              "processedAt" -> Json.fromLong(System.currentTimeMillis()),
              "status" -> Json.fromString("accepted"),
              "nodeId" -> Json.fromString(api.selfId.toString)
            ))
          } yield response
      }

      override val publicRoutes: HttpRoutes[IO] = l1InfoRoute <+> l1SubmitRoute
      override val p2pRoutes: HttpRoutes[IO] = HttpRoutes.empty[IO]
    }
  }

  /**
   * L0 Genesis node with custom endpoints
   */
  val l0Node: TypedCoCellNode[L0InputState, L0OutputState] =
    BabelApp.l0Node(
      CombinedL0Typed.withConfig(StaticConfigs.SingleNodeGenesis.l0Config),
      Some(StaticConfigs.SingleNodeGenesis.l0Config),
      List(l0CustomRoutes)  // <-- L0 custom endpoints
    )

  /**
   * L1 State Channel node with custom endpoints
   */
  val stateChannelNode: TypedCoCellNode[L0OutputState, StateChannelState] =
    BabelApp.l1Node(
      CombinedStateChannelTyped.withConfig(StaticConfigs.SingleNodeGenesis.l1Config),
      L1,
      Some(StaticConfigs.SingleNodeGenesis.l1Config),
      List(l1CustomRoutes)  // <-- L1 custom endpoints
    )

  val pipelines: List[CoCellPipeline[L0InputState, StateChannelState]] =
    List(CoCellPipeline.from(l0Node) >>> stateChannelNode)
}
```

### Creating a Reusable Routes Factory Class

For reusable routes, create a separate class:

```scala
class CustomCellRoutes[F[_]: {Async, SecurityProvider}](nodeApi: HttpApi[F])
    extends AdditionalRoutes[F] with Http4sDsl[F] {

  // Custom endpoint: GET /cell/status
  private val cellStatusRoutes: HttpRoutes[F] = HttpRoutes.of[F] {
    case GET -> Root / "cell" / "status" =>
      Ok(Json.obj(
        "status" -> Json.fromString("running"),
        "nodeId" -> Json.fromString(nodeApi.selfId.toString),
        "version" -> Json.fromString(nodeApi.nodeVersion)
      ))
  }

  // Custom endpoint: POST /cell/process
  private val cellProcessRoutes: HttpRoutes[F] = HttpRoutes.of[F] {
    case req @ POST -> Root / "cell" / "process" =>
      for {
        body <- req.as[Json]
        data = body.hcursor.downField("data").as[String].getOrElse("no data")
        response <- Ok(Json.obj(
          "processed" -> Json.fromString(s"Processed: $data"),
          "timestamp" -> Json.fromLong(System.currentTimeMillis())
        ))
      } yield response
  }

  // Combine all public routes using <+>
  override val publicRoutes: HttpRoutes[F] = cellStatusRoutes <+> cellProcessRoutes
  override val p2pRoutes: HttpRoutes[F] = HttpRoutes.empty[F]
}

// Usage with factory function
val customRoutes: HttpApi[IO] => AdditionalRoutes[IO] = { api =>
  given SecurityProvider[IO] = new SecurityProvider[IO] {
    val provider = new BouncyCastleProvider()
  }
  new CustomCellRoutes[IO](api)
}
```

### HttpApi Access

The `HttpApi` parameter provides access to the node's runtime context:

| Property | Description |
|----------|-------------|
| `selfId` | This node's PeerId |
| `nodeVersion` | Build version string |
| `storages` | Access to node storage layers |
| `queues` | Access to processing queues |
| `services` | Node services (cluster, session, etc.) |
| `programs` | High-level operations (joining, discovery) |
| `key` | Node's KeyPair for signing |

### Running and Testing

Start the cluster:

```bash
sbt "combined/runMain org.reality.combined.examples.TypedCocellWithCustomEndpoints"
```

Test L0 endpoints (port 9000):

```bash
# Get L0 node info
curl http://localhost:9000/l0/info

# Get L0 metrics
curl http://localhost:9000/l0/metrics
```

Test L1 endpoints (port 9010):

```bash
# Get L1 node info
curl http://localhost:9010/l1/info

# Submit data to L1
curl -X POST -H "Content-Type: application/json" \
     -d '{"payload": "test data"}' http://localhost:9010/l1/submit
```

Example responses:

```json
// GET /l0/info
{
  "layer": "L0",
  "nodeId": "00b8a56a20fc2e2a...",
  "version": "0.0.0+1019-9e64fe47",
  "type": "Genesis Node",
  "description": "L0 base layer with custom endpoints"
}

// POST /l1/submit
{
  "layer": "L1",
  "received": "test data",
  "processedAt": 1767019489218,
  "status": "accepted",
  "nodeId": "fbf91bc197ece694..."
}
```

### Important Notes

1. **SecurityProvider Required**: The `given SecurityProvider[IO]` must be in scope when creating routes. The simplest approach is to use BouncyCastleProvider.

2. **Import cats syntax**: Make sure to import `cats.syntax.flatMap.*`, `cats.syntax.functor.*`, and `cats.syntax.semigroupk.*` for the route combinators and for-comprehensions to work.

3. **Multiple Routes**: Pass multiple route factories as a `List`:
   ```scala
   BabelApp.l1Node(cell, L1, Some(config), List(routes1, routes2, routes3))
   ```

4. **P2P Routes**: Custom P2P routes are also supported via the `p2pRoutes` member of `AdditionalRoutes`.

---

## Best Practices

1. **Prefer TypedCoCell** for new development to catch composition errors at compile time

2. **Use CoCell** when integrating existing systems or when types are determined at runtime

3. **Keep cells small and focused** — each cell should do one thing well

4. **Use fan-out (`&&&`)** when you need the same input processed multiple ways

5. **Use parallel (`***`)** when processing independent data streams

6. **Leverage configuration** — use `CoCellConfig` for environment-specific settings rather than hardcoding

7. **Test pipelines in isolation** — each TypedCoCell can be tested independently before composition

8. **Use dynamic endpoints** for cell-specific REST APIs via `BabelApp.l1NodeWithRoutes`
