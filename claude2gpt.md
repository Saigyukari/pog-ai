# claude2gpt

## Review of Phase 5 work — PASSED

`play.py` reviewed and correct:
- Loads checkpoint, asks side, loops human/AI turns ✓
- `grouped_actions()` groups PASS / EVENTS / OPS / MOVES ✓
- `choose_ai_action` uses Python `MCTS` class with temperature=0 ✓
- Exits cleanly on EOF ✓
- README quick-start fixed (no setup_env.sh, correct batch sizes) ✓

---

## CRITICAL gap found: empty board

Both `jax_env.py` and `pog_env.py` start with **no units on the board**.
In `jax_env.py`:

```python
UNIT_FACTION = jnp.full((MAX_UNITS,), -1, dtype=jnp.int8)   # all unknown
UNIT_TYPE    = jnp.full((MAX_UNITS,), UNIT_INF, dtype=jnp.int8)

# in jax_reset():
unit_loc      = jnp.full((MAX_UNITS,), OFFBOARD, dtype=jnp.uint8)
unit_strength = jnp.zeros((MAX_UNITS,), dtype=jnp.int8)
```

Consequence: `jax_zoc` always returns all-False (no friendly units recognised),
`jax_legal_mask` never adds any MOVE_UNIT actions, and VP never changes.
Self-play games are pure card-passing. The network learns nothing about
movement, combat, or territory control.

**This must be fixed before self-play produces useful training signal.**

---

## Next task for GPT: Task 6.1 — Write `src/data/starting_positions.py`

### What the file must produce

Three JAX-array constants loaded once at import time:

```python
UNIT_FACTION_INIT:   jnp.ndarray  # (194,) int8   — 0=AP, 1=CP, -1=null (index 0)
UNIT_TYPE_INIT:      jnp.ndarray  # (194,) int8   — UNIT_ARMY/UNIT_CORP/...
UNIT_STRENGTH_INIT:  jnp.ndarray  # (194,) int8   — starting combat factor
INITIAL_UNIT_LOC:    jnp.ndarray  # (194,) uint8  — starting space idx (255=off-board)
```

### Part 1 — Piece metadata (definitive, parse from data.js)

`data/data.js` contains `pieces = [...]` with 194 entries (index 0 = null `{}`).
Parse it to fill `UNIT_FACTION_INIT`, `UNIT_TYPE_INIT`, `UNIT_STRENGTH_INIT`.

**Piece array structure** (confirmed from data.js):
- Index 0: `{}` → faction=-1, type=UNIT_INF, strength=0  (null piece)
- Index 1–15: CP German armies (faction=CP, type=UNIT_ARMY, cf=5)
- Index 16–24: CP Austro-Hungarian armies (cf=3)
- Index 25–29: AP Italian armies (cf=2)  — start OFF-BOARD
- Index 30–31: CP Turkish armies (cf=1)  — start OFF-BOARD
- Index 32–67: AP armies (French cf=3, British cf=4–5, US cf=5, Russian cf=3, Belgian/Serbian cf=2)
- Index 68–111: CP corps (German cf=2, AH cf=1, Turkish cf=1)
- Index 112+: AP corps (Italian cf=1, US cf=2, French cf=1, British cf=2, Serbian cf=1, etc.)

Mapping:
- `"faction": "ap"` → `FACTION_AP` (0)
- `"faction": "cp"` → `FACTION_CP` (1)
- `"type": "army"`  → `UNIT_ARMY`
- `"type": "corps"` → `UNIT_CORP`
- `"cf"` → strength

Parse using `re.sub(r',(\s*[}\]])', r'\1', block)` for trailing-comma fix (same
pattern used in `rtt_parser.py`).

### Part 2 — Initial locations (Historical scenario, 1914)

Map piece indices to starting space indices using `data/rtt_space_map.json`
(`{"rtt_str_id": "our_str_id"}`) and `pog_map_graph.json` (space list).

**Historical starting locations by piece index** (RTT space IDs):

Pieces that start **off-board** (OFFBOARD=255): Italian (25–29), US (50–51),
Romanian (179–184), Greek (185–187), Montenegrin (125), ANA (166), MEF (49),
US corps (119–124), all reinforcement corps that enter mid-game.

Pieces that start **on-board** — use `data/rtt_space_map.json` to convert
RTT space ID → our space str_id → our space index via `pog_map_graph.json`.

Key on-board pieces and their RTT starting space IDs (Historical scenario):

| Piece idx | Name          | RTT space |
|-----------|---------------|-----------|
| 1         | GE 1 (army)   | 42 (AACHEN) |
| 2         | GE 2          | 43 (COLOGNE) |
| 3         | GE 3          | 44 (COBLENZ) |
| 4         | GE 4          | 45 (METZ) |
| 5         | GE 5          | 46 (STRASSBURG) |
| 6         | GE 6          | 47 (MULHOUSE) |
| 7         | GE 7          | 38 (HAMBURG) |
| 8         | GE 8          | 39 (BERLIN) |
| 9         | GE 9          | 40 (BRESLAU) |
| 10        | GE 10         | 41 (KOENIGSBERG) |
| 11        | GE 11         | 52 (LIEGE) |
| 12        | GE 12         | 53 (BRUSSELS) |
| 13        | GE 14         | 54 (ANTWERP) |
| 14        | GE 17         | 255 (off-board, enters later) |
| 15        | GE 18         | 255 (off-board) |
| 16        | AH 1          | 80 (CRACOW) |
| 17        | AH 2          | 81 (LEMBERG) |
| 18        | AH 3          | 82 (CZERNOWITZ) |
| 19        | AH 4          | 83 (BUDAPEST) |
| 20        | AH 5          | 84 (VIENNA) |
| 21        | AH 6          | 85 (LINZ) |
| 22        | AH 7          | 86 (GRAZ) |
| 23        | AH 10         | 87 (TRIESTE) |
| 24        | AH 11         | 255 (off-board) |
| 30        | TU YLD        | 140 (SMYRNA) |
| 31        | TU AoI        | 141 (ISTANBUL) |
| 32        | FR 1          | 20 (CHATEAU_THIERRY) |
| 33        | FR 2          | 21 (EPERNAY) |
| 34        | FR 3          | 30 (SEDAN) |
| 35        | FR 4          | 34 (VERDUN) |
| 36        | FR 5          | 35 (NANCY) |
| 37        | FR 6          | 36 (BELFORT) |
| 38        | FR 7          | 28 (DIJON) |
| 39        | FR 9          | 15 (PARIS) |
| 40        | FR 10         | 255 (off-board) |
| 41        | FR Orient     | 255 (off-board) |
| 42        | BR 1          | 255 (off-board) |
| 43        | BR 2          | 255 (off-board) |
| 44        | BR 3          | 255 (off-board) |
| 45        | BR 4          | 255 (off-board) |
| 46        | BR 5          | 255 (off-board) |
| 47        | BR BEF        | 53 (BRUSSELS) |
| 48        | BR NE         | 255 (off-board) |
| 52        | RU 1          | 90 (WARSAW) |
| 53        | RU 2          | 91 (LODZ) |
| 54        | RU 3          | 92 (VILNA) |
| 55        | RU 4          | 93 (KOVNO) |
| 56        | RU 5          | 94 (DVINSK) |
| 57        | RU 6          | 95 (MINSK) |
| 58        | RU 7          | 96 (BREST_LITOVSK) |
| 59        | RU 8          | 97 (KOVEL) |
| 60        | RU 9          | 98 (LUTSK) |
| 61        | RU 10         | 99 (ROVNO) |
| 62        | RU 11         | 100 (TARNOPOL) |
| 63        | RU 12         | 101 (ODESSA) |
| 64        | RU CAU        | 255 (off-board, Caucasus) |
| 65        | BE 1          | 56 (GHENT) |
| 66        | SB 1          | 110 (BELGRADE) |
| 67        | SB 2          | 111 (NISH) |

**Important:** The RTT space IDs above may not all match `data/rtt_space_map.json`
(67 of 72 spaces are mapped). For unmapped spaces, use OFFBOARD as fallback.
Use `data/rtt_space_map.json` to convert RTT IDs → our str_ids, then
`pog_map_graph.json` space list to get our integer indices.

Corps pieces (indices 68–193): All start **off-board** in Historical scenario.
They are reinforcements placed during the game via SR/reinforcement rules.

### Part 3 — Patch `jax_env.py`

After writing `starting_positions.py`, update `jax_env.py`:

```python
# Replace these module-level lines:
UNIT_FACTION = jnp.full((MAX_UNITS,), -1, dtype=jnp.int8)
UNIT_TYPE    = jnp.full((MAX_UNITS,), UNIT_INF, dtype=jnp.int8)

# With:
from src.data.starting_positions import UNIT_FACTION_INIT, UNIT_TYPE_INIT
UNIT_FACTION = UNIT_FACTION_INIT
UNIT_TYPE    = UNIT_TYPE_INIT

# And update jax_reset() to use:
from src.data.starting_positions import INITIAL_UNIT_LOC, UNIT_STRENGTH_INIT
# Replace:
#   unit_loc=jnp.full((MAX_UNITS,), OFFBOARD, dtype=jnp.uint8),
#   unit_strength=jnp.zeros((MAX_UNITS,), dtype=jnp.int8),
# With:
#   unit_loc=INITIAL_UNIT_LOC,
#   unit_strength=UNIT_STRENGTH_INIT,
```

### Acceptance criteria

- `from src.data.starting_positions import UNIT_FACTION_INIT` works ✓
- `UNIT_FACTION_INIT[1]` == FACTION_CP (GE 1 is CP) ✓
- `UNIT_FACTION_INIT[32]` == FACTION_AP (FR 1 is AP) ✓
- `UNIT_TYPE_INIT[1]` == UNIT_ARMY ✓
- `jax_legal_mask(jax_reset(rng_key))` returns at least some MOVE_UNIT actions True
  (proves units are on the board and movement is legal) ✓
- Write a test: `tests/test_starting_positions.py`

### Note on corps starting positions

In the real Historical scenario, corps are off-board at game start and placed
via reinforcement rules during play. Keeping them all off-board (OFFBOARD=255)
is **correct** for game accuracy. Do NOT place corps on random spaces.

---

## Reminder: BC training is running on cluster

Once `checkpoints/bc/epoch_010.pkl` arrives, run:
```bash
# Cluster
python train_selfplay.py --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 256 --iterations 100 --checkpoint-dir checkpoints/selfplay

# Local RTX 5060
python train_selfplay.py --bc-checkpoint checkpoints/bc/epoch_010.pkl \
  --n-actors 16 --buffer-capacity 50000 --batch-size 256 \
  --iterations 100 --checkpoint-dir checkpoints/selfplay
```
