# RTT Data Pipeline — Guide for LLM Agents & Humans

> **If you are an LLM agent asked to process game files: read §0 and §1 only — that is everything you need.**

---

## §0 — Directory contract (where files live)

```
PoGAIV1/                          ← project root, run all commands from here
│
├── data/
│   ├── rtt_games/                ← PUT RAW RTT JSON FILES HERE (input)
│   │     176409.json               one file per game, any filename
│   │     177001.json
│   │     ...
│   │
│   ├── training/                 ← WRITE EXTRACTED JSONL HERE (output)
│   │     expert_games.jsonl        one line per training step
│   │
│   ├── rtt_space_map.json        ← RTT space-id → our space-id mapping (DO NOT EDIT)
│   └── data.js                   ← RTT server source data (DO NOT EDIT)
│
├── pog_map_graph.json            ← board map (DO NOT EDIT)
├── pog_cards_db.json             ← card database (DO NOT EDIT)
└── src/data/rtt_parser.py        ← parser (import from here)
```

Both `data/rtt_games/` and `data/training/` are in `.gitignore` — large files, never commit.

---

## §1 — How to extract training data (three ways)

### Option A — Python, one game at a time (simplest)

```python
from src.data.rtt_parser import extract_training_records

records = extract_training_records("data/rtt_games/176409.json")
# records → List[Dict], ready to feed into bc_pipeline
# typical game: ~195 records
```

### Option B — Python, whole directory at once (recommended)

```python
from src.data.rtt_parser import convert_rtt_directory

n = convert_rtt_directory(
    input_dir      = "data/rtt_games/",
    output_path    = "data/training/expert_games.jsonl",
    space_map_json = "data/rtt_space_map.json",
)
print(f"{n} training records written")
```

### Option C — Command line

```bash
python -m src.data.rtt_parser \
    --input     data/rtt_games/ \
    --output    data/training/expert_games.jsonl \
    --map       pog_map_graph.json \
    --cards     pog_cards_db.json \
    --space_map data/rtt_space_map.json
```

### Verify the output

```bash
# How many records?
wc -l data/training/expert_games.jsonl

# Action type breakdown + game count
python3 -c "
import json
from collections import Counter
records = [json.loads(l) for l in open('data/training/expert_games.jsonl')]
c = Counter()
for r in records:
    a = r['action_taken']
    if   a == 0:   c['PASS']      += 1
    elif a < 111:  c['EVENT']     += 1
    elif a < 441:  c['OPS']       += 1
    else:          c['MOVE_UNIT'] += 1
games = len(set(r['game_id'] for r in records))
print(f'{len(records)} records from {games} games')
print(dict(c))
"
```

**Training data targets:**

| Records | Games | Quality |
|---|---|---|
| 5,000 | ~25 | Smoke test only |
| 20,000 | ~100 | Minimum viable BC |
| 50,000–100,000 | ~250–500 | Recommended |
| 150,000+ | ~750+ | Excellent |

---

## §2 — Output record schema

Each record returned by `extract_training_records()` or written to `expert_games.jsonl`
has exactly these fields, compatible with `bc_pipeline.make_bc_batches()`:

```json
{
  "game_id":      "176409",
  "step":         0,
  "turn":         1,
  "action_round": 1,
  "player":       "CP",
  "obs_tensor":   [[...32 rows × 72 cols, float32...]],
  "card_context": [[...7 rows × 16 cols, float32...]],
  "legal_mask":   [...5341 bools...],
  "action_taken": 72,
  "outcome":      -1
}
```

| Field | Type / Shape | Meaning |
|---|---|---|
| `game_id` | str | filename stem of source game |
| `step` | int | sequential index within the game |
| `turn` | int | game turn (1–20) |
| `action_round` | int | action round within the turn |
| `player` | `"AP"` or `"CP"` | who is acting |
| `obs_tensor` | (32, 72) float32 | board state — 32 feature planes × 72 spaces |
| `card_context` | (7, 16) float32 | active player's hand encoding |
| `legal_mask` | (5341,) bool | which of the 5341 actions are legal |
| `action_taken` | int [0, 5341) | flat action index the human expert chose |
| `outcome` | +1 / −1 / 0 | result **from this player's perspective** |

### Action space layout

```
index 0           → PASS
index 1–110       → PLAY_AS_EVENT   card_idx = flat − 1
index 111–440     → PLAY_AS_OPS     card_idx = (flat−111) // 3
                                    op_type  = (flat−111) % 3
                                               0=MOVE  1=ATTACK  2=SR
index 441–5340    → MOVE_UNIT       src_idx  = (flat−441) // 72
                                    tgt_idx  = (flat−441) % 72
```

---

## §3 — What is an RTT game file?

RTT (Rally the Troops) is the online server for Paths of Glory. Each exported game
is a JSON with three sections:

| Key | Content |
|---|---|
| `players` | `[{"role": "Allied Powers", "name": ...}, {"role": "Central Powers", "name": ...}]` |
| `state` | Final game state — `.turn`, `.vp`, `.location` (piece index → RTT space id) |
| `replay` | Full action log — 3000+ entries of `[role, action, *args]` |

The parser reads `replay`, groups entries by `end_action` boundaries,
identifies the strategic decision in each group, and emits one record per decision.

---

## §4 — Card ID mapping (RTT integer → our str_id)

RTT encodes cards as integers in `play_event`, `play_ops`, `play_sr`, `play_rps`.

| RTT integer | Our str_id | Formula |
|---|---|---|
| 1–65 | AP-01 … AP-65 | `f"AP-{n:02d}"` |
| 66–130 | CP-01 … CP-65 | `f"CP-{n-65:02d}"` |

Examples confirmed from `data/data.js`:
- RTT `3` → `AP-03` = "Russian Reinforcements"
- RTT `66` → `CP-01` = "Guns of August"

---

## §5 — Space ID mapping (RTT integer → our space index)

RTT uses integers 1–283 for board spaces. Our AI uses 72 named spaces (index 0–71).
The mapping is stored in `data/rtt_space_map.json` — **do not regenerate, use as-is**.

67 of our 72 spaces are mapped. 5 spaces have no RTT equivalent
(NAMUR, COLOGNE, FREIBURG, GORIZIA, SUEZ) — piece moves through them are skipped.

### Full mapping table

| RTT ID | RTT Name | Our str_id | Our idx | Nation | VP |
|---|---|---|---|---|---|
| 12 | Rouen | ROUEN | 9 | fr | 0 |
| 15 | Paris | PARIS | 0 | fr | 5 |
| 16 | Amiens | AMIENS | 1 | fr | 0 |
| 17 | Calais | CALAIS | 10 | fr | 1 |
| 18 | Ostend | OSTEND | 15 | be | 0 |
| 19 | Cambrai | CAMBRAI | 11 | fr | 0 |
| 20 | Chateau Thierry | CHATEAU_THIERRY | 8 | fr | 0 |
| 28 | Dijon | DIJON | 7 | fr | 0 |
| 29 | Bar le Duc | BAR_LE_DUC | 6 | fr | 0 |
| 30 | Sedan | SEDAN | 2 | fr | 1 |
| 31 | Brussels | BRUSSELS | 13 | be | 1 |
| 32 | Antwerp | ANTWERP | 14 | be | 1 |
| 33 | Liege | LIEGE | 16 | be | 1 |
| 34 | Verdun | VERDUN | 3 | fr | 2 |
| 35 | Nancy | NANCY | 4 | fr | 0 |
| 36 | Belfort | BELFORT | 5 | fr | 1 |
| 38 | Mulhouse | MULHOUSE | 12 | ge | 0 |
| 39 | Strasbourg | STRASBOURG | 19 | ge | 1 |
| 40 | Metz | METZ | 18 | ge | 1 |
| 41 | Koblenz *(RTT spelling)* | COBLENZ | 20 | ge | 0 |
| 42 | Aachen | AACHEN | 22 | ge | 0 |
| 57 | Trent | TRENT | 70 | ah | 1 |
| 58 | Milan | MILAN | 71 | it | 0 |
| 71 | Venice | VENICE | 67 | it | 1 |
| 79 | Berlin | BERLIN | 24 | ge | 5 |
| 88 | Trieste | TRIESTE | 68 | ah | 2 |
| 90 | Vienna | VIENNA | 39 | ah | 5 |
| 94 | Breslau | BRESLAU | 27 | ge | 1 |
| 98 | Danzig | DANZIG | 26 | ge | 1 |
| 99 | Konigsberg | KONIGSBERG | 25 | ge | 2 |
| 102 | Lodz | LODZ | 30 | ru | 1 |
| 104 | Cracow | CRACOW | 37 | ah | 1 |
| 106 | Budapest | BUDAPEST | 40 | ah | 3 |
| 117 | Salonika | SALONIKA | 49 | gr | 2 |
| 118 | Monastir | MONASTIR | 51 | sb | 1 |
| 122 | Nis *(RTT spelling)* | NISH | 50 | sb | 0 |
| 124 | Sarajevo | SARAJEVO | 46 | ah | 1 |
| 125 | Belgrade | BELGRADE | 45 | sb | 2 |
| 134 | Warsaw | WARSAW | 28 | ru | 3 |
| 140 | Riga | RIGA | 32 | ru | 2 |
| 143 | Petrograd | PETROGRAD | 35 | ru | 5 |
| 148 | Vilna | VILNA | 31 | ru | 1 |
| 152 | Moscow | MOSCOW | 36 | ru | 5 |
| 157 | Minsk | MINSK | 33 | ru | 1 |
| 163 | Brest Litovsk | BREST_LITOVSK | 29 | ru | 2 |
| 165 | Lublin | LUBLIN | 43 | ru | 0 |
| 166 | Kovel *(RTT spelling)* | KOWEL | 44 | ru | 0 |
| 171 | Kiev | KIEV | 34 | ru | 2 |
| 176 | Przemysl | PRZEMYSL | 38 | ah | 1 |
| 177 | Lemberg | LEMBERG | 41 | ah | 1 |
| 182 | Czernowitz | CZERNOWITZ | 42 | ah | 0 |
| 198 | Sofia | SOFIA | 47 | bu | 2 |
| 204 | Bucharest | BUCHAREST | 48 | ro | 3 |
| 212 | Gallipoli | GALLIPOLI | 53 | tu | 0 |
| 215 | Izmir *(RTT spelling)* | SMYRNA | 54 | tu | 1 |
| 219 | Constantinople | CONSTANTINOPLE | 52 | tu | 5 |
| 224 | Ankara | ANKARA | 55 | tu | 1 |
| 241 | Erzerum *(RTT spelling)* | ERZURUM | 56 | tu | 2 |
| 251 | Aleppo | ALEPPO | 57 | tu | 1 |
| 254 | Mosul | MOSUL | 66 | tu | 1 |
| 258 | Damascus | DAMASCUS | 58 | tu | 1 |
| 263 | Baghdad | BAGHDAD | 63 | tu | 2 |
| 264 | Kut | KUT | 64 | tu | 0 |
| 269 | Basra | BASRA | 65 | pe | 1 |
| 274 | Beersheba | BEERSHEBA | 60 | tu | 0 |
| 275 | Jerusalem | JERUSALEM | 59 | tu | 3 |
| 278 | Cairo | CAIRO | 62 | eg | 0 |

RTT IDs 282–361 are off-board boxes (reserve, eliminated, reinforcements) — ignored.

---

## §6 — How the parser works internally

```
RTT JSON
  │
  ├─ extract_outcome()           scan for [loser, ".resign", winner] → +1/-1/0
  │
  └─ group_replay_by_action()    split at "end_action"; undo rolls back prior entry
         │
         └─ per group:
               _extract_primary_action()   play_event/ops/sr/rps → flat index
                                           "next" with no card play → PASS (0)
                                           card plays always win over "next"
               PogEnv.snapshot()           obs_tensor, card_context, legal_mask
               emit record
               if activate_move in group:
                   _extract_move_unit_records()  piece→space → MOVE_UNIT flat idx
                   emit additional records
```

---

## §7 — Known limitations

| Limitation | Effect |
|---|---|
| `obs_tensor` comes from our env (not RTT's) | Board state is approximate — good enough for BC pre-training |
| `piece_locations` starts from final game state | MOVE_UNIT source spaces may be wrong for early moves |
| 5 spaces absent from RTT | ~5% of MOVE_UNIT records not generated |
| Only completed games have `.resign` | Incomplete games get `outcome=0` |

The most valuable signal — **which card, event vs ops** — is always correct.
