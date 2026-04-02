# PoG-AI Simplified Rulebook

This rulebook describes the **exact rules implemented in the PoG-AI simulation** (`pog_env.py` / `jax_env.py`). It is not the real Paths of Glory rulebook. The AI is trained and plays within these rules only.

---

## 1. Overview

Two players: **AP** (Allied Powers) and **CP** (Central Powers).  
Goal: control VP spaces to reach a VP advantage, or survive to the end.  
The game is card-driven: every action costs a card from your hand.

---

## 2. Victory Points (VP)

- VP is a **single integer**: positive = AP winning, negative = CP winning.
- VP is computed at all times as:  
  `VP = sum(vp_value of AP-controlled spaces) − sum(vp_value of CP-controlled spaces)`
- Starting VP (from initial control) is approximately **−1** (CP slight advantage).

**Game ends immediately when:**
- `|VP| ≥ 20` → side with positive VP wins
- `Turn > 20` → side with positive VP wins; 0 VP = draw

**VP-changing spaces** (the spaces that matter most):

| Space | VP | Nation | Terrain |
|---|---|---|---|
| Paris | 5 | FR | Fort |
| Berlin | 5 | GE | Fort |
| Vienna | 5 | AH | Fort |
| Petrograd | 5 | RU | Fort |
| Moscow | 5 | RU | Fort |
| Constantinople | 5 | TU | Fort |
| Warsaw | 3 | RU | Fort |
| Budapest | 3 | AH | Fort |
| Bucharest | 3 | RO | Clear |
| Jerusalem | 3 | TU | Fort |
| Baghdad | 2 | TU | Desert |
| Verdun | 2 | FR | Fort |
| Erzurum | 2 | TU | Mountain |
| Riga | 2 | RU | Fort |
| Kiev | 2 | RU | Clear |
| Konigsberg | 2 | GE | Fort |
| Belrade | 2 | SE | Fort |
| Sofia | 2 | BU | Clear |
| Salonika | 2 | GR | Clear |
| Trieste | 2 | AH | Fort |
| + many 1-VP spaces | 1 | various | various |

---

## 3. Map — 72 Spaces

### 3.1 Terrain Types

| Terrain | Effect on Combat |
|---|---|
| Clear | No modifier |
| Fort | Defender gets fortress benefit (see §8) |
| Mountain | No special effect in current sim |
| Desert | No special effect in current sim |
| Sea | Not enterable |

### 3.2 Nation Control (Starting)

| Nation Code | Faction | Spaces |
|---|---|---|
| FR | AP | Paris, Amiens, Sedan, Verdun, Nancy, Belfort, Bar-le-Duc, Dijon, Chateau-Thierry, Rouen, Calais, Cambrai, Mulhouse |
| BE | AP | Brussels, Antwerp, Ostend, Liege, Namur |
| RU | AP | Warsaw, Brest-Litovsk, Lodz, Vilna, Riga, Minsk, Kiev, Petrograd, Moscow, Lublin, Kowel |
| IT | AP | Venice, Milan |
| BR | AP | Suez, Cairo |
| GR | AP | Salonika |
| RO | AP | Bucharest |
| SE | AP | Belgrade, Nish, Monastir |
| GE | CP | Metz, Strasbourg, Coblenz, Freiburg, Aachen, Cologne, Berlin, Konigsberg, Danzig, Breslau |
| AH | CP | Cracow, Przemysl, Vienna, Budapest, Lemberg, Czernowitz, Trieste, Gorizia, Trent, Sarajevo |
| TU | CP | Constantinople, Gallipoli, Smyrna, Ankara, Erzurum, Aleppo, Damascus, Jerusalem, Beersheba, Baghdad, Kut, Basra, Mosul |
| BU | CP | Sofia |

### 3.3 Full Space List & Connections

```
THEATRE: WESTERN FRONT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[00] PARIS         (FR, Fort, VP5)  ── AMIENS, SEDAN, CHATEAU_THIERRY, DIJON
[01] AMIENS        (FR, Clear, VP0) ── PARIS, CALAIS, SEDAN, CAMBRAI, ROUEN
[02] SEDAN         (FR, Clear, VP1) ── AMIENS, PARIS, VERDUN, CAMBRAI, LIEGE
[03] VERDUN        (FR, Fort, VP2)  ── SEDAN, NANCY, BAR_LE_DUC, METZ
[04] NANCY         (FR, Clear, VP0) ── VERDUN, BELFORT, BAR_LE_DUC, STRASBOURG, METZ
[05] BELFORT       (FR, Fort, VP1)  ── NANCY, DIJON, MULHOUSE
[06] BAR_LE_DUC    (FR, Clear, VP0) ── VERDUN, NANCY, CHATEAU_THIERRY
[07] DIJON         (FR, Clear, VP0) ── PARIS, BELFORT, BAR_LE_DUC
[08] CHATEAU_THIERRY (FR, Clear,VP0)── PARIS, BAR_LE_DUC
[09] ROUEN         (FR, Clear, VP0) ── AMIENS, CALAIS
[10] CALAIS        (FR, Clear, VP1) ── AMIENS, ROUEN, OSTEND
[11] CAMBRAI       (FR, Clear, VP0) ── AMIENS, SEDAN, BRUSSELS
[12] MULHOUSE      (FR, Clear, VP0) ── BELFORT, STRASBOURG, FREIBURG

THEATRE: BELGIUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[13] BRUSSELS      (BE, Clear, VP1) ── CAMBRAI, ANTWERP, LIEGE, OSTEND, NAMUR
[14] ANTWERP       (BE, Fort, VP1)  ── BRUSSELS, OSTEND
[15] OSTEND        (BE, Clear, VP0) ── CALAIS, BRUSSELS, ANTWERP
[16] LIEGE         (BE, Fort, VP1)  ── BRUSSELS, SEDAN, AACHEN, NAMUR
[17] NAMUR         (BE, Fort, VP0)  ── BRUSSELS, LIEGE

THEATRE: GERMANY (WEST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[18] METZ          (GE, Fort, VP1)  ── VERDUN, NANCY, STRASBOURG, COBLENZ
[19] STRASBOURG    (GE, Fort, VP1)  ── NANCY, METZ, MULHOUSE, FREIBURG
[20] COBLENZ       (GE, Clear, VP0) ── METZ, AACHEN, COLOGNE, BERLIN
[21] FREIBURG      (GE, Clear, VP0) ── STRASBOURG, MULHOUSE
[22] AACHEN        (GE, Clear, VP0) ── LIEGE, COBLENZ, COLOGNE
[23] COLOGNE       (GE, Clear, VP0) ── AACHEN, COBLENZ, BERLIN
[24] BERLIN        (GE, Fort, VP5)  ── COBLENZ, COLOGNE, DANZIG, BRESLAU

THEATRE: GERMANY (EAST) / POLAND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[25] KONIGSBERG    (GE, Fort, VP2)  ── DANZIG, RIGA
[26] DANZIG        (GE, Clear, VP1) ── BERLIN, KONIGSBERG, LODZ
[27] BRESLAU       (GE, Clear, VP1) ── BERLIN, CRACOW, LODZ
[28] WARSAW        (RU, Fort, VP3)  ── LODZ, BREST_LITOVSK, LUBLIN, CRACOW
[29] BREST_LITOVSK (RU, Fort, VP2)  ── WARSAW, KOWEL, MINSK, LUBLIN
[30] LODZ          (RU, Clear, VP1) ── WARSAW, DANZIG, BRESLAU, LUBLIN
[31] VILNA         (RU, Clear, VP1) ── RIGA, MINSK, BREST_LITOVSK
[43] LUBLIN        (RU, Clear, VP0) ── WARSAW, BREST_LITOVSK, LEMBERG, LODZ
[44] KOWEL         (RU, Clear, VP0) ── BREST_LITOVSK, LUBLIN, LEMBERG, KIEV

THEATRE: RUSSIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[32] RIGA          (RU, Fort, VP2)  ── VILNA, KONIGSBERG, PETROGRAD
[33] MINSK         (RU, Clear, VP1) ── BREST_LITOVSK, VILNA, MOSCOW
[34] KIEV          (RU, Clear, VP2) ── KOWEL, CZERNOWITZ, MINSK
[35] PETROGRAD     (RU, Fort, VP5)  ── RIGA, MOSCOW
[36] MOSCOW        (RU, Fort, VP5)  ── PETROGRAD, MINSK, KIEV

THEATRE: AUSTRIA-HUNGARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[37] CRACOW        (AH, Fort, VP1)  ── BRESLAU, WARSAW, PRZEMYSL, VIENNA, LEMBERG
[38] PRZEMYSL      (AH, Fort, VP1)  ── CRACOW, LEMBERG, CZERNOWITZ
[39] VIENNA        (AH, Fort, VP5)  ── CRACOW, BUDAPEST, TRIESTE
[40] BUDAPEST      (AH, Fort, VP3)  ── VIENNA, BELGRADE, SARAJEVO, BUCHAREST, LEMBERG
[41] LEMBERG       (AH, Clear, VP1) ── CRACOW, PRZEMYSL, CZERNOWITZ, LUBLIN, KOWEL, BUDAPEST
[42] CZERNOWITZ    (AH, Clear, VP0) ── LEMBERG, PRZEMYSL, KIEV, BUCHAREST

THEATRE: ITALY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[67] VENICE        (IT, Clear, VP1) ── TRIESTE, GORIZIA, MILAN
[68] TRIESTE       (AH, Fort, VP2)  ── VENICE, GORIZIA, VIENNA
[69] GORIZIA       (AH, Mountain,VP0)──VENICE, TRIESTE, TRENT
[70] TRENT         (AH, Mountain,VP1)──GORIZIA, MILAN, VIENNA
[71] MILAN         (IT, Clear, VP0) ── VENICE, TRENT

THEATRE: BALKANS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[45] BELGRADE      (SE, Fort, VP2)  ── BUDAPEST, SARAJEVO, NISH
[46] SARAJEVO      (AH, Mountain,VP1)──BELGRADE, NISH, MONASTIR, BUDAPEST
[47] SOFIA         (BU, Clear, VP2) ── NISH, SALONIKA, BUCHAREST, CONSTANTINOPLE
[48] BUCHAREST     (RO, Clear, VP3) ── CZERNOWITZ, SOFIA, BUDAPEST
[49] SALONIKA      (GR, Clear, VP2) ── SOFIA, MONASTIR
[50] NISH          (SE, Clear, VP0) ── BELGRADE, SARAJEVO, SOFIA, MONASTIR
[51] MONASTIR      (SE, Clear, VP1) ── SARAJEVO, NISH, SALONIKA

THEATRE: TURKEY / MIDDLE EAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[52] CONSTANTINOPLE(TU, Fort, VP5)  ── GALLIPOLI, SMYRNA, ANKARA, SOFIA
[53] GALLIPOLI     (TU, Clear, VP0) ── CONSTANTINOPLE, SMYRNA
[54] SMYRNA        (TU, Clear, VP1) ── CONSTANTINOPLE, GALLIPOLI, ANKARA
[55] ANKARA        (TU, Clear, VP1) ── CONSTANTINOPLE, SMYRNA, ERZURUM, ALEPPO
[56] ERZURUM       (TU, Mountain,VP2)──ANKARA, ALEPPO
[57] ALEPPO        (TU, Desert, VP1) ── ANKARA, ERZURUM, DAMASCUS, MOSUL
[58] DAMASCUS      (TU, Desert, VP1) ── ALEPPO, JERUSALEM
[59] JERUSALEM     (TU, Fort, VP3)  ── DAMASCUS, BEERSHEBA
[60] BEERSHEBA     (TU, Desert, VP0) ── JERUSALEM, SUEZ
[61] SUEZ          (BR, Clear, VP1) ── BEERSHEBA, CAIRO
[62] CAIRO         (BR, Clear, VP0) ── SUEZ
[63] BAGHDAD       (TU, Desert, VP2) ── MOSUL, KUT
[64] KUT           (TU, Desert, VP0) ── BAGHDAD, BASRA
[65] BASRA         (TU, Desert, VP1) ── KUT
[66] MOSUL         (TU, Desert, VP1) ── ALEPPO, BAGHDAD
```

### 3.4 Topology Graph (by Theatre)

```
WESTERN FRONT
─────────────────────────────────────────────────────────
ROUEN ── CALAIS ── OSTEND ── ANTWERP
  |           |       |          |
AMIENS ── CAMBRAI ── BRUSSELS ── LIEGE ── AACHEN
  |      /    |          |          |        |
PARIS  SEDAN  |         NAMUR     SEDAN    COLOGNE
  |      |    |                             |
DIJON  VERDUN─METZ ── STRASBOURG ── FREIBURG COBLENZ
  |      |  \    \         |            |      |
BELFORT BAR  NANCY MULHOUSE ────────────┘    BERLIN
  |      LE   |
MULHOUSE DUC  BAR_LE_DUC ── CHATEAU_THIERRY ── PARIS
```

```
EASTERN FRONT (simplified)
─────────────────────────────────────────────────────────
PETROGRAD ── RIGA ── KONIGSBERG ── DANZIG ── BERLIN
    |           |                     |         |
  MOSCOW     VILNA                  LODZ     BRESLAU
    |           |                  /    \       |
   KIEV       MINSK ── BREST ── WARSAW  LUBLIN CRACOW
    |           |       LITOVSK    |      |      |
  KOWEL      LUBLIN      |       CRACOW  LEMBERG |
    |                  KOWEL       |      |    VIENNA
  LEMBERG ──────────────┘        ...   PRZEMYSL  |
                                              BUDAPEST
```

---

## 4. Turn Structure

```
TURN 1..20
  └─ ACTION ROUND 1..7   (alternating AP → CP → AP → ...)
       └─ Active player plays 1 card → takes 1 action
  └─ End of Turn 7:
       • Both played piles reshuffled into decks
       • Both players draw up to 7 cards
       • War status check: if VP ≥ +5, AP enters Total War; if VP ≤ −5, CP enters Total War
       • Turn counter +1
```

**Starting hand sizes:**
- AP: 6 cards at game start, 7 cards after turn 1
- CP: 7 cards from game start

**Hand cap:** 7 cards maximum.

---

## 5. War Status

Each faction has an independent war status: **Limited War** or **Total War**.

| Faction | Condition to enter Total War |
|---|---|
| AP | VP ≥ +5 at end of any turn |
| CP | VP ≤ −5 at end of any turn |

- Transition is **one-way** (cannot revert to Limited War).
- Starting VP is approximately −1, so neither side starts in Total War.
- Cards gated to **Limited War only** cannot be played once your faction is in Total War.
- Cards gated to **Total War only** cannot be played until your faction is in Total War.
- Cards marked **Either** can be played in any phase.

---

## 6. Cards

### 6.1 Card Structure

Each card has:
- **ID**: unique identifier (e.g. AP-01, CP-09)
- **Faction**: AP or CP — only that faction can hold/play this card
- **OPS**: operations value (2–5) — determines action strength (see §7)
- **SR**: strategic redeployment value (2–5)
- **Phase**: Limited War / Total War / Either
- **Combat Card**: yes/no — required to attack into Level-3 trench
- **Discard**: PERMANENT (removed from game after event) or REUSE (returns to deck)

### 6.2 AP Cards (71 total, 65 unique)

| ID | Name | OPS | SR | Phase | Discard | Combat |
|---|---|---|---|---|---|---|
| AP-01 | British Reinforcements | 4 | 4 | Limited | PERM | |
| AP-02 | Blockade | 4 | 4 | Limited | PERM | |
| AP-03 | Russian Reinforcements | 3 | 4 | Limited | PERM | |
| AP-04 | Pleve | 2 | 2 | Limited | REUSE | ✓ |
| AP-05 | Putnik | 2 | 2 | Limited | PERM | ✓ |
| AP-06 | Withdrawal | 2 | 2 | Either | REUSE | ✓ |
| AP-07 | Severe Weather | 2 | 2 | Either | REUSE | ✓ |
| AP-08 | Russian Reinforcements | 2 | 2 | Limited | PERM | |
| AP-09 | Moltke | 3 | 4 | Limited | PERM | |
| AP-10 | French Reinforcements | 3 | 4 | Limited | PERM | |
| AP-11 | Russian Reinforcements | 3 | 4 | Limited | PERM | |
| AP-12 | Entrench | 3 | 4 | Either | PERM | |
| AP-13 | Rape of Belgium | 4 | 4 | Limited | PERM | |
| AP-14 | British Reinforcements | 4 | 4 | Limited | PERM | |
| AP-15 | British Reinforcements | 3 | 4 | Limited | PERM | |
| AP-16 | Romania | 5 | 5 | Limited | PERM | |
| AP-17 | Italy | 5 | 5 | Limited | PERM | |
| AP-18 | Hurricane Barrage | 2 | 2 | Limited | REUSE | ✓ |
| AP-19 | Air Superiority | 2 | 2 | Limited | REUSE | ✓ |
| AP-20 | British Reinforcements | 2 | 2 | Limited | PERM | |
| AP-21 | Phosgene Gas | 2 | 2 | Limited | REUSE | ✓ |
| AP-22 | Italian Reinforcements | 3 | 4 | Limited | PERM | |
| AP-23 | Cloak and Dagger | 2 | 2 | Limited | PERM | |
| AP-24 | French Reinforcements | 3 | 4 | Limited | PERM | |
| AP-25 | Russian Reinforcements | 3 | 4 | Limited | PERM | |
| AP-26 | Lusitania | 3 | 4 | Limited | PERM | |
| AP-27 | Great Retreat | 3 | 4 | Limited | PERM | |
| AP-28 | Landships | 4 | 4 | Limited | PERM | |
| AP-29 | Yudenitch | 4 | 4 | Limited | PERM | |
| AP-30 | Salonika | 4 | 4 | Limited | PERM | |
| AP-31 | MEF | 4 | 4 | Limited | PERM | |
| AP-32 | Russian Reinforcements | 2 | 2 | Limited | PERM | |
| AP-33 | Grand Fleet | 2 | 2 | Limited | PERM | |
| AP-34 | British Reinforcements | 3 | 4 | Limited | PERM | |
| AP-35 | Yanks and Tanks | 4 | 4 | Total | PERM | |
| AP-36 | Mine Attack | 2 | 2 | Total | PERM | ✓ |
| AP-37 | Independent Air Force | 2 | 2 | Total | PERM | |
| AP-38 | USA Reinforcements | 2 | 2 | Total | PERM | |
| AP-39 | They Shall Not Pass | 2 | 2 | Total | REUSE | ✓ |
| AP-40 | 14 Points | 2 | 2 | Total | PERM | |
| AP-41 | Arab Northern Army | 3 | 4 | Total | PERM | |
| AP-42 | British Reinforcements | 3 | 4 | Total | PERM | |
| AP-43 | USA Reinforcements | 3 | 4 | Total | PERM | |
| AP-44 | Greece | 3 | 4 | Total | PERM | |
| AP-45 | Kerensky Offensive | 3 | 4 | Total | PERM | |
| AP-46 | Brusilov Offensive | 4 | 4 | Total | PERM | |
| AP-47 | USA Reinforcements | 4 | 4 | Total | PERM | |
| AP-48 | Royal Tank Corps | 4 | 4 | Total | PERM | ✓ |
| AP-49 | Sinai Pipeline | 4 | 4 | Total | PERM | |
| AP-50 | Allenby | 4 | 4 | Total | PERM | |
| AP-51 | Everyone Into Battle | 4 | 4 | Total | PERM | |
| AP-52 | Convoy | 4 | 4 | Total | PERM | |
| AP-53 | Army of the Orient | 5 | 5 | Total | PERM | |
| AP-54 | Zimmermann Telegram | 5 | 5 | Total | PERM | |
| AP-55 | Over There | 5 | 5 | Total | PERM | |
| AP-56 | Paris Taxis | 2 | 2 | Limited | PERM | |
| AP-57 | Russian Cavalry | 3 | 4 | Limited | PERM | |
| AP-58 | Russian Guards | 2 | 2 | Either | REUSE | ✓ |
| AP-59 | Alpine Troops | 2 | 2 | Either | REUSE | ✓ |
| AP-60 | Czech Legion | 3 | 4 | Total | PERM | |
| AP-61 | Maude | 4 | 4 | Total | REUSE | ✓ |
| AP-62 | The Sixtus Affair | 2 | 2 | Total | PERM | |
| AP-63 | Backs to the Wall | 3 | 4 | Total | REUSE | ✓ |
| AP-64 | USA Reinforcements | 3 | 4 | Total | PERM | |
| AP-65 | Influenza | 4 | 4 | Total | PERM | |

### 6.3 CP Cards (70 total, 65 unique)

| ID | Name | OPS | SR | Phase | Discard | Combat |
|---|---|---|---|---|---|---|
| CP-01 | Guns of August | 3 | 4 | Limited | PERM | |
| CP-02 | Wireless Intercepts | 2 | 2 | Limited | PERM | ✓ |
| CP-03 | Von Francois | 2 | 2 | Either | REUSE | ✓ |
| CP-04 | Severe Weather | 2 | 2 | Either | PERM | ✓ |
| CP-05 | Landwehr | 2 | 2 | Limited | PERM | |
| CP-06 | Entrench | 3 | 4 | Either | PERM | |
| CP-07 | German Reinforcements | 3 | 4 | Limited | PERM | |
| CP-08 | Race to the Sea | 3 | 4 | Limited | PERM | |
| CP-09 | Reichstag Truce | 4 | 4 | Limited | PERM | |
| CP-10 | Sud Army | 3 | 4 | Limited | PERM | |
| CP-11 | Oberost | 2 | 2 | Limited | PERM | |
| CP-12 | German Reinforcements | 4 | 4 | Limited | PERM | |
| CP-13 | Falkenhayn | 4 | 4 | Limited | PERM | |
| CP-14 | Austria-Hungary Reinforcements | 4 | 4 | Limited | PERM | |
| CP-15 | Chlorine Gas | 2 | 2 | Limited | REUSE | ✓ |
| CP-16 | Liman von Sanders | 2 | 2 | Limited | REUSE | ✓ |
| CP-17 | Mata Hari | 2 | 2 | Limited | PERM | |
| CP-18 | Fortified Machine Guns | 2 | 2 | Limited | REUSE | ✓ |
| CP-19 | Flamethrowers | 2 | 2 | Limited | REUSE | ✓ |
| CP-20 | Austria-Hungary Reinforcements | 3 | 4 | Limited | PERM | |
| CP-21 | German Reinforcements | 3 | 4 | Limited | PERM | |
| CP-22 | German Reinforcements | 3 | 4 | Limited | PERM | |
| CP-23 | Austria-Hungary Reinforcements | 3 | 4 | Limited | PERM | |
| CP-24 | Libyan Revolt | 3 | 4 | Limited | PERM | |
| CP-25 | High Seas Fleet | 4 | 4 | Limited | PERM | |
| CP-26 | Place of Execution | 4 | 4 | Limited | REUSE | ✓ |
| CP-27 | Zeppelin Raids | 4 | 4 | Limited | PERM | |
| CP-28 | Tsar Takes Command | 4 | 4 | Limited | PERM | |
| CP-29 | 11th Army | 2 | 2 | Limited | PERM | |
| CP-30 | Alpenkorps | 2 | 2 | Limited | REUSE | ✓ |
| CP-31 | Kemal | 3 | 4 | Limited | REUSE | ✓ |
| CP-32 | War In Africa | 3 | 4 | Limited | PERM | |
| CP-33 | Walter Rathenau | 5 | 5 | Limited | PERM | |
| CP-34 | Bulgaria | 5 | 5 | Limited | PERM | |
| CP-35 | Mustard Gas | 2 | 2 | Total | PERM | ✓ |
| CP-36 | U-Boats Unleashed | 2 | 2 | Total | PERM | |
| CP-37 | Hoffmann | 2 | 2 | Total | PERM | |
| CP-38 | German Reinforcements | 2 | 2 | Total | PERM | |
| CP-39 | German Reinforcements | 2 | 2 | Total | PERM | |
| CP-40 | Air Superiority | 3 | 4 | Total | REUSE | ✓ |
| CP-41 | German Reinforcements | 3 | 4 | Total | PERM | |
| CP-42 | Turkish Reinforcements | 3 | 4 | Total | PERM | |
| CP-43 | Von Below | 3 | 4 | Total | REUSE | ✓ |
| CP-44 | Von Hutier | 3 | 4 | Total | REUSE | ✓ |
| CP-45 | Treaty of Brest Litovsk | 4 | 4 | Total | PERM | |
| CP-46 | German Reinforcements | 4 | 4 | Total | PERM | |
| CP-47 | French Mutiny | 4 | 4 | Total | PERM | |
| CP-48 | Turkish Reinforcements | 4 | 4 | Total | PERM | |
| CP-49 | Michael | 4 | 4 | Total | REUSE | ✓ |
| CP-50 | Blucher | 4 | 4 | Total | REUSE | ✓ |
| CP-51 | Peace Offensive | 4 | 4 | Total | PERM | ✓ |
| CP-52 | Fall of the Tsar | 5 | 5 | Total | PERM | |
| CP-53 | Bolshevik Revolution | 5 | 5 | Total | PERM | |
| CP-54 | H-L Take Command | 5 | 5 | Total | PERM | |
| CP-55 | Lloyd George | 4 | 4 | Total | PERM | |
| CP-56 | Withdrawal | 2 | 2 | Either | REUSE | ✓ |
| CP-57 | Kaisertreu | 3 | 4 | Either | REUSE | ✓ |
| CP-58 | STAVKA Timidity | 2 | 2 | Total | REUSE | |
| CP-59 | Polish Restoration | 3 | 4 | Total | PERM | |
| CP-60 | Turk Determination | 3 | 4 | Either | REUSE | ✓ |
| CP-61 | Haig | 4 | 4 | Total | PERM | |
| CP-62 | Achtung: Panzer | 2 | 2 | Total | REUSE | ✓ |
| CP-63 | Russian Desertions | 3 | 4 | Total | PERM | |
| CP-64 | Alberich | 3 | 4 | Either | REUSE | ✓ |
| CP-65 | Prince Max | 4 | 4 | Total | PERM | |

### 6.4 Special Card: Reichstag Truce (CP-09)

The **only** card with an implemented VP effect:  
Playing CP-09 as an **Event** immediately adjusts VP by −1 (improving CP position by 1).

### 6.5 Reinforcement Cards (Event effect)

Cards whose name contains "reinforcement" (case-insensitive) bring **1 OFFBOARD unit** of the active faction onto the map when played as an Event. The unit appears at the first available friendly supply source space.

This applies to: AP-01, AP-03, AP-08, AP-11, AP-14, AP-15, AP-20, AP-22, AP-25, AP-32, AP-34, AP-38, AP-42, AP-43, AP-47, AP-64, CP-05, CP-07, CP-12, CP-14, CP-20, CP-21, CP-22, CP-23, CP-38, CP-39, CP-41, CP-42, CP-46, CP-48

### 6.6 Deck Composition Note

Some card IDs appear twice in the physical deck (e.g. AP-56 Paris Taxis appears ×2). The simulation includes all duplicate entries.

---

## 7. Actions

Each turn, the active player plays **one card** from their hand and chooses **one action type**:

### 7.1 PASS (Action 0)

- Discard no card (no card is consumed).
- Turn passes to opponent.

### 7.2 EVENT (Actions 1–110)

Play card index `k` as its event:
- Card is removed from hand.
- If card is **PERM**: removed from the game (goes to permanent discard).
- If card is **REUSE**: goes to the played pile and is reshuffled into the deck at turn end.
- **Event effect** (only two cards have mechanical effects):
  - "reinforcement" in name → bring 1 friendly OFFBOARD unit onto a supply source space
  - Reichstag Truce (CP-09) → VP −1

All other event texts are **flavour only** — no mechanical effect.

### 7.3 OPS — MOVE (Action 111 + card×3 + 0)

Play a card for its OPS value as Movement Operations:
- Card goes to played pile (REUSE — all OPS play is reusable).
- **1 unit** from any friendly space moves to any adjacent space.
- If destination is empty or friendly: unit occupies it, control changes to your faction.
- If destination has enemy units: **combat is resolved** (see §8).

> Note: OPS value on the card is not currently used to determine how many units move.  
> Every OPS-MOVE action moves exactly 1 unit.

### 7.4 OPS — ATTACK (Action 111 + card×3 + 1)

Play a card for Attack Operations:
- Card goes to played pile (REUSE).
- Legal only if you have at least one friendly unit adjacent to an enemy unit.
- **1 unit** attacks from its space into an adjacent enemy-occupied space.
- Combat resolved per §8.

### 7.5 OPS — SR (Strategic Redeployment) (Action 111 + card×3 + 2)

Play a card for Strategic Redeployment:
- Card goes to played pile (REUSE).
- Legal only if you have at least one friendly unit on-map.
- **1 OFFBOARD unit** of your faction is placed at the first available friendly supply source space.

> Note: Real SR moves units between rail-connected spaces. In this simulation, SR is equivalent to a reinforcement drop.

### 7.6 MOVE_UNIT (Actions 441–5340)

Encoded as `441 + src×72 + tgt`.  
Move one friendly unit from space `src` to adjacent space `tgt`:
- Does **not** cost a card — this action is separate from OPS.
- Only legal if the two spaces are connected on the map.
- If `tgt` has enemy units: combat is resolved (see §8).
- If `tgt` is empty or friendly: unit moves, control of `tgt` changes to your faction.

> Note: MOVE_UNIT and OPS-MOVE are functionally similar. The AI learns to use both.

---

## 8. Combat

Combat triggers when a unit moves into an enemy-occupied space.

### 8.1 Combat Resolution Table (CRT)

Compute the **attack ratio**: `attacker_strength / effective_defender_strength`

- `effective_defender_strength = max(1, defender_strength + trench_DRM)`
- `trench_DRM = min(trench_level, 2)`

Roll 1d6. Look up result:

| Ratio | Roll 1–3 | Roll 4–6 |
|---|---|---|
| < 0.5 | Engaged | Engaged |
| 0.5–1.0 | Engaged | A1 |
| 1.0–1.5 | A1 | EX |
| 1.5–2.0 | EX | D1 |
| 2.0–3.0 | D1 | D1R |
| 3.0–4.0 | D1R | D2 |
| ≥ 4.0 | D2 | DE |

**Results:**
| Code | Meaning |
|---|---|
| Engaged | No losses, no movement |
| A1 | Attacker loses 1 unit (eliminated) |
| EX | Both sides lose 1 unit |
| D1 | Defender loses 1 unit |
| D1R | Defender loses 1 unit AND retreats to first available adjacent non-enemy space |
| D2 | Defender loses 2 units |
| DE | Defender eliminated; attacker takes control of space |

**Unit strength used:** count of non-eliminated units of each faction at the combat spaces (all units count equally regardless of type).

**Level-3 trench special rule:** If trench_level ≥ 3, the first defender step loss from the result is absorbed (removed). A combat card is required to attack a Level-3 trench at all.

### 8.2 Advance After Combat

- On DE: attacker takes control of the space; VP recalculated.
- On D1R: defender retreats one space; control does not change unless space is now empty.
- Attacker does **not** advance after combat (units stay at their source space).

---

## 9. Starting Positions

### 9.1 AP Starting Units (11 on-board, 102 OFFBOARD)

| Unit | Type | Strength | Location |
|---|---|---|---|
| #032 | ARMY | 3 | CHATEAU_THIERRY |
| #034 | ARMY | 3 | SEDAN |
| #035 | ARMY | 3 | VERDUN |
| #036 | ARMY | 3 | NANCY |
| #037 | ARMY | 3 | BELFORT |
| #038 | ARMY | 3 | DIJON |
| #039 | ARMY | 3 | PARIS |
| #052 | ARMY | 3 | VIENNA |
| #056 | ARMY | 3 | BRESLAU |
| #060 | ARMY | 3 | DANZIG |
| #061 | ARMY | 3 | KONIGSBERG |

### 9.2 CP Starting Units (6 on-board, 74 OFFBOARD)

| Unit | Type | Strength | Location |
|---|---|---|---|
| #001 | ARMY | 5 | AACHEN |
| #007 | ARMY | 5 | MULHOUSE |
| #008 | ARMY | 5 | STRASBOURG |
| #009 | ARMY | 5 | METZ |
| #010 | ARMY | 5 | COBLENZ |
| #030 | ARMY | 1 | RIGA |

> CP starts with significantly stronger on-board forces (5 strength vs 3) but AP has more units available.

---

## 10. Supply

Supply is checked via **BFS** (breadth-first search) from each space back to a supply source.

**AP supply sources:** French (FR) and Belgian (BE) spaces currently controlled by AP.  
**CP supply sources:** German (GE) and Austro-Hungarian (AH) spaces currently controlled by CP.

A space is **Out of Supply (OOS)** if no path exists through contiguous **friendly-controlled** spaces to a supply source.

OOS status is recorded on the observation but does **not** currently prevent movement or combat in this simulation.

---

## 11. Zone of Control (ZOC)

All non-eliminated combat units (any type) project ZOC into all adjacent spaces.  
ZOC is recorded on the observation but does **not** currently restrict movement in this simulation.

---

## 12. Action Encoding (for developers)

```
Action 0          → PASS
Actions 1–110     → EVENT card (card_idx = action - 1)
Actions 111–440   → OPS  (card_idx = (action-111)//3, op_type = (action-111)%3)
                    op_type 0 = MOVE, 1 = ATTACK, 2 = SR
Actions 441–5340  → MOVE_UNIT (src = (action-441)//72, tgt = (action-441)%72)
```

Total action space: **5341 actions**.

---

## 13. Observation Tensor (32 planes × 72 spaces)

| Plane | Contents |
|---|---|
| 0 | Terrain type / 4.0 |
| 1 | VP value / 5.0 |
| 2 | Control (0=AP, 0.5=Neutral, 1.0=CP) |
| 3 | Trench level / 3.0 |
| 4 | Trench level ≥ 3 (bool) |
| 5 | Fort destroyed (bool) |
| 6 | AP out-of-supply (bool) |
| 7 | CP out-of-supply (bool) |
| 8–15 | AP unit counts by type / 3.0 |
| 16–23 | CP unit counts by type / 3.0 |
| 24 | AP hand size / 10.0 |
| 25 | CP hand size / 10.0 |
| 26 | AP mean OPS / 5.0 |
| 27 | CP mean OPS / 5.0 |
| 28 | VP track (normalised: (VP+20)/40) |
| 29 | AP in Total War (bool) |
| 30 | CP in Total War (bool) |
| 31 | Active player is CP (bool) |

Card context: **7 slots × 16 features** (ops, sr, is_combat, phase_gate, faction, padding).

---

*This rulebook was generated from the actual simulation code. Any discrepancy between this document and the code in `src/env/pog_env.py` / `src/env/jax_env.py` should be resolved in favour of the code.*
