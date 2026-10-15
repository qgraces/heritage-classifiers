# Classifier use in two Mandarin–English heritage children and a monolingual child

Data and code for "A Three-Child Case Study on Classifier Use in Heritage and Monolingual Mandarin".

The recordings are in CHILDES (Child Heritage Chinese Corpus: Avia, Luna; Zhou Corpus 3: XueXue). This repository
contains the coded classifier tokens and the scripts that reproduce every table, figure and in-text number.

## Files

| File | Content |
|---|---|
| `data/children_tokens.csv` | 2,675 classifier tokens: Avia (857), Luna (943), XueXue (875) |
| `data/mothers_tokens.csv` | 802 classifier tokens: Avia's mother (196), Luna's mother (70), XueXue's mother (536) |
| `data/nouns_in_paper.csv` | The nouns behind two counts in Section 3.3: the nouns each child used with both 个 and a specific classifier (`use` = `alternation`), and the nouns Avia used with 只, 本 and 条 (`use` = `specific_classifier_pairing`) |
| `analysis.py` | Table 1, Table 2 and all numbers quoted in the text → `output/table1.csv`, `output/table2.csv`, `output/numbers.txt` |
| `figures.py` | Figures 1 and 2 → `output/figure1.png`, `output/figure2.png` |
| `random_states.json` | Random-number-generator state at the start of each bootstrap/permutation procedure (see below) |

## Running

```
pip install -r requirements.txt
python analysis.py     # about one minute
python figures.py      # a Chinese-capable font (e.g., Noto Sans CJK SC) is needed for 个 in the axis labels
```

Both scripts read only the two CSV files. `output/numbers.txt` is organised by manuscript section.

## Columns (both files)

| Column | Values | Meaning |
|---|---|---|
| `speaker` | Avia, Luna, XueXue; Avia's mother, Luna's mother, XueXue's mother | Speaker of the token |
| `session` | e.g. `Avia_020015.cha` | Recording session (transcript file); the unit resampled in the bootstrap |
| `age_months` | integer | Child's age in months at the session (2;6 = 30) |
| `utterance` | text | Utterance containing the classifier, as transcribed (CHAT conventions, words separated by spaces) |
| `classifier` | e.g. 个, 只, 本 | The classifier produced |
| `classifier_type` | `general`, `specific` | 个 gè versus any other classifier |
| `misselection` | `yes`, `no`, blank | For specific classifiers: whether the classifier is unsuitable for its noun. Blank for 个 |
| `noun_status` | `overt`, `omitted` | Whether the noun is overt or omitted (pronominal use, with or without an adjective) |
| `construction` | `DEM+CL`, `NUM+CL`, `CL+N`, `DEM+NUM+CL`, `Other` | Construction. `Other` = fillers, quantifier phrases and unclassified tokens |
| `note` | text, blank | Manuscript labels attached to individual tokens: standard classifier for the four misselections, the two kinds of redundant classifier phrase, and the self-correction |

The token files have no noun column. `data/nouns_in_paper.csv` holds the only noun-level information the paper uses (Section 3.3). Columns: `use`, `speaker`, `noun`, `classifier` (the specific classifier), `n_general` (个 tokens with the noun), `n_specific` (specific-classifier tokens with the noun), `within_one_session` (`yes` if both forms occurred in one session; alternation rows only).

## Definitions

* **个 share** = tokens with `classifier_type == general` / all tokens. Misselections count as specific.
* **Shared window** = 2;0–3;11 (24–47 months).
* **Intervals.** Children: 95% percentile interval from 5,000 bootstrap resamples of whole sessions. Mothers: exact (Clopper–Pearson) binomial interval.
* **Comparing children (Table 2).** Pooled difference with a session-bootstrap interval, and a session-level permutation test (child labels shuffled across sessions within year of age, 10,000 permutations), Holm-adjusted over the three contrasts.
* **Rarefaction.** Expected number of distinct specific-classifier types in a random draw of 13 tokens (the number of correctly selected specific tokens that Luna produced in the shared window), after Hurlbert (1971).

## Numbers that cannot come from the token files

Three kinds of numbers in the manuscript describe things that are not classifier tokens. They are written as
constants in `analysis.py` (marked `UPSTREAM`, `OMISSIONS`, `REDUNDANT_UTTERANCE_WITHOUT_TOKEN`) and printed next to the numbers derived from the data:

* the corpus search and de-duplication counts (2,244 candidate tokens, 444 removed for the heritage children; 1,045 and 170 for XueXue). The script checks that candidates minus duplicates equals the number of rows in `children_tokens.csv`;
* classifier omissions (absences, found by searching numeral and demonstrative contexts);
* one utterance with a redundant classifier phrase (Luna, 4;0) that has no classifier token in the file. The utterance, &~en <我 想 讲> [//] &~m &~那个 我 想 讲 一 个 那个 什么 这个 ., was coded as sung material, which was generally excluded to avoid non-spontaneous song lyrics, but it does not appear to be part of a song, so it is counted in the text (14 utterances = 13 in the file + 1).

## Random numbers

Bootstrap and permutation procedures are random. `random_states.json` stores the generator state (numpy PCG64, seed 2026) at the start of each procedure in the run that produced the reported values, so the printed digits are identical. Deleting a key makes that procedure fall back to a fixed seed, which changes the last digit of some intervals and p values.
