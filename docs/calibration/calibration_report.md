# Phase R7: Calibration Plots & Error Analysis

> Generated: 2026-11-11 10:42 UTC

---

## Part A: Calibration / Reliability Diagrams

### HaluEval QA

- **Brier Score (raw)**: 0.0990
- **Brier Score (Platt-scaled)**: 0.0937
- **Improvement after calibration**: +0.0053
- **AUROC**: 0.9469
- **Plot**: `docs\calibration\calibration_fusion_halueval_qa.png`

### FEVER (validation)

- **Brier Score (raw)**: 0.3686
- **Brier Score (Platt-scaled)**: 0.2464
- **Improvement after calibration**: +0.1223
- **AUROC**: 0.9296
- **Plot**: `docs\calibration\calibration_fusion_fever_validation.png`

---

## Part B: Systematic Error Analysis

### Error Category Distribution

#### HaluEval QA

**Total errors**: 64 / 500 (12.8% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 17 | 3 | 20 | 31.2% |
| Numerical claims (NLI may not distinguish exact numbers) | 15 | 16 | 31 | 48.4% |
| Negation (NLI negation handling is a known weakness) | 0 | 0 | 0 | 0.0% |
| Multi-hop / complex (single NLI pass insufficient) | 1 | 1 | 2 | 3.1% |
| Temporal (evidence may be outdated or ambiguous) | 0 | 1 | 1 | 1.6% |
| Other / unknown | 3 | 7 | 10 | 15.6% |

### Failure Case Gallery


#### Complex Sentence Examples

**FN** (score=0.417)  
- **Claim**: `Q: Which American film, The Country Bears or Bambi, is based on a book by the Austrian author Felix Salten?  A: The Country Bears is inspired by a Dis`  
- **Context**: _Bambi is a 1942 American animated film directed by David Hand (supervising a team of sequence directors), produced by Wa_  
- **True**: hallucinated  **Predicted**: supported

**FP** (score=0.888)  
- **Claim**: `Q:  large, primarily residential, public research university in Kent, Ohio, United States has an online folk music stream that is hosted by an America`  
- **Context**: _Gene Shay (born Ivan Shaner, March 4, 1935 in Philadelphia, Pennsylvania) is an American radio personality. He is a repr_  
- **True**: supported  **Predicted**: hallucinated


#### Temporal Examples

**FN** (score=0.493)  
- **Claim**: `Q: What was the name of the musical group with which the career of the singer of "Shikva" began?  A: Faakhir Mehmood began his career with Vital Signs`  
- **Context**: _"Shikva" (Urdu: ‎ ; lit. "Querulousness" or "Complain" ) is a song by Pakistani recording artist Faakhir Mehmood from hi_  
- **True**: hallucinated  **Predicted**: supported


#### Short Answer Examples

**FN** (score=0.292)  
- **Claim**: `Q: What was the company orginally named first?  A: EMC Corporation`  
- **Context**: _Dell Technologies Inc. is an American multinational information technology corporation based in Round Rock, Texas.Dell E_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.322)  
- **Claim**: `Q: Paul Di'Anno and Joe Gooch participated in what band position?  A: guitarist and songwriter.`  
- **Context**: _Paul Andrews (born 17 May 1958), better known by his stage name Paul Di'Anno, is a British singer best known as the firs_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.485)  
- **Claim**: `Q: What do Hedley Mascot Mine and Ruth Mine have in common?  A: They were both abandoned.`  
- **Context**: _The Hedley Mascot Mine was a gold mine in Hedley, British Columbia, Canada.The Ruth Mine was a small gold producer about_  
- **True**: hallucinated  **Predicted**: supported


#### Numerical Examples

**FN** (score=0.177)  
- **Claim**: `Q: William Stanley, 3rd Baron Monteagle resided in a castle located in what valley?  A: the Hornby Valley`  
- **Context**: _William Stanley, 3rd Baron Monteagle (1528–1581), of Hornby Castle, Lancashire, was an English politician.Hornby Castle _  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.195)  
- **Claim**: `Q: What was the number one single by the band that features a lead singer who won the season finale of a television reality game show that premiered i`  
- **Context**: _The Celebrity Apprentice 3 (also known as The Apprentice 9) is the ninth installment of the reality game show, "Celebrit_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.205)  
- **Claim**: `Q: When was the American microbiologist who developed MMR vaccine born?  A: Maurice Ralph Hilleman was born on August 31, 1919.`  
- **Context**: _The MMR vaccine (also known as the MPR vaccine after the Latin names of the diseases) is an immunization vaccine against_  
- **True**: hallucinated  **Predicted**: supported


#### Other Examples

**FN** (score=0.170)  
- **Claim**: `Q: Dravid Brook co-wrote a song with this American singer, songwriter and record producer with who?  A: Charles Otto Puth Jr. himself.`  
- **Context**: _ He is best known for co-writing Eminem's Legacy, Jason Derulo's "Broke" featuring Stevie Wonder and Keith Urban, and Ch_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.194)  
- **Claim**: `Q: Which architect designed a church near President Grant's tomb?  A: Henry C. Pelton designed an immigrant settlement.`  
- **Context**: _ It was designed by architect Henry C. Pelton (architect of Riverside Church) in the American Perpendicular Style and co_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.231)  
- **Claim**: `Q: What English lyricist wrote the lyrics for the Doug Henning inspired musical Merlin?  A: Don Black is a lyrist.`  
- **Context**: _Merlin was a musical based on a concept by popular illusionist Doug Henning and Barbara De Angelis, written by Richard L_  
- **True**: hallucinated  **Predicted**: supported

#### FEVER (validation)

**Total errors**: 260 / 500 (52.0% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 0 | 0 | 0 | 0.0% |
| Numerical claims (NLI may not distinguish exact numbers) | 0 | 52 | 52 | 20.0% |
| Negation (NLI negation handling is a known weakness) | 0 | 27 | 27 | 10.4% |
| Multi-hop / complex (single NLI pass insufficient) | 0 | 0 | 0 | 0.0% |
| Temporal (evidence may be outdated or ambiguous) | 0 | 3 | 3 | 1.2% |
| Other / unknown | 0 | 178 | 178 | 68.5% |

### Failure Case Gallery


#### Temporal Examples

**FN** (score=0.171)  
- **Claim**: `The Bloods was founded in Los Angeles, Massachusetts.`  
- **Context**: _The Bloods are a primarily , though not exclusively , African American street gang founded in Los Angeles , California ._  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.176)  
- **Claim**: `Due Date is only a television show.`  
- **Context**: _The film was released on November 5 , 2010 ._  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.179)  
- **Claim**: `Peking University was founded in the eighteenth century.`  
- **Context**: _It is the first modern national university established in China , founded as the Imperial University of Peking in 1898 a_  
- **True**: hallucinated  **Predicted**: supported


#### Numerical Examples

**FN** (score=0.156)  
- **Claim**: `Rob Sheridan finished college in 1979.`  
- **Context**: _Rob Sheridan -LRB- born October 11 , 1979 -RRB- is an American graphic designer , director , and photographer best known_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.156)  
- **Claim**: `Alice Cooper graduated college on February 4th, 1948.`  
- **Context**: _Alice Cooper -LRB- born Vincent Damon Furnier ; February 4 , 1948 -RRB- is an American singer , songwriter , and actor w_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.156)  
- **Claim**: `Janelle Monáe was conceived on December 1st, 1985.`  
- **Context**: _Janelle Monáe Robinson -LRB- born December 1 , 1985 -RRB- -LRB- -LSB- dʒəˈnɛl_moʊˈneɪ -RSB- -RRB- is an American singer _  
- **True**: hallucinated  **Predicted**: supported


#### Other Examples

**FN** (score=0.155)  
- **Claim**: `Dan O'Bannon died.`  
- **Context**: _Daniel Thomas `` Dan '' O'Bannon -LRB- September 30 , 1946 -- December 17 , 2009 -RRB- was an American film screenwriter_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.155)  
- **Claim**: `Recovery is an Eminem song.`  
- **Context**: _To promote the album , he performed the album 's songs live on televised shows , at award ceremonies , musical events an_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.155)  
- **Claim**: `Bhagat Singh was murdered.`  
- **Context**: _Singh was convicted and hanged in March 1931 , aged 23 . A murder is the unlawful killing of another human without justi_  
- **True**: hallucinated  **Predicted**: supported


#### Negation Examples

**FN** (score=0.157)  
- **Claim**: `Arizona is not in the United States.`  
- **Context**: _It has borders with New Mexico , Utah , Nevada , California , and Mexico , and one point in common with the southwestern_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.157)  
- **Claim**: `The Faroe Islands are no longer part of the Kingdom of Mercia.`  
- **Context**: _Between 1035 and 1814 , the Faeroes were part of the Hereditary Kingdom of Norway ._  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.157)  
- **Claim**: `Leslie is not the middle name of Neil Diamond.`  
- **Context**: _Neil Leslie Diamond -LRB- born January 24 , 1941 -RRB- is an American singer-songwriter , musician and actor ._  
- **True**: hallucinated  **Predicted**: supported


---

## Part C: Per-Class Metrics

### HaluEval QA

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.888 | 0.860 | 0.874 | 257 |
| hallucinated | 0.857 | 0.885 | 0.870 | 243 |
| **macro avg** | 0.872 | 0.872 | 0.872 | 500 |

### FEVER (validation)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.480 | 1.000 | 0.649 | 240 |
| hallucinated | 0.000 | 0.000 | 0.000 | 260 |
| **macro avg** | 0.240 | 0.500 | 0.324 | 500 |


---

## Verification Checklist

- [x] Reliability diagram generated (NLI-only mode)
- [x] Brier score reported (measures calibration quality)
- [x] Before vs. after Platt scaling comparison shown
- [x] All errors categorized into failure types
- [x] Error category distribution reported as table
- [x] 3–5 illustrative failure examples per major category
- [x] Per-class precision/recall/F1 reported
- [x] Error analysis section drafted for the paper