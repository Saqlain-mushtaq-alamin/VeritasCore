# Phase R7: Calibration Plots & Error Analysis

> Generated: 2026-11-11 14:36 UTC

---

## Part A: Calibration / Reliability Diagrams

### HaluEval QA

- **Brier Score (raw)**: 0.2092
- **Brier Score (Platt-scaled)**: 0.2060
- **Improvement after calibration**: +0.0033
- **AUROC**: 0.7523

### FEVER (validation)

- **Brier Score (raw)**: 0.1674
- **Brier Score (Platt-scaled)**: 0.1547
- **Improvement after calibration**: +0.0127
- **AUROC**: 0.9068

---

## Part B: Systematic Error Analysis

### Error Category Distribution

#### HaluEval QA

**Total errors**: 150 / 500 (30.0% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 54 | 3 | 57 | 38.0% |
| Numerical claims (NLI may not distinguish exact numbers) | 32 | 36 | 68 | 45.3% |
| Negation (NLI negation handling is a known weakness) | 1 | 0 | 1 | 0.7% |
| Multi-hop / complex (single NLI pass insufficient) | 2 | 0 | 2 | 1.3% |
| Temporal (evidence may be outdated or ambiguous) | 0 | 3 | 3 | 2.0% |
| Other / unknown | 3 | 16 | 19 | 12.7% |

### Failure Case Gallery


#### Negation Examples

**FP** (score=0.680)  
- **Claim**: `Q: Fir and Imperata, are a type of tree?  A: no`  
- **Context**: _Firs (Abies) are a genus of 48–56 species of evergreen coniferous trees in the family Pinaceae.Imperata is a small but w_  
- **True**: supported  **Predicted**: hallucinated


#### Numerical Examples

**FN** (score=0.255)  
- **Claim**: `Q: Which single seater auto-racing formula did 1977 Can-Am season series used  A: The 1977 Can-Am season series used modified version of Formula 5000.`  
- **Context**: _ This series used Formula 5000 Lola T332s with fenders added.Formula 5000 (or F5000) was an open wheel, single seater au_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.284)  
- **Claim**: `Q: Eugene Isaac Meyer was the father of a publisher whose memoir won what in 1998?  A: Eugene Isaac Meyer was the father of a publisher whose memoir w`  
- **Context**: _Eugene Isaac Meyer (October 31, 1875 – July 17, 1959) was an American financier, public official, and newspaper publishe_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.311)  
- **Claim**: `Q: When was the Russian librettist whose work was used in the opera Dubrovsky born?  A: Modest Ilyich Tchaikovsky was born in the mid-1800s.`  
- **Context**: _Dubrovsky (Russian: "Дубровский" ) is an opera in four acts (5 scenes) Op. 58, by Eduard Nápravník, to a Russian librett_  
- **True**: hallucinated  **Predicted**: supported


#### Temporal Examples

**FN** (score=0.302)  
- **Claim**: `Q: Who was born first, Sherman Alexie or Adunis?  A: Sherman Alexie was born first.`  
- **Context**: _Sherman Joseph Alexie, Jr. (born October 7, 1966) is a Spokane-Coeur d'Alene-American novelist, short story writer, poet_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.412)  
- **Claim**: `Q: Who was born Earlier, Jon Amiel or Michael Ritchie?  A: Jon Amiel was born earlier.`  
- **Context**: _Michael Brunswick Ritchie (November 28, 1938 – April 16, 2001) was an American film director of films with comical or sa_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.480)  
- **Claim**: `Q: Which magazine was founded first, The General or XXL?  A: XXL was founded before The General.`  
- **Context**: _The General (1964–1998) was a bi-monthly periodical devoted to supporting Avalon Hill's line of wargames, with articles _  
- **True**: hallucinated  **Predicted**: supported


#### Short Answer Examples

**FN** (score=0.422)  
- **Claim**: `Q: Which players' team, Seth or Stephen Curry, is closer to New Orleans?  A: Stephen Curry`  
- **Context**: _Seth Adham Curry (born August 23, 1990) is an American professional basketball player for the Dallas Mavericks of the Na_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.426)  
- **Claim**: `Q: What was the company orginally named first?  A: EMC Corporation`  
- **Context**: _Dell Technologies Inc. is an American multinational information technology corporation based in Round Rock, Texas.Dell E_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.428)  
- **Claim**: `Q: Paul Di'Anno and Joe Gooch participated in what band position?  A: guitarist and songwriter.`  
- **Context**: _Paul Andrews (born 17 May 1958), better known by his stage name Paul Di'Anno, is a British singer best known as the firs_  
- **True**: hallucinated  **Predicted**: supported


#### Other Examples

**FN** (score=0.271)  
- **Claim**: `Q: Kim Warwick and Matt Lucena both play what sport?  A: Kim Warwick and Matt Lucena both played professional tennis.`  
- **Context**: _Kim Warwick (born 8 April 1952) is an Australian former professional male tennis player who competed on the ATP Tour fro_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.305)  
- **Claim**: `Q: Patrick MacLeamy is the chairman and former CEO of which American architecture, engineering and planning firm?  A: Patrick MacLeamy used to work at`  
- **Context**: _Patrick MacLeamy, FAIA, LEED AP, (born October 2, 1942, in Alton, Illinois) is an American architect who is chairman and_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.309)  
- **Claim**: `Q: Ares and Right On!, is which type of publication?  A: Ares and Right On! are publications.`  
- **Context**: _Ares was a science fiction wargame magazine published by Simulations Publications, Inc. (SPI), and then TSR, Inc., betwe_  
- **True**: hallucinated  **Predicted**: supported


#### Complex Sentence Examples

**FP** (score=0.803)  
- **Claim**: `Q:  large, primarily residential, public research university in Kent, Ohio, United States has an online folk music stream that is hosted by an America`  
- **Context**: _Gene Shay (born Ivan Shaner, March 4, 1935 in Philadelphia, Pennsylvania) is an American radio personality. He is a repr_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.651)  
- **Claim**: `Q: While both singer-songwriters have displayed incredible talent, what sets Robert Palmer apart from Wayne Coyne, causing him to gain populatiry in t`  
- **Context**: _ He is the lead singer, occasional backing vocalist, guitarist, keyboardist, theremin player and songwriter for the band_  
- **True**: supported  **Predicted**: hallucinated

#### FEVER (validation)

**Total errors**: 75 / 500 (15.0% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 0 | 0 | 0 | 0.0% |
| Numerical claims (NLI may not distinguish exact numbers) | 0 | 37 | 37 | 49.3% |
| Negation (NLI negation handling is a known weakness) | 0 | 4 | 4 | 5.3% |
| Multi-hop / complex (single NLI pass insufficient) | 0 | 0 | 0 | 0.0% |
| Temporal (evidence may be outdated or ambiguous) | 1 | 0 | 1 | 1.3% |
| Other / unknown | 11 | 22 | 33 | 44.0% |

### Failure Case Gallery


#### Negation Examples

**FN** (score=0.362)  
- **Claim**: `Arizona is not in the United States.`  
- **Context**: _It has borders with New Mexico , Utah , Nevada , California , and Mexico , and one point in common with the southwestern_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.374)  
- **Claim**: `Leslie is not the middle name of Neil Diamond.`  
- **Context**: _Neil Leslie Diamond -LRB- born January 24 , 1941 -RRB- is an American singer-songwriter , musician and actor ._  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.375)  
- **Claim**: `The Faroe Islands are no longer part of the Kingdom of Mercia.`  
- **Context**: _Between 1035 and 1814 , the Faeroes were part of the Hereditary Kingdom of Norway ._  
- **True**: hallucinated  **Predicted**: supported


#### Numerical Examples

**FN** (score=0.274)  
- **Claim**: `Rob Sheridan finished college in 1979.`  
- **Context**: _Rob Sheridan -LRB- born October 11 , 1979 -RRB- is an American graphic designer , director , and photographer best known_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.288)  
- **Claim**: `Alice Cooper graduated college on February 4th, 1948.`  
- **Context**: _Alice Cooper -LRB- born Vincent Damon Furnier ; February 4 , 1948 -RRB- is an American singer , songwriter , and actor w_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.288)  
- **Claim**: `Janelle Monáe was conceived on December 1st, 1985.`  
- **Context**: _Janelle Monáe Robinson -LRB- born December 1 , 1985 -RRB- -LRB- -LSB- dʒəˈnɛl_moʊˈneɪ -RSB- -RRB- is an American singer _  
- **True**: hallucinated  **Predicted**: supported


#### Temporal Examples

**FP** (score=0.627)  
- **Claim**: `San Diego Comic-Con was founded by Richard Alf.`  
- **Context**: _It was founded as the Golden State Comic Book Convention in 1970 by a group of San Diegans that included Shel Dorf , Ric_  
- **True**: supported  **Predicted**: hallucinated


#### Other Examples

**FN** (score=0.272)  
- **Claim**: `Bhagat Singh was murdered.`  
- **Context**: _Singh was convicted and hanged in March 1931 , aged 23 . A murder is the unlawful killing of another human without justi_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.298)  
- **Claim**: `Recovery is an Eminem song.`  
- **Context**: _To promote the album , he performed the album 's songs live on televised shows , at award ceremonies , musical events an_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.309)  
- **Claim**: `Kuching is a province of Taiwan.`  
- **Context**: _The Japanese government set up a Batu Lintang camp near Kuching to hold prisoners of war and civilian internees . Batu L_  
- **True**: hallucinated  **Predicted**: supported


---

## Part C: Per-Class Metrics

### HaluEval QA

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.740 | 0.642 | 0.688 | 257 |
| hallucinated | 0.668 | 0.761 | 0.712 | 243 |
| **macro avg** | 0.704 | 0.702 | 0.700 | 500 |

### FEVER (validation)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.784 | 0.950 | 0.859 | 240 |
| hallucinated | 0.943 | 0.758 | 0.840 | 260 |
| **macro avg** | 0.863 | 0.854 | 0.849 | 500 |


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