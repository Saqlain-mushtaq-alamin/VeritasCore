# Phase R7: Calibration Plots & Error Analysis

> Generated: 2026-11-11 17:07 UTC

---

## Part A: Calibration / Reliability Diagrams

### HaluEval QA

- **Brier Score (raw)**: 0.2035
- **Brier Score (Platt-scaled)**: 0.1997
- **Improvement after calibration**: +0.0038
- **AUROC**: 0.7707

### FEVER (validation)

- **Brier Score (raw)**: 0.1674
- **Brier Score (Platt-scaled)**: 0.1749
- **Improvement after calibration**: -0.0075
- **AUROC**: 0.9068

---

## Part B: Systematic Error Analysis

### Error Category Distribution

#### HaluEval QA

**Total errors**: 183 / 500 (36.6% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 99 | 0 | 99 | 54.1% |
| Numerical claims (NLI may not distinguish exact numbers) | 38 | 20 | 58 | 31.7% |
| Negation (NLI negation handling is a known weakness) | 8 | 1 | 9 | 4.9% |
| Multi-hop / complex (single NLI pass insufficient) | 2 | 0 | 2 | 1.1% |
| Temporal (evidence may be outdated or ambiguous) | 1 | 1 | 2 | 1.1% |
| Other / unknown | 5 | 8 | 13 | 7.1% |

### Failure Case Gallery


#### Numerical Examples

**FN** (score=0.241)  
- **Claim**: `Q: What 1993 Disney movie starring Elijah Wood was based on a Mark Twain novel?  A: The Disney movie with Huck Finn.`  
- **Context**: _Tom and Huck is a 1995 American adventure comedy-drama film based on Mark Twain's novel "The Adventures of Tom Sawyer", _  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.298)  
- **Claim**: `Q: What Beatles song that opens their 1963 debut album, was covered by American singer Tiffany Darwish?  A: Tiffany Darwish covered The Beatles.`  
- **Context**: _Tiffany Renee Darwish (born October 2, 1971), also known as Tiffany, is an American singer, songwriter, actress, and for_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.302)  
- **Claim**: `Q: This brand of Swiss watches is manufactured by a fashion designer founded in what year?  A: Zodiac Watches was founded in 1984.`  
- **Context**: _Fossil Group, Inc. is an American fashion designer and manufacturer founded in 1984 by Tom Kartsotis and based in Richar_  
- **True**: hallucinated  **Predicted**: supported


#### Short Answer Examples

**FP** (score=0.811)  
- **Claim**: `Q: Gerald William Heaney, served for nearly forty years as a federal judge on the United States Court of Appeals for the Eighth Circuit, and was an or`  
- **Context**: _ Between the end of World War II and his appointment to the federal bench, he rewrote the Free State of Bavaria’s labor _  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.792)  
- **Claim**: `Q: he Central Harbourfront is a waterfront site it is the result of Central and Wan Chai Reclamation, and it sits to the east of the International Fin`  
- **Context**: _The Central Harbourfront is a waterfront site in Central, Hong Kong. IFC, branded as "ifc") is a skyscraper and an integ_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.791)  
- **Claim**: `Q: What  is a low-cost airline headquartered at Gurgaon, that is the largest airline in India, the third largest domestic airline being the flag carri`  
- **Context**: _ Additionally, the carrier is the third largest domestic airline in India in terms of passengers carried (after IndiGo a_  
- **True**: supported  **Predicted**: hallucinated


#### Negation Examples

**FN** (score=0.335)  
- **Claim**: `Q: Who is an American musician, Peter Chelsom or Billy Bob Thornton?  A: Peter Chelsom is not an American musician.`  
- **Context**: _Peter Chelsom (born 20 April 1956) is a British film director, writer, and actor.William Robert "Billy Bob" Thornton (bo_  
- **True**: hallucinated  **Predicted**: supported

**FP** (score=0.708)  
- **Claim**: `Q: Not So Quiet is a parody of the film based on the novel written by whom?  A: Erich Maria Remarque`  
- **Context**: _ The title is a parody of "All Quiet on the Western Front", an Academy Award-winning Universal film released in the same_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.697)  
- **Claim**: `Q: Are Raoul Walsh and Puneet Sira both British directors?  A: no`  
- **Context**: _Raoul A. Walsh (March 11, 1887December 31, 1980) was an American film director, actor, founding member of the Academy of_  
- **True**: supported  **Predicted**: hallucinated


#### Complex Sentence Examples

**FP** (score=0.545)  
- **Claim**: `Q: What locations is the folklore which Huldra belongs to representative of?  A: Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands`  
- **Context**: _ The mountain is named after Huldra in Scandinavian folklore.Scandinavian folklore or Nordic folklore is the folklore of_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.519)  
- **Claim**: `Q: What was the baptismal name for the composer for whose works, along with Bach and Handel, Alan Bennett is most noted for his interpretation?    A: `  
- **Context**: _Alan Bennett (born 1962) is an American lyric tenor known mostly for his performances in concert and oratorio work. He i_  
- **True**: supported  **Predicted**: hallucinated


#### Other Examples

**FN** (score=0.255)  
- **Claim**: `Q: Signet Jewelers owns what Delaware based retailer, along with Kay and Jared?  A: Signet Jewelers owns Delaware Jewelry Retailer.`  
- **Context**: _ Signet Jewelers owns and operates the companies Zales, Kay Jewelers, and Jared.The Zale Corporation is an American jewe_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.263)  
- **Claim**: `Q: Mathilde Lemoine is employed by what  international, family-owned and independent financial group?  A: Mathilde Lemoine works for Rothschild Bank.`  
- **Context**: _ She is currently the Group Chief Economist of Edmond de Rothschild.The Edmond de Rothschild Group is an international, _  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=0.263)  
- **Claim**: `Q: John Huggins was the leader of an organization orginally called what?  A: John Huggins was the leader of the Black Nationalist Party.`  
- **Context**: _John Jerome Huggins, Jr. (February 11, 1945 – January 17, 1969) was an American activist. Huggins was the leader in the _  
- **True**: hallucinated  **Predicted**: supported


#### Temporal Examples

**FN** (score=0.381)  
- **Claim**: `Q: For what ride is Pepper's ghost, an illusion technique, used when the Blue Fairy disappears, leaving a pile of fiber-optic fairy dust on the floor?`  
- **Context**: _Pepper's ghost is an illusion technique used in theatre, amusement parks, museums, television, and concerts. Examples of_  
- **True**: hallucinated  **Predicted**: supported

**FP** (score=0.620)  
- **Claim**: `Q: Kalle Makinen is nicknamed 'Sergio' after another famous footballer; what teams does the original Sergio play for?  A: Real Madrid and the Spain na`  
- **Context**: _Kalle Mäkinen (born 1 February 1989) is a Finnish footballer who last played for the Finnish Veikkausliiga club Maskun P_  
- **True**: supported  **Predicted**: hallucinated

#### FEVER (validation)

**Total errors**: 57 / 500 (11.4% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 0 | 0 | 0 | 0.0% |
| Numerical claims (NLI may not distinguish exact numbers) | 0 | 12 | 12 | 21.1% |
| Negation (NLI negation handling is a known weakness) | 0 | 3 | 3 | 5.3% |
| Multi-hop / complex (single NLI pass insufficient) | 0 | 0 | 0 | 0.0% |
| Temporal (evidence may be outdated or ambiguous) | 1 | 0 | 1 | 1.8% |
| Other / unknown | 21 | 20 | 41 | 71.9% |

### Failure Case Gallery


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


#### Temporal Examples

**FP** (score=0.627)  
- **Claim**: `San Diego Comic-Con was founded by Richard Alf.`  
- **Context**: _It was founded as the Golden State Comic Book Convention in 1970 by a group of San Diegans that included Shel Dorf , Ric_  
- **True**: supported  **Predicted**: hallucinated


---

## Part C: Per-Class Metrics

### HaluEval QA

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.773 | 0.400 | 0.527 | 255 |
| hallucinated | 0.584 | 0.878 | 0.701 | 245 |
| **macro avg** | 0.678 | 0.639 | 0.614 | 500 |

### FEVER (validation)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.862 | 0.908 | 0.884 | 240 |
| hallucinated | 0.911 | 0.865 | 0.888 | 260 |
| **macro avg** | 0.886 | 0.887 | 0.886 | 500 |


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