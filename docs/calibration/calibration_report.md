# Phase R7: Calibration Plots & Error Analysis

> Generated: 2026-10-14 19:26 UTC

---

## Part A: Calibration / Reliability Diagrams

### HaluEval QA

- **Brier Score (raw)**: 0.2252
- **Brier Score (Platt-scaled)**: 0.2192
- **Improvement after calibration**: +0.0060
- **AUROC**: 0.6945
- **Plot**: `docs\calibration\calibration_nli_halueval_qa.png`

### FEVER (validation)

- **Brier Score (raw)**: 0.1653
- **Brier Score (Platt-scaled)**: 0.0826
- **Improvement after calibration**: +0.0827
- **AUROC**: 0.9541
- **Plot**: `docs\calibration\calibration_nli_fever_validation.png`

---

## Part B: Systematic Error Analysis

### Error Category Distribution

#### HaluEval QA

**Total errors**: 218 / 500 (43.6% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 108 | 0 | 108 | 49.5% |
| Numerical claims (NLI may not distinguish exact numbers) | 77 | 6 | 83 | 38.1% |
| Negation (NLI negation handling is a known weakness) | 4 | 0 | 4 | 1.8% |
| Multi-hop / complex (single NLI pass insufficient) | 2 | 2 | 4 | 1.8% |
| Temporal (evidence may be outdated or ambiguous) | 0 | 2 | 2 | 0.9% |
| Other / unknown | 5 | 12 | 17 | 7.8% |

### Failure Case Gallery


#### Complex Sentence Examples

**FN** (score=-0.204)  
- **Claim**: `Q: The NBC-affiliated television station, owned by an American telecommunications company whose founder is Julian Sinclair Smith, called WJAR is more `  
- **Context**: _WJAR (more commonly known as NBC 10) is the NBC-affiliated television station for the state of Rhode Island and Bristol _  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.199)  
- **Claim**: `Q: Steve Peterson appeared as which character in a Shakespeare comedy written for the close of the Christmas season?  A: Steve Peterson played a chara`  
- **Context**: _Steve Peterson is an American actor was seen as Stanley in "The Body" at the Matrix Theatre, King Arthur in Dennis Gerst_  
- **True**: hallucinated  **Predicted**: supported

**FP** (score=-0.001)  
- **Claim**: `Q: While both singer-songwriters have displayed incredible talent, what sets Robert Palmer apart from Wayne Coyne, causing him to gain populatiry in t`  
- **Context**: _ He is the lead singer, occasional backing vocalist, guitarist, keyboardist, theremin player and songwriter for the band_  
- **True**: supported  **Predicted**: hallucinated


#### Numerical Examples

**FN** (score=-0.200)  
- **Claim**: `Q: Irma Pamela Hall, an American actress who has appeared in numerous films and television shows since the 1970s, and is best known for playing matria`  
- **Context**: _ She is best known for playing matriarchal figures in the films "A Family Thing", "Soul Food", and "The Ladykillers".The_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.198)  
- **Claim**: `Q: The End is an EP by English heavy metal band Black Sabbath, released on which date, it was only available at dates on their final tour The End Tour`  
- **Context**: _The End is an EP by English heavy metal band Black Sabbath, released on January 20, 2016.The End Tour was the farewell t_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.196)  
- **Claim**: `Q: When did the football coach, who led the 2004 Oklahoma Sooners team,  announce his retirement?  A: Bob Stoops retired in 2017.`  
- **Context**: _The 2004 Oklahoma Sooners football team represented the University of Oklahoma during the 2004 NCAA Division I-A footbal_  
- **True**: hallucinated  **Predicted**: supported


#### Temporal Examples

**FN** (score=-0.192)  
- **Claim**: `Q: What era was the man associated with Fanny Talbot the leading art critic of?  A: John Ruskin, a prominent social thinker, was associated with Fanny`  
- **Context**: _Fanny Talbot (née) Browne (1824–1917) was a landowner and philanthropist, and a friend and correspondent of the influent_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.187)  
- **Claim**: `Q: Where was the British singer-songwriter who composed The Princess Bride's soundtrack born?  A: Mark Knopfler, the British singer-songwriter who com`  
- **Context**: _The Princess Bride is a soundtrack album by British singer-songwriter and guitarist Mark Knopfler, released on 12 Novemb_  
- **True**: hallucinated  **Predicted**: supported


#### Short Answer Examples

**FP** (score=0.967)  
- **Claim**: `Q: Who was born first, Francis Nethersole or Elizabeth Stuart?  A: Sir Francis Nethersole`  
- **Context**: _Sir Francis Nethersole (1587–1659) was an English diplomat, secretary to the Electress Elizabeth, Member of Parliament f_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.942)  
- **Claim**: `Q: What country are Brockton High School and Dufferin Mall in?  A: Canada`  
- **Context**: _Brockton High School (also known as Brockton HS, BHS, or simply known as Brockton) is a Toronto District School Board le_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.921)  
- **Claim**: `Q: What star of Common appears in Game of Thrones?  A: Michelle Fairley`  
- **Context**: _Common is a 2014 BBC One 90-minute made-for-television drama, written by Jimmy McGovern, directed by David Blair and sta_  
- **True**: supported  **Predicted**: hallucinated


#### Negation Examples

**FP** (score=0.997)  
- **Claim**: `Q: Are both Print and National Journal periodicals?  A: no`  
- **Context**: _Print, A Quarterly Journal of the Graphic Arts was a limited edition quarterly periodical begun in 1940 and continued un_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.241)  
- **Claim**: `Q: Yukio Mishima and Roberto Bolaño, are Chilean?  A: no`  
- **Context**: _Yukio Mishima (三島 由紀夫 , Mishima Yukio ) is the pen name of Kimitake Hiraoka (平岡 公威 , Hiraoka Kimitake , January 14, 1925_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.007)  
- **Claim**: `Q: Do the documentaries Traceroute and War/Dance have the same director?  A: no`  
- **Context**: _Traceroute is a 2016 Austrian/American documentary film directed by Johannes Grenzfurthner.War/Dance is a 2007 American _  
- **True**: supported  **Predicted**: hallucinated


#### Other Examples

**FN** (score=-0.197)  
- **Claim**: `Q: True Williams was the first to illustrate which title character of a Mark Twain novel?  A: True Williams was the first to illustrate the character `  
- **Context**: _ He drew all illustrations to the first edition of "The Adventures of Tom Sawyer" (1876) and was thus the first to illus_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.195)  
- **Claim**: `Q: A Place to Live and The Panama Deception are both what?  A: Both have won Academy Awards.`  
- **Context**: _A Place to Live is a 1941 documentary film directed by Irving Lerner and produced by the Philadelphia Housing Associatio_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.194)  
- **Claim**: `Q: What nationality are Ralph Tresvant and Chad Smith?  A: Ralph Tresvant and Chad Smith are both American-born musicians.`  
- **Context**: _Ralph Edward Tresvant Sr. (born May 16, 1968), also known as Rizz and Rizzo, is an American contemporary R&B singer best_  
- **True**: hallucinated  **Predicted**: supported

#### FEVER (validation)

**Total errors**: 114 / 500 (22.8% error rate)

| Error Category | FP | FN | Total | % of Errors |
|---------------|:--:|:--:|:-----:|:-----------:|
| Short answers (NLI struggles with near-empty hypothesis) | 0 | 0 | 0 | 0.0% |
| Numerical claims (NLI may not distinguish exact numbers) | 15 | 2 | 17 | 14.9% |
| Negation (NLI negation handling is a known weakness) | 0 | 0 | 0 | 0.0% |
| Multi-hop / complex (single NLI pass insufficient) | 0 | 0 | 0 | 0.0% |
| Temporal (evidence may be outdated or ambiguous) | 3 | 0 | 3 | 2.6% |
| Other / unknown | 88 | 6 | 94 | 82.5% |

### Failure Case Gallery


#### Numerical Examples

**FN** (score=-0.199)  
- **Claim**: `Janelle Monáe was conceived on December 1st, 1985.`  
- **Context**: _Janelle Monáe Robinson -LRB- born December 1 , 1985 -RRB- -LRB- -LSB- dʒəˈnɛl_moʊˈneɪ -RSB- -RRB- is an American singer _  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.168)  
- **Claim**: `Camden, New Jersey is the home of a school that was founded as the South Jersey Law School in 1920.`  
- **Context**: _The city is the home of Rutgers University -- Camden , which was founded as the South Jersey Law School in 1926 , and Co_  
- **True**: hallucinated  **Predicted**: supported

**FP** (score=0.001)  
- **Claim**: `Hush (2016 film) was produced.`  
- **Context**: _The film was produced by Trevor Macy , through Intrepid Pictures , and Jason Blum through his Blumhouse Productions bann_  
- **True**: supported  **Predicted**: hallucinated


#### Temporal Examples

**FP** (score=0.944)  
- **Claim**: `San Diego Comic-Con was founded by Richard Alf.`  
- **Context**: _It was founded as the Golden State Comic Book Convention in 1970 by a group of San Diegans that included Shel Dorf , Ric_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.020)  
- **Claim**: `Bea Arthur was born under a different name.`  
- **Context**: _Bea Arthur -LRB- born Bernice Frankel ; May 13 , 1922 -- April 25 , 2009 -RRB- was an American actress , comedian , sing_  
- **True**: supported  **Predicted**: hallucinated

**FP** (score=0.002)  
- **Claim**: `Tuscaloosa, Alabama is a place where Due Date was shot.`  
- **Context**: _The film was shot in Las Cruces , New Mexico , Atlanta , Georgia , and Tuscaloosa , Alabama ._  
- **True**: supported  **Predicted**: hallucinated


#### Other Examples

**FN** (score=-0.200)  
- **Claim**: `Mary-Kate Olsen and Ashley Olsen are former child artists.`  
- **Context**: _Mary-Kate Olsen and Ashley Olsen -LRB- born June 13 , 1986 -RRB- , also known as the Olsen twins collectively , are Amer_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.200)  
- **Claim**: `Rupert Murdoch is a broadcaster, forming BSkyB.`  
- **Context**: _Murdoch formed the British broadcaster BSkyB in 1990 , and during the 1990s expanded into Asian networks and South Ameri_  
- **True**: hallucinated  **Predicted**: supported

**FN** (score=-0.199)  
- **Claim**: `Gin is a name.`  
- **Context**: _Gin was developed on the basis of the older jenever , and became popular in Great Britain -LRB- particularly in London -_  
- **True**: hallucinated  **Predicted**: supported


---

## Part C: Per-Class Metrics

### HaluEval QA

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.672 | 0.187 | 0.292 | 241 |
| hallucinated | 0.547 | 0.915 | 0.685 | 259 |
| **macro avg** | 0.609 | 0.551 | 0.489 | 500 |

### FEVER (validation)

| Class | Precision | Recall | F1 | Support |
|-------|:---------:|:------:|:--:|:-------:|
| supported | 0.944 | 0.558 | 0.702 | 240 |
| hallucinated | 0.704 | 0.969 | 0.816 | 260 |
| **macro avg** | 0.824 | 0.764 | 0.759 | 500 |


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