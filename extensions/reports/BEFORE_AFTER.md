# Before/After Diff — amr_lda

Comparing **run3_clean** (before) vs **run4_clean** (after).

- Coverage delta: 0 new items judged | 0 dropped
- Flips NEQ → EQ: **18**
- Flips EQ → NEQ: **0** (regression)
- Stable EQ: 38
- Stable NEQ: 34

## Items the patch RECOVERED (18 items)

**amr_lda::S003::implication** (implication)  
Input: _If a triangle has three equal sides, then it has three equal angles._  
Output: **The triangle has no 3 sides or 3 angles.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S007::contraposition** (contraposition)  
Input: _If a meeting is scheduled for Friday, then no team members are available for travel that day._  
Output: **The meeting was not scheduled if team members were available to travel on the day of the meeting.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S008::contraposition** (contraposition)  
Input: _If a patient takes the medication and follows the doctor's instructions, then they recover within two weeks._  
Output: **If the patient does not recover within 2 weeks, he or she will take the medication and follow the doctor's instructions.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S009::commutative** (commutative)  
Input: _The bald eagle is clever and the wolf is fierce._  
Output: **The Wolf is fierce and the Bald Eagle is clever.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=EQUIVALENT  

**amr_lda::S010::commutative** (commutative)  
Input: _Either the report is filed by Monday or the deadline extension is approved._  
Output: **Approved extension of deadline or report filed by Monday.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=EQUIVALENT  

**amr_lda::S011::commutative** (commutative)  
Input: _The athlete trains every morning and competes on weekends._  
Output: **The athletes compete on weekends and train in the morning.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S014::de_morgan** (de_morgan)  
Input: _Neither the manager nor the assistant attended the meeting._  
Output: **There were no managers or assistants attending the meeting.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=EQUIVALENT  

**amr_lda::S020::contraposition** (contraposition)  
Input: _If it is not the case that the alarm did not sound, then the security system did not malfunction._  
Output: **If the security system malfunctions, the alarm sounds.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S020::double_negation** (double_negation)  
Input: _If it is not the case that the alarm did not sound, then the security system did not malfunction._  
Output: **If the alarm didn't sound, the security system was malfunctioning.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S023::double_negation** (double_negation)  
Input: _Some answer in the exam is incorrect._  
Output: **Some answers in the exam were correct.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S025::commutative** (commutative)  
Input: _Every cat is a mammal and every mammal is a vertebrate._  
Output: **Every vertebrate is a mammal, every cat is a mammal.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S027::commutative** (commutative)  
Input: _When the chairperson is unavailable, the vice chair conducts the meeting, and minutes are circulated within 24 hours._  
Output: **If the chairman is unavailable, the meeting will circulate for up to 24 hours and the vice chairman will conduct the meeting.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S029::inverse_relation** (inverse_relation)  
Input: _Tom Cruise's mother is Mary Lee Pfeiffer._  
Output: **Tom Cruise's son is Mary Lee Pfeiffer.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S030::inverse_relation** (inverse_relation)  
Input: _Marie Curie discovered radium in 1898._  
Output: **Radium discovered Marie Curie in 1898.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S031::inverse_relation** (inverse_relation)  
Input: _The teacher praised the student._  
Output: **Students appreciate teachers.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S034::inverse_relation** (inverse_relation)  
Input: _Beijing is the capital of China._  
Output: **Beijing is the country of China.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S047::inverse_relation** (inverse_relation)  
Input: _Marie Curie was awarded the Nobel Prize in Physics in 1903, and she was awarded the Nobel Prize in Chemistry in 1911._  
Output: **In 1903, Marie Curieuse was awarded the Nobel Prize in Physics and the Nobel Prize in Chemistry in 1911.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S050::contraposition** (contraposition)  
Input: _It is not the case that, if Alice is not invited, then Bob will not attend the party._  
Output: **If Bob had not attended the party, Alice would have been invited.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  
