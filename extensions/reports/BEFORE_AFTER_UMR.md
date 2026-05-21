# Before/After Diff — amr_lda

Comparing **run4_clean** (before) vs **run5_clean** (after).

- Coverage delta: 0 new items judged | 0 dropped
- Flips NEQ → EQ: **8**
- Flips EQ → NEQ: **0** (regression)
- Stable EQ: 56
- Stable NEQ: 26

## Items the patch RECOVERED (8 items)

**amr_lda::S019::modal_strength_inversion** (modal_strength_inversion)  
Input: _It is not impossible that the experiment will succeed._  
Output: **The experiment must not have succeeded.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S030::tense_transformation** (tense_transformation)  
Input: _Marie Curie discovered radium in 1898._  
Output: **Marie Curie discovered radium in 1898, tense-shifted.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S037::aspect_equivalence** (aspect_equivalence)  
Input: _Alice ran a marathon yesterday._  
Output: **Alice ran a marathon yesterday as a complete performance.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S037::tense_transformation** (tense_transformation)  
Input: _Alice ran a marathon yesterday._  
Output: **Alice ran a tense shifted marathon yesterday.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S040::modal_strength_inversion** (modal_strength_inversion)  
Input: _Alice must finish her homework before dinner._  
Output: **Alice can't finish her homework before dinner.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S042::modal_strength_inversion** (modal_strength_inversion)  
Input: _The conference cannot be held without proper accreditation._  
Output: **The conference must not be held without proper accreditation.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S043::doc_level_temporal_transitivity** (doc_level_temporal_transitivity)  
Input: _After the manager approved the budget, the team began the project._  
Output: **The team began the project with budget approvals after the manager.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S044::doc_level_temporal_transitivity** (doc_level_temporal_transitivity)  
Input: _The flight took off after the weather cleared, which happened after the storm passed._  
Output: **After the weather cleared, the flight took off after the storm passed.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  
