# Before/After Diff — amr_lda

Comparing **run5_clean** (before) vs **run6_clean** (after).

- Coverage delta: 0 new items judged | 0 dropped
- Flips NEQ → EQ: **3**
- Flips EQ → NEQ: **0** (regression)
- Stable EQ: 64
- Stable NEQ: 23

## Items the patch RECOVERED (3 items)

**amr_lda::S032::symmetric_asymmetric** (symmetric_asymmetric)  
Input: _Alice is married to Bob._  
Output: **Bob and Alice got married.**  
Verdicts: amr_struct=UNKNOWN · llm_gpt-4o-mini=EQUIVALENT  

**amr_lda::S041::modal_strength_inversion** (modal_strength_inversion)  
Input: _Visitors may use the WiFi after registering at the front desk._  
Output: **No, visitors are not prohibited from using WiFi after registering at the front desk.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  

**amr_lda::S048::modal_strength_inversion** (modal_strength_inversion)  
Input: _It is mandatory that the report not be released until the audit is complete._  
Output: **The report could not be released until the audit was completed.**  
Verdicts: amr_struct=EQUIVALENT · llm_gpt-4o-mini=NOT_EQUIVALENT  
