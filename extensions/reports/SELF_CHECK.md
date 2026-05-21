# Self-Consistency Check — Quality Breakdown

Demonstrates the value of the T5wtense self-consistency check by
comparing quality of items that passed vs failed the polarity
parity check.

| Status group | N | V1 AMR EQ | V2 LLM EQ | Consensus EQ | Consensus NEQ | Pending |
|---|---|---|---|---|---|---|
| ok | 62 | 52 (83%) | 7 (11%) | 7 | 9 | 46 |
| self_check_failed | 15 | 14 (93%) | 2 (13%) | 2 | 1 | 12 |
| rule_did_not_fire | 13 | 0 (0%) | 0 (0%) | 0 | 13 | 0 |

## Items flagged by self-consistency check

These outputs had a polarity-parity mismatch between the rule-applied AMR (expected) and the AMR re-parsed from T5wtense's generated text. Each one is a known generator failure — the rule worked correctly but the text generator dropped or flipped a negation.

| ID | Rule | Input | Output | Status |
|---|---|---|---|---|
| S004 | contraposition | _If the water reaches 100 degrees Celsius at sea level, then ..._ | **Without boiling the water would reach 100C at sea level.** | polarity_parity_flipped: expected 2, got 1 |
| S004 | implication | _If the water reaches 100 degrees Celsius at sea level, then ..._ | **At sea level, the water did not reach 100 degrees Celsius (104 Fahrenheit), or b** | polarity_parity_flipped: expected 1, got 2 |
| S005 | contraposition | _If Mary owns a car, then she has a driver's license._ | **Mary does not have a license to drive.** | polarity_parity_flipped: expected 2, got 1 |
| S008 | contraposition | _If a patient takes the medication and follows the doctor's i..._ | **If the patient does not recover within 2 weeks, he or she will take the medicati** | polarity_parity_flipped: expected 2, got 1 |
| S013 | de_morgan | _It is not the case that Alice is tall and Bob is short._ | **In no case was Alice tall or Bob short.** | polarity_parity_flipped: expected 2, got 1 |
| S014 | de_morgan | _Neither the manager nor the assistant attended the meeting._ | **There were no managers or assistants attending the meeting.** | polarity_parity_flipped: expected 2, got 1 |
| S022 | contraposition | _If every employee submits their timesheet, then payroll is p..._ | **If the payroll is not processed then every employee will submit his timesheet.** | polarity_parity_flipped: expected 2, got 1 |
| S026 | contraposition | _If a country produces more goods than it consumes, it export..._ | **If the country does not export surplus to trade partners, it produces more goods** | polarity_parity_flipped: expected 2, got 1 |
| S026 | implication | _If a country produces more goods than it consumes, it export..._ | **The country does not produce more goods than it consumes or exports surplus to t** | polarity_parity_flipped: expected 1, got 2 |
| S028 | contraposition | _If a software vulnerability is reported and the patch is del..._ | **If there is no system risk and the security team must be notified, the software ** | polarity_parity_flipped: expected 2, got 1 |
| S040 | modal_strength_inversion | _Alice must finish her homework before dinner._ | **Alice can't finish her homework before dinner.** | polarity_parity_flipped: expected 2, got 1 |
| S041 | modal_strength_inversion | _Visitors may use the WiFi after registering at the front des..._ | **No, visitors are not prohibited from using WiFi after registering at the front d** | polarity_parity_flipped: expected 2, got 1 |
| S045 | contraposition | _If every passenger boards by 8 AM and the crew is ready, the..._ | **If the flight didn't depart at 8:15 am, every passenger would have boarded by 8 ** | polarity_parity_flipped: expected 2, got 1 |
| S048 | double_negation | _It is mandatory that the report not be released until the au..._ | **The report is not mandated to be released until the audit is completed.** | polarity_parity_flipped: expected 2, got 1 |
| S050 | implication | _It is not the case that, if Alice is not invited, then Bob w..._ | **Alice was invited or, in no case, Bob didn't attend the party.** | polarity_parity_flipped: expected 2, got 1 |