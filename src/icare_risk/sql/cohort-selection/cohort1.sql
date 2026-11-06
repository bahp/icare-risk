/*
=============================================================================
Query Purpose:
Extract all healthcare records for admitted patients who had a blood culture sample collected
within 48 hours of their admission.

Logic:
1. Identify blood culture tests in microbiology table.
2. Joins to the episodes (admissions) table on subject ID.
3. Filters to ensure the test collection falls within the 48-hour admission window
=============================================================================
*/
WITH MICRO_BLDCUL AS (
    SELECT LAB_TEST_ID
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE order_code IN ('bldcul', 'blpcul', 'fbcul', '9fbcul')
),
MICRO AS (
    SELECT *
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE LAB_TEST_ID IN (SELECT DISTINCT LAB_TEST_ID FROM MICRO_BLDCUL)
)
SELECT 
    m.*, a.SPELL_IDENTIFIER, a.ENCNTR_ID, a.EPISODE_IDENTIFIER, a.ADMISSION_DATE,
    a.ADMISSION_METHOD, a.ADMISSION_METHOD_DESC, a.ADMISSION_SOURCE,
    a.ADMISSION_SOURCE_DESC, a.ADMISSION_TIME, a.AGE_AT_ADMISSION,
    a.DISCHARGE_DATE, a.DISCHARGE_DESTINATION, a.DISCHARGE_DESTINATION_DESC,
    a.DISCHARGE_METHOD, a.DISCHARGE_METHOD_DESC,
    a.EPISODE_START_DATE, a.EPISODE_START_TIME, a.EPISODE_END_DATE, a.EPISODE_END_TIME,
    a.MAIN_SPECIALTY_CODE, a.MAIN_SPECIALTY_CODE_DESC, a.ORDER_NO_OF_EPISODE,
    a.POSTCODE_ON_ADMISSION, a.INDEX_OF_MULTIPLE_DEPRIVATION_DECILE,
    a.TREATMENT_FUNCTION_CODE, a.TREATMENT_FUNCTION_CODE_DES
FROM ICHT_PROD.ICARE_ICHT.ICARE_EPISODES_ANON a
INNER JOIN MICRO m ON m.SUBJECT = a.SUBJECT
WHERE a.discharge_date != 'none'
AND m.latest_collect_dt BETWEEN a.admission_date AND DATEADD(HOUR, 48, a.ADMISSION_DATE);