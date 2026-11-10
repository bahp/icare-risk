/*
================================================================================
Query Purpose:
For each individual hospital spell (admission) of a patient, identify if there
is a microbiology/blood culture sample collected within the first 24 hours of
admission and return patient, spell and culture results details.
================================================================================
*/

WITH ADMISSION_SPELLS AS (
    SELECT 
        SUBJECT,
        SPELL_IDENTIFIER,
        ENCNTR_ID,
        EPISODE_IDENTIFIER,
        ADMISSION_DATE,
        ADMISSION_TIME,
        DISCHARGE_DATE,
        AGE_AT_ADMISSION,
        CAST(CONCAT(ADMISSION_DATE, ' ', COALESCE(ADMISSION_TIME, '00:00:00')) AS TIMESTAMP) AS ADMISSION_TIMESTAMP
    FROM ICHT_PROD.ICARE_ICHT.ICARE_EPISODES_ANON
),
MICRO_TESTS AS (
    SELECT 
        SUBJECT,
        LAB_TEST_ID,
        ORDER_CODE,
        ORDER_NAME,
        SITE,
        ORGANISM_BUG,
        SENSITIVITY,
        LATEST_COLLECT_DT
    FROM ICHT_PROD.ICARE_ICHT.ICARE_MICROBIOLOGY_ANON
    WHERE ORDER_CODE IN ('bldcul', 'blpcul', 'fbcul', '9fbcul')
)
SELECT 
    s.SUBJECT,
    s.SPELL_IDENTIFIER,
    s.ENCNTR_ID,
    s.EPISODE_IDENTIFIER,
    s.ADMISSION_DATE,
    s.ADMISSION_TIME,
    s.AGE_AT_ADMISSION,
    m.LAB_TEST_ID,
    m.ORDER_CODE,
    m.ORDER_NAME,
    m.SITE,
    m.ORGANISM_BUG,
    m.SENSITIVITY,
    m.LATEST_COLLECT_DT
FROM ADMISSION_SPELLS s
INNER JOIN MICRO_TESTS m ON s.SUBJECT = m.SUBJECT
WHERE m.LATEST_COLLECT_DT BETWEEN s.ADMISSION_TIMESTAMP AND DATEADD(HOUR, 48, s.ADMISSION_TIMESTAMP);