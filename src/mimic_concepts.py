"""Clinical concept dictionaries for MIMIC-IV v3.1.

Everything that maps a raw MIMIC identifier to a clinical concept lives here so
`mimic_iv.py` stays pure ETL mechanics. Three groups:

  * `CHART_ITEMS` / `LAB_ITEMS`  -- itemid -> concept, for the streaming
    aggregators over `icu/chartevents.csv` and `hosp/labevents.csv`.
  * `INPUT_ITEMS` / `PROC_ITEMS` / `OUTPUT_ITEMS` -- ICU treatment exposure,
    the "interventions and medications" block of the SBSCGM feature encoder
    (arXiv:2508.00615 sec. 3.2).
  * `CHARLSON_ICD9` / `CHARLSON_ICD10` -- Quan et al. (2005) code-prefix maps
    for the comorbidity block.

Physiologic plausibility ranges are attached to every numeric concept.
MIMIC chart/lab values contain transcription errors (a heart rate of 8000, a
creatinine of -1); clipping to a clinical range before aggregation is standard
practice and stops one bad row from destroying a patient's min/max features.
"""

# --------------------------------------------------------------- ICU vitals
# itemid -> (concept, unit_hint, lo, hi). Both invasive and non-invasive
# pressures map to the same concept: which line a patient has is a treatment
# decision, not a different physiologic quantity.
CHART_ITEMS = {
    220045: ("heart_rate", "bpm", 10, 300),
    220050: ("sbp", "mmHg", 20, 300),
    220179: ("sbp", "mmHg", 20, 300),
    225309: ("sbp", "mmHg", 20, 300),
    220051: ("dbp", "mmHg", 5, 200),
    220180: ("dbp", "mmHg", 5, 200),
    225310: ("dbp", "mmHg", 5, 200),
    220052: ("map", "mmHg", 10, 250),
    220181: ("map", "mmHg", 10, 250),
    225312: ("map", "mmHg", 10, 250),
    220210: ("resp_rate", "bpm", 2, 80),
    224690: ("resp_rate", "bpm", 2, 80),
    220277: ("spo2", "%", 20, 100),
    223761: ("temp_f", "F", 70, 115),
    223762: ("temp_c", "C", 20, 46),
    220739: ("gcs_eye", "pts", 1, 4),
    223900: ("gcs_verbal", "pts", 1, 5),
    223901: ("gcs_motor", "pts", 1, 6),
    220621: ("glucose_chart", "mg/dL", 10, 1500),
    225664: ("glucose_chart", "mg/dL", 10, 1500),
    226512: ("weight_kg", "kg", 20, 400),
    223835: ("fio2", "%", 20, 100),
}

# ------------------------------------------------------------------- labs
LAB_ITEMS = {
    50912: ("creatinine", "mg/dL", 0.1, 25),
    51006: ("bun", "mg/dL", 1, 250),
    50971: ("potassium", "mEq/L", 1, 12),
    50983: ("sodium", "mEq/L", 90, 200),
    50902: ("chloride", "mEq/L", 60, 160),
    50882: ("bicarbonate", "mEq/L", 3, 60),
    50931: ("glucose_lab", "mg/dL", 10, 1500),
    51221: ("hematocrit", "%", 5, 70),
    51222: ("hemoglobin", "g/dL", 2, 25),
    51301: ("wbc", "K/uL", 0.1, 200),
    51265: ("platelets", "K/uL", 5, 1500),
    50813: ("lactate", "mmol/L", 0.2, 30),
    50820: ("ph", "", 6.5, 8.0),
    50821: ("po2", "mmHg", 10, 700),
    50818: ("pco2", "mmHg", 5, 150),
    50802: ("base_excess", "mEq/L", -30, 30),
    51237: ("inr", "", 0.5, 15),
    51275: ("ptt", "s", 10, 200),
    50885: ("bilirubin", "mg/dL", 0.1, 50),
    50862: ("albumin", "g/dL", 0.5, 7),
    50863: ("alk_phos", "IU/L", 10, 2000),
    50861: ("alt", "IU/L", 5, 5000),
    50878: ("ast", "IU/L", 5, 5000),
    50960: ("magnesium", "mg/dL", 0.5, 8),
    50893: ("calcium", "mg/dL", 3, 20),
    50970: ("phosphate", "mg/dL", 0.5, 20),
    51003: ("troponin_t", "ng/mL", 0.0, 30),
    50889: ("crp", "mg/L", 0.0, 400),
}

# --------------------------------------------------- ICU treatment exposure
# Binary "did this patient receive X in the window" flags. These are the
# strongest single predictors in an ICU cohort and are exactly the
# "high-risk intervention" indicators the SBSCGM encoder calls for.
INPUT_ITEMS = {
    221906: "norepinephrine",
    221289: "epinephrine",
    221749: "phenylephrine",
    222315: "vasopressin",
    221662: "dopamine",
    221653: "dobutamine",
    221468: "milrinone",
    222168: "propofol",
    221668: "midazolam",
    221744: "fentanyl",
    225158: "crystalloid_ns",
    225828: "crystalloid_lr",
    220862: "albumin_infusion",
    225168: "prbc_transfusion",
}

# Vasopressors get their own aggregate flag: any of these means shock.
VASOPRESSORS = ("norepinephrine", "epinephrine", "phenylephrine",
                "vasopressin", "dopamine")

PROC_ITEMS = {
    225792: "invasive_ventilation",
    225794: "noninvasive_ventilation",
    227194: "extubation",
    224385: "intubation",
    225802: "crrt",
    225803: "crrt",
    225809: "crrt",
    225955: "crrt",
    225441: "hemodialysis",
    225805: "peritoneal_dialysis",
    224263: "central_line",
    224264: "central_line",
    224267: "arterial_line",
    225752: "arterial_line",
    225454: "chest_tube",
}

# Urine output is charted under a long tail of itemids (Foley, void, ileoconduit
# ...). All of them are summed into a single hourly rate.
# Urine output. 227489 is "GU Irrigant/Urine Volume Out", which includes the
# irrigant that was instilled; 227488 is that instilled volume ("GU Irrigant
# Volume In") and is therefore *subtracted*, matching MIMIC's own
# `urine_output` concept. Summing 227488 as if it were urine inflates output
# for anyone on bladder irrigation.
OUTPUT_URINE_ITEMS = (
    226559, 226560, 226561, 226584, 226563, 226564, 226565,
    226567, 226557, 226558, 227489,
)
OUTPUT_URINE_NEGATIVE_ITEMS = (227488,)

# A single charted void or bag reading above this is a charting error, not a
# measurement. The derived rate is bounded separately in `mimic_iv`.
URINE_EVENT_MAX_ML = 2500.0
# Sustained output above this over a whole observation window is not
# physiologic (brisk post-obstructive diuresis peaks well below it).
URINE_RATE_MAX_ML_PER_HR = 1000.0

# ------------------------------------------------------------- comorbidities
# Quan et al. (2005) ICD code-prefix mapping for the Charlson index. Matched on
# `icd_code.startswith(prefix)` after stripping dots, which is how MIMIC-IV
# stores codes.
CHARLSON_ICD9 = {
    "myocardial_infarction": ("410", "412"),
    "congestive_heart_failure": ("39891", "40201", "40211", "40291", "40401",
                                 "40403", "40411", "40413", "40491", "40493",
                                 "4254", "4255", "4257", "4258", "4259", "428"),
    "peripheral_vascular": ("0930", "4373", "440", "441", "4431", "4432",
                            "4438", "4439", "4471", "5571", "5579", "V434"),
    "cerebrovascular": ("36234", "430", "431", "432", "433", "434", "435",
                        "436", "437", "438"),
    "dementia": ("290", "2941", "3312"),
    "chronic_pulmonary": ("4168", "4169", "490", "491", "492", "493", "494",
                          "495", "496", "500", "501", "502", "503", "504",
                          "505", "5064", "5081", "5088"),
    "rheumatic": ("4465", "7100", "7101", "7102", "7103", "7104", "7140",
                  "7141", "7142", "7148", "725"),
    "peptic_ulcer": ("531", "532", "533", "534"),
    "mild_liver_disease": ("07022", "07023", "07032", "07033", "07044",
                           "07054", "0706", "0709", "570", "571", "5733",
                           "5734", "5738", "5739", "V427"),
    "diabetes_uncomplicated": ("2500", "2501", "2502", "2503", "2508", "2509"),
    "diabetes_complicated": ("2504", "2505", "2506", "2507"),
    "hemiplegia": ("3341", "342", "343", "3440", "3441", "3442", "3443",
                   "3444", "3445", "3446", "3449"),
    "renal_disease": ("40301", "40311", "40391", "40402", "40403", "40412",
                      "40413", "40492", "40493", "582", "5830", "5831",
                      "5832", "5834", "5836", "5837", "585", "586", "5880",
                      "V420", "V451", "V56"),
    "malignancy": ("140", "141", "142", "143", "144", "145", "146", "147",
                   "148", "149", "150", "151", "152", "153", "154", "155",
                   "156", "157", "158", "159", "160", "161", "162", "163",
                   "164", "165", "17", "18", "190", "191", "192", "193",
                   "194", "195", "200", "201", "202", "203", "204", "205",
                   "206", "207", "208", "2386"),
    "severe_liver_disease": ("4560", "4561", "4562", "5722", "5723", "5724",
                             "5728"),
    "metastatic_cancer": ("196", "197", "198", "199"),
    "hiv": ("042", "043", "044"),
}

CHARLSON_ICD10 = {
    "myocardial_infarction": ("I21", "I22", "I252"),
    "congestive_heart_failure": ("I099", "I110", "I130", "I132", "I255",
                                 "I420", "I425", "I426", "I427", "I428",
                                 "I429", "I43", "I50", "P290"),
    "peripheral_vascular": ("I70", "I71", "I731", "I738", "I739", "I771",
                            "I790", "I792", "K551", "K558", "K559", "Z958",
                            "Z959"),
    "cerebrovascular": ("G45", "G46", "H340", "I60", "I61", "I62", "I63",
                        "I64", "I65", "I66", "I67", "I68", "I69"),
    "dementia": ("F00", "F01", "F02", "F03", "F051", "G30", "G311"),
    "chronic_pulmonary": ("I278", "I279", "J40", "J41", "J42", "J43", "J44",
                          "J45", "J46", "J47", "J60", "J61", "J62", "J63",
                          "J64", "J65", "J66", "J67", "J684", "J701", "J703"),
    "rheumatic": ("M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353",
                  "M360"),
    "peptic_ulcer": ("K25", "K26", "K27", "K28"),
    "mild_liver_disease": ("B18", "K700", "K701", "K702", "K703", "K709",
                           "K713", "K714", "K715", "K717", "K73", "K74",
                           "K760", "K762", "K763", "K764", "K768", "K769",
                           "Z944"),
    "diabetes_uncomplicated": ("E100", "E101", "E106", "E108", "E109",
                               "E110", "E111", "E116", "E118", "E119",
                               "E120", "E121", "E126", "E128", "E129",
                               "E130", "E131", "E136", "E138", "E139",
                               "E140", "E141", "E146", "E148", "E149"),
    "diabetes_complicated": ("E102", "E103", "E104", "E105", "E107", "E112",
                             "E113", "E114", "E115", "E117", "E122", "E123",
                             "E124", "E125", "E127", "E132", "E133", "E134",
                             "E135", "E137", "E142", "E143", "E144", "E145",
                             "E147"),
    "hemiplegia": ("G041", "G114", "G801", "G802", "G81", "G82", "G830",
                   "G831", "G832", "G833", "G834", "G839"),
    "renal_disease": ("I120", "I131", "N032", "N033", "N034", "N035", "N036",
                      "N037", "N052", "N053", "N054", "N055", "N056", "N057",
                      "N18", "N19", "N250", "Z490", "Z491", "Z492", "Z940",
                      "Z992"),
    "malignancy": ("C0", "C1", "C2", "C30", "C31", "C32", "C33", "C34",
                   "C37", "C38", "C39", "C40", "C41", "C43", "C45", "C46",
                   "C47", "C48", "C49", "C50", "C51", "C52", "C53", "C54",
                   "C55", "C56", "C57", "C58", "C60", "C61", "C62", "C63",
                   "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71",
                   "C72", "C73", "C74", "C75", "C76", "C81", "C82", "C83",
                   "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94",
                   "C95", "C96", "C97"),
    "severe_liver_disease": ("I850", "I859", "I864", "I982", "K704", "K711",
                             "K721", "K729", "K765", "K766", "K767"),
    "metastatic_cancer": ("C77", "C78", "C79", "C80"),
    "hiv": ("B20", "B21", "B22", "B24"),
}

# Weights from the original Charlson index; used for the scalar summary score.
CHARLSON_WEIGHTS = {
    "myocardial_infarction": 1, "congestive_heart_failure": 1,
    "peripheral_vascular": 1, "cerebrovascular": 1, "dementia": 1,
    "chronic_pulmonary": 1, "rheumatic": 1, "peptic_ulcer": 1,
    "mild_liver_disease": 1, "diabetes_uncomplicated": 1,
    "diabetes_complicated": 2, "hemiplegia": 2, "renal_disease": 2,
    "malignancy": 2, "severe_liver_disease": 3, "metastatic_cancer": 6,
    "hiv": 6,
}

COMORBIDITIES = tuple(CHARLSON_WEIGHTS)

# ------------------------------------------------------- categorical binning
# MIMIC-IV race has ~30 levels, most of them sparse sub-categories. Collapsing
# to the OMB roll-up keeps the one-hot block small and the cell counts usable.
RACE_GROUPS = {
    "WHITE": "white", "BLACK": "black", "HISPANIC": "hispanic",
    "ASIAN": "asian", "AMERICAN INDIAN": "other", "NATIVE HAWAIIAN": "other",
    "OTHER": "other", "UNKNOWN": "unknown", "UNABLE TO OBTAIN": "unknown",
    "PATIENT DECLINED": "unknown", "PORTUGUESE": "other",
    "SOUTH AMERICAN": "hispanic", "MULTIPLE": "other",
}

ADMISSION_TYPES = (
    "EW EMER.", "OBSERVATION ADMIT", "URGENT", "EU OBSERVATION",
    "ELECTIVE", "SURGICAL SAME DAY ADMISSION", "DIRECT EMER.",
    "DIRECT OBSERVATION", "AMBULATORY OBSERVATION",
)

CARE_UNITS = (
    "Medical Intensive Care Unit (MICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Coronary Care Unit (CCU)",
    "Trauma SICU (TSICU)",
    "Neuro Intermediate", "Neuro Stepdown", "Neuro Surgical Intensive Care Unit (Neuro SICU)",
)


def race_group(value):
    """Collapse a raw MIMIC race string to its OMB roll-up."""
    if not isinstance(value, str):
        return "unknown"
    up = value.upper()
    for prefix, group in RACE_GROUPS.items():
        if up.startswith(prefix):
            return group
    return "other"


def charlson_groups(icd_code, icd_version):
    """Comorbidity groups a single ICD code belongs to (usually 0 or 1)."""
    code = str(icd_code).replace(".", "").strip().upper()
    table = CHARLSON_ICD9 if int(icd_version) == 9 else CHARLSON_ICD10
    return [g for g, prefixes in table.items() if code.startswith(prefixes)]
