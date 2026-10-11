import csv
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/raw/who_air_quality_2026.xlsx")

OUTPUT_FILE = Path(
    "data/processed/who_air_quality_parsed.csv"
)

MALFORMED_FILE = Path(
    "data/processed/who_air_quality_malformed.csv"
)

REPORT_FILE = Path(
    "data/processed/who_air_quality_parsing_report.csv"
)

SHEET_NAME = "GlobalCity_Db_V8.0_2026-06-28"


# ============================================================
# COLONNES ATTENDUES
# ============================================================

EXPECTED_COLUMNS = [
    "who_region",
    "iso3",
    "country_name",
    "city",
    "year",
    "version",
    "pm10_concentration",
    "pm25_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "pm25_tempcov",
    "no2_tempcov",
    "type_of_stations",
    "number_stations",
    "reference",
    "web_link",
    "population",
    "population_source",
    "latitude",
    "longitude",
    "who_ms",
]


# ============================================================
# LECTURE DU FICHIER EXCEL
# ============================================================

print("=" * 70)
print("LECTURE DU FICHIER WHO")
print("=" * 70)

raw = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
    header=None,
    dtype=str,
)

print(f"Dimensions Excel : {raw.shape}")

# Le fichier WHO contient les données CSV dans la première colonne.
cells = raw.iloc[:, 0].dropna().astype(str).tolist()

print(f"Cellules contenant des données : {len(cells)}")


# ============================================================
# PARSING
# ============================================================

parsed_rows = []
excluded_rows = []
parsing_report = []

header_detected = False

for excel_index, cell in enumerate(cells, start=1):

    try:
        values = next(csv.reader([cell]))
    except Exception as e:

        excluded_rows.append({
            "excel_row": excel_index,
            "column_count": -1,
            "reason": f"CSV parsing error: {e}",
            "raw_data": cell,
        })

        continue

    # --------------------------------------------------------
    # PREMIERE LIGNE = EN-TETE
    # --------------------------------------------------------

    if not header_detected:

        if values == EXPECTED_COLUMNS:
            header_detected = True

            print("\nEn-tête détecté :")

            for i, column in enumerate(values, start=1):
                print(f"{i:2d}. {column}")

            continue

    column_count = len(values)

    # --------------------------------------------------------
    # FORMAT COMPLET : 21 COLONNES
    # --------------------------------------------------------

    if column_count == len(EXPECTED_COLUMNS):

        row = values

        parsed_rows.append(row)

        parsing_report.append({
            "column_count": column_count,
            "status": "parsed_complete",
        })

        continue

    # --------------------------------------------------------
    # FORMATS PARTIELS EXPLOITABLES
    #
    # 13 colonnes :
    # jusqu'à type_of_stations
    #
    # 15 colonnes :
    # jusqu'à reference
    #
    # 16 colonnes :
    # jusqu'à web_link
    #
    # 18 colonnes :
    # jusqu'à population_source
    # --------------------------------------------------------

    if column_count in [13, 15, 16, 18]:

        missing_count = len(EXPECTED_COLUMNS) - column_count

        row = values + ["NA"] * missing_count

        parsed_rows.append(row)

        parsing_report.append({
            "column_count": column_count,
            "status": "parsed_partial",
        })

        continue

    # --------------------------------------------------------
    # FORMATS NON EXPLOITABLES
    #
    # 6 colonnes :
    # aucune concentration disponible
    #
    # 4 colonnes :
    # ligne tronquée
    # --------------------------------------------------------

    reason = "non_exploitable"

    if column_count == 6:
        reason = "insufficient_fields_no_pollution_measurements"

    elif column_count == 4:
        reason = "truncated_row"

    else:
        reason = "unexpected_column_count"

    excluded_rows.append({
        "excel_row": excel_index,
        "column_count": column_count,
        "reason": reason,
        "raw_data": cell,
    })

    parsing_report.append({
        "column_count": column_count,
        "status": "excluded",
    })


# ============================================================
# DATAFRAME FINAL
# ============================================================

df = pd.DataFrame(
    parsed_rows,
    columns=EXPECTED_COLUMNS,
)


# ============================================================
# NORMALISATION DES VALEURS MANQUANTES
# ============================================================

df = df.replace(
    {
        "NA": pd.NA,
        "": pd.NA,
        "N/A": pd.NA,
        "NULL": pd.NA,
    }
)


# ============================================================
# CONVERSION DES COLONNES NUMERIQUES
# ============================================================

numeric_columns = [
    "year",
    "pm10_concentration",
    "pm25_concentration",
    "no2_concentration",
    "pm10_tempcov",
    "pm25_tempcov",
    "no2_tempcov",
    "number_stations",
    "population",
    "latitude",
    "longitude",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# SAUVEGARDE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SAUVEGARDE DES LIGNES EXCLUES
# ============================================================

excluded_df = pd.DataFrame(excluded_rows)

if not excluded_df.empty:

    excluded_df.to_csv(
        MALFORMED_FILE,
        index=False,
    )


# ============================================================
# RAPPORT
# ============================================================

report_df = pd.DataFrame(parsing_report)

if not report_df.empty:

    report_summary = (
        report_df
        .groupby(["column_count", "status"])
        .size()
        .reset_index(name="count")
    )

else:

    report_summary = pd.DataFrame(
        columns=[
            "column_count",
            "status",
            "count",
        ]
    )


report_summary.to_csv(
    REPORT_FILE,
    index=False,
)


# ============================================================
# RAPPORT FINAL
# ============================================================

print("\n")
print("=" * 70)
print("RAPPORT DE PARSING")
print("=" * 70)

print(
    f"Lignes Excel analysées       : {len(cells):,}"
    .replace(",", " ")
)

print(
    f"Lignes correctement parsées : {len(df):,}"
    .replace(",", " ")
)

print(
    f"Lignes exclues              : {len(excluded_rows):,}"
    .replace(",", " ")
)

print("\nRépartition des lignes parsées :")

parsed_counts = (
    report_df[
        report_df["status"].isin(
            ["parsed_complete", "parsed_partial"]
        )
    ]["column_count"]
    .value_counts()
    .sort_index()
)

print(parsed_counts)


print("\nRépartition des lignes exclues :")

if excluded_rows:

    excluded_counts = (
        pd.DataFrame(excluded_rows)["column_count"]
        .value_counts()
        .sort_index()
    )

    print(excluded_counts)

else:

    print("Aucune")


print("\n")
print("=" * 70)
print("DATASET FINAL")
print("=" * 70)

print(f"Dimensions : {df.shape}")

print("\nColonnes :")
for column in df.columns:
    print(f"- {column}")

print("\nValeurs manquantes :")

missing = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing)


print("\nFichiers générés :")

print(f"- {OUTPUT_FILE}")
print(f"- {MALFORMED_FILE}")
print(f"- {REPORT_FILE}")

print("\nParsing terminé.")
