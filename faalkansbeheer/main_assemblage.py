"""Script om faalkans dijktraject te berekenen."""

from pathlib import Path

import pandas as pd


def doorsnedekansen_inlezen(input: Path) -> pd.DataFrame:
    """Lees de doorsnedekansen in uit een Excel-bestand en sla ze op als CSV."""
    print(f"Lezen van doorsnedekansen uit {input}")
    df = pd.read_excel(input, sheet_name="componenten")

    return df


def bereken_vakkansen(df: pd.DataFrame, dL: float) -> pd.DataFrame:
    """Opschalen naar vakkansen."""
    from helper_functions import assemblage_functions as af

    df["N_vak"] = df.apply(
        lambda row: af.bepaal_N_vak(float(row["vak_lengte"]), float(row["a"]), dL),
        axis=1,
    )
    df["pf_vak"] = df["pf_dsn"] * df["N_vak"]

    return df


def bereken_faalkans_traject(df: pd.DataFrame):
    """Bereken de faalkans van het dijktraject."""
    from helper_functions import assemblage_functions as af

    pf_array = df["pf_vak"].to_numpy()
    pf_ondergrens_traject, pf_bovengrens_traject = af.combin_seriesysteem(pf_array)
    print(f"Ondergrens faalkans traject: {pf_ondergrens_traject:.2e}")
    print(f"Bovengrens faalkans traject: {pf_bovengrens_traject:.2e}")
    return pf_ondergrens_traject, pf_bovengrens_traject


def main():
    # Inlezen doorsnedekansen
    doorsnedekansen = doorsnedekansen_inlezen(input=input_file_full_path)
    # opschalen naar vakkansen
    df_vakkansen = bereken_vakkansen(doorsnedekansen, dL=50.0)
    # opslaan vakkansen
    # Rename filename from first underscore to "_vakkansen"
    base_name = input_file.split("_")[0] + "_vakkansen.xlsx"
    output_file = Path(output_dir, base_name)
    df_vakkansen.to_excel(output_file, index=False)
    print(f"Vakkansen opgeslagen in {output_file}")
    pf_ondergrens_traject, pf_bovengrens_traject = bereken_faalkans_traject(
        df_vakkansen
    )
    # TODO: grafieken maken van betrouwbaarheidsnivaus van doorsneden, vakken en traject
    # TODO: resultaten wegschrijven naar een geopackage.


if __name__ == "__main__":
    # set working directory to script location
    workspace = Path(__file__).parent.parent / "workspace"
    input_file = "DT16-1_dsn_kansen.xlsx"
    input_file_full_path = Path(workspace, "input", input_file)
    output_dir = Path(workspace, "output")
    output_dir.mkdir(exist_ok=True)

    main()
