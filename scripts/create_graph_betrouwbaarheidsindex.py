import pandas as pd
from pathlib import Path
from pandas import DataFrame
import scipy.stats as sct
from failure_paths.common.traject_normering import TrajectNormering
from failure_paths.common.graph_betrouwbaarheidsindex import GraphBetaValuesSingleInteractive


def _export_graph(
    df: DataFrame,
    export_dir: str,
    dijktraject: str,
    vak_section_pf_col: str = "Vak_Section_Pf",
    traject_pf_col: str = "Traject_Pf_bovengrens",
):
    df = df.rename(columns={"M_VAN": "m_start", "M_TOT": "m_end", "dijkvaknummer": "id"})
    df["beta"] = df[vak_section_pf_col].apply(lambda x: -1 * sct.norm.ppf(x))
    df_beta_vak = df[["id", "m_start", "m_end", "beta"]]

    beta_traject = -1 * sct.norm.ppf(df[traject_pf_col].iloc[0])
    print(f"Beta traject {dijktraject}: {beta_traject:.3f}")

    traject_normering = TrajectNormering(traject_id=dijktraject, norm_is_ondergrens=True)
    GraphBetaValuesSingleInteractive(
        traject_normering=traject_normering, df_beta_vak=df_beta_vak, beta_traject=beta_traject, export_dir=export_dir
    )


if __name__ == "__main__":
    # location of the excel file with the data:
    # DT16-1
    # fn = "C:/Users/SAKA/Downloads/tmp/DT16-1/99 Assemblage 16-1_result_2026-07-16_1328_effect_indirecte_mechanismen.xlsx"
    # DT16-2
    fn = "C:/Users/SAKA/Downloads/tmp/DT16-2/99 Assemblage 16-2_result_2026-06-28_1035_effect_indirecte_mechanismen.xlsx"

    # load the data from excel file:
    df = pd.read_excel(fn)

    # input for exporting the graph:
    export_dir = str(Path(fn).resolve().parent)
    dijktraject = "16-2"

    # without indirect mechanisms
    vak_section_pf_col = "Vak_Section_Pf"
    traject_pf_col = "Traject_Pf_bovengrens"

    # with indirect mechanisms
    # vak_section_pf_col = "Vak_Section_Pf_indirect"
    # traject_pf_col = "Traject_Pf_indirect"
        

    _export_graph(
        df=df,
        export_dir=export_dir,
        dijktraject=dijktraject,
        vak_section_pf_col=vak_section_pf_col,
        traject_pf_col=traject_pf_col,
    )
