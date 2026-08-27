"""Dashboard for verified Spanish security and resilience data.

Run from the repository root with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
EXTERNAL_DATA_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports"
INTERNATIONAL_MISSIONS_MAP_PATH = ROOT / "app" / "assets" / "international_missions_map.png"
PERSONNEL_IMAGE_PATHS = {
    "active": ROOT / "app" / "assets" / "personnel_active.svg",
    "international": ROOT / "app" / "assets" / "personnel_international.svg",
    "reserve": ROOT / "app" / "assets" / "personnel_reserve.svg",
}

COUNTRY_ORDER = [
    "Spain",
    "Morocco",
    "France",
    "Germany",
    "Italy",
    "United Kingdom",
    "United States",
]
EVIDENCE_ORDER = [
    "Hecho confirmado",
    "Información oficial",
    "Información periodística",
    "Atribución",
    "Hipótesis",
    "Interpretación",
]
# ECB annual average for 2024: 1 EUR = 1.0824 USD.
# A fixed base-year conversion preserves the comparability of constant-price series.
EUR_PER_USD_2024 = 1 / 1.0824
CEUTA_MELILLA_REFERENCE_POINTS = pd.DataFrame(
    [
        {
            "location": "Ceuta",
            "latitude": 35.8894,
            "longitude": -5.3213,
            "description": "Ciudad autónoma española; punto de referencia geográfico.",
        },
        {
            "location": "Melilla",
            "latitude": 35.2923,
            "longitude": -2.9381,
            "description": "Ciudad autónoma española; punto de referencia geográfico.",
        },
    ]
)


def load_csv(filename: str, required_columns: set[str]) -> tuple[pd.DataFrame, str | None]:
    """Read a processed CSV and return an explanation if it cannot be charted."""
    path = PROCESSED_DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"No se encontró `data/processed/{filename}`."

    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return pd.DataFrame(), f"No se pudo leer `{filename}`: {error}"

    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return pd.DataFrame(), f"`{filename}` no contiene las columnas requeridas: {missing}."
    return frame, None


def processed_data_revision() -> tuple[tuple[str, int], ...]:
    """Create a cache key that changes whenever a processed CSV is updated."""
    return tuple(
        sorted(
            (path.name, path.stat().st_mtime_ns)
            for path in PROCESSED_DATA_DIR.glob("*.csv")
        )
    )


def external_data_revision() -> tuple[tuple[str, int], ...]:
    """Create a cache key that changes whenever an external CSV is updated."""
    return tuple(
        sorted(
            (path.name, path.stat().st_mtime_ns)
            for path in EXTERNAL_DATA_DIR.glob("*.csv")
        )
    )


@st.cache_data
def load_project_data(
    _data_revision: tuple[tuple[str, int], ...],
) -> dict[str, tuple[pd.DataFrame, str | None]]:
    """Load all optional, processed data sets used by the dashboard."""
    return {
        "military": load_csv(
            "military_expenditure.csv",
            {
                "country",
                "year",
                "military_expenditure_constant_usd",
                "military_expenditure_pct_gdp",
                "military_expenditure_per_capita_usd",
            },
        ),
        "arms": load_csv(
            "arms_transfers.csv",
            {"year", "supplier", "recipient", "sipri_tiv"},
        ),
        "macro": load_csv(
            "macro_economic.csv",
            {"country", "year", "gdp_current_usd", "population"},
        ),
        "border": load_csv(
            "border_irregular_arrivals.csv",
            {"period", "territory", "arrivals"},
        ),
        "events": load_csv(
            "security_events_timeline.csv",
            {"event_date", "event_title", "topic", "evidence_level", "source_url"},
        ),
    }


@st.cache_data
def load_sources() -> pd.DataFrame:
    """Load the project source register, which remains separate from the metrics."""
    source_path = EXTERNAL_DATA_DIR / "sources.csv"
    if not source_path.exists():
        return pd.DataFrame()
    return pd.read_csv(source_path)


@st.cache_data
def load_defence_programmes(
    external_revision: tuple[tuple[str, int], ...],
) -> tuple[pd.DataFrame, str | None]:
    """Load the programme catalogue without treating it as expenditure data."""
    path = EXTERNAL_DATA_DIR / "defence_programmes.csv"
    if not path.exists():
        return pd.DataFrame(), "No se encontró `data/external/defence_programmes.csv`."
    try:
        programmes = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return pd.DataFrame(), f"No se pudo leer `defence_programmes.csv`: {error}"

    required_columns = {
        "programa",
        "sistema_o_capacidad",
        "periodo_cobertura",
        "tipo_de_registro",
        "importe_eur",
        "estado_financiero",
        "fuente_id",
        "fuente_url",
        "resumen_modelo",
        "cantidad_aproximada",
        "ubicacion_publica",
        "contexto_fuente_url",
        "image_url",
        "image_attribution_url",
        "image_credit",
        "nota_de_lectura",
    }
    missing_columns = required_columns.difference(programmes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return pd.DataFrame(), f"`defence_programmes.csv` no contiene las columnas requeridas: {missing}."
    return programmes, None


@st.cache_data
def load_public_defence_equipment(
    external_revision: tuple[tuple[str, int], ...],
) -> tuple[pd.DataFrame, str | None]:
    """Load a public equipment catalogue without operational inventory details."""
    path = EXTERNAL_DATA_DIR / "public_defence_equipment.csv"
    if not path.exists():
        return pd.DataFrame(), "No se encontró `data/external/public_defence_equipment.csv`."
    try:
        equipment = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return pd.DataFrame(), f"No se pudo leer `public_defence_equipment.csv`: {error}"

    required_columns = {
        "service",
        "category",
        "system_or_family",
        "public_role",
        "public_status",
        "official_source_url",
        "accessed_on",
        "notes",
    }
    missing_columns = required_columns.difference(equipment.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return pd.DataFrame(), f"`public_defence_equipment.csv` no contiene las columnas requeridas: {missing}."
    return equipment, None


@st.cache_data
def load_external_catalog(
    external_revision: tuple[tuple[str, int], ...],
    filename: str,
    required_columns: frozenset[str],
) -> tuple[pd.DataFrame, str | None]:
    """Load a traceable external reference catalogue."""
    path = EXTERNAL_DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"No se encontró `data/external/{filename}`."
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        return pd.DataFrame(), f"No se pudo leer `{filename}`: {error}"

    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        return pd.DataFrame(), f"`{filename}` no contiene las columnas requeridas: {missing}."
    return frame, None


@st.cache_data
def load_presentation_script() -> str:
    script_path = REPORTS_DIR / "presentation_script.md"
    if not script_path.exists():
        return "El guion aún no está disponible."
    return script_path.read_text(encoding="utf-8")


def empty_data_message(error: str | None, source_ids: str) -> None:
    """Explain missing data without presenting a placeholder as a real result."""
    st.info(
        f"{error or 'No hay datos disponibles para este gráfico.'} "
        f"Consulta las fuentes {source_ids} en la pestaña **Fuentes** y guarda "
        "el CSV limpio con el esquema documentado en el README."
    )


def latest_by_country(military: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest available record for each country."""
    clean = military.copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean = clean.dropna(subset=["year", "country"])
    return clean.loc[clean.groupby("country")["year"].idxmax()].copy()


def format_eur(value: float) -> str:
    if pd.isna(value):
        return "No disponible"
    return f"EUR {value * EUR_PER_USD_2024 / 1_000_000_000:,.1f} mil millones"


def convert_usd_to_eur(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Create a display copy with monetary USD fields converted to base-year euros."""
    converted = frame.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce") * EUR_PER_USD_2024
    return converted


def render_overview(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    st.header("Visión general")
    st.markdown(
        "**Pregunta de investigación:** ¿está España preparada para el nuevo "
        "escenario de seguridad y para las amenazas que afectan a Europa y al Mediterráneo?"
    )
    st.caption(
        "El panel separa datos cuantitativos, hechos documentados, atribuciones e "
        "interpretaciones. La ausencia de datos no se interpreta como ausencia de riesgo."
    )

    military, military_error = data["military"]
    if military.empty:
        empty_data_message(military_error, "D01, D03 y D04")
        render_international_missions_map()
        render_ceuta_melilla_map(data)
        return

    latest = latest_by_country(military)
    spain = latest[latest["country"].eq("Spain")]
    countries = latest["country"].nunique()
    if spain.empty:
        st.warning("El CSV se cargó, pero no contiene el país `Spain`.")
        return

    spanish_record = spain.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Gasto militar de España", format_eur(spanish_record["military_expenditure_constant_usd"]))
    columns[1].metric("Gasto / PIB", f"{spanish_record['military_expenditure_pct_gdp']:.2f}%")
    columns[2].metric(
        "Gasto per cápita",
        f"EUR {spanish_record['military_expenditure_per_capita_usd'] * EUR_PER_USD_2024:,.0f}",
    )
    columns[3].metric("Países analizados", countries)
    st.caption(
        f"Último año disponible para España: {int(spanish_record['year'])}. "
        "Los importes se muestran en EUR aproximados usando el tipo medio EUR/USD de 2024; "
        "los CSV originales permanecen en USD."
    )
    render_international_missions_map()
    render_ceuta_melilla_map(data)


def render_international_missions_map() -> None:
    """Show the user-provided non-operational geographic missions overview."""
    st.divider()
    st.subheader("Misiones internacionales de las Fuerzas Armadas españolas")
    st.caption(
        "Mapa de referencia geográfica aportado para el proyecto. Los marcadores señalan países o "
        "zonas representadas en el mapa; no indican posiciones, número de efectivos, rutas, nivel de "
        "alerta ni actividad en tiempo real."
    )
    if not INTERNATIONAL_MISSIONS_MAP_PATH.exists():
        st.info("No se encontró la imagen del mapa de misiones internacionales.")
        return

    st.image(
        INTERNATIONAL_MISSIONS_MAP_PATH,
        width="stretch",
        caption="Mapa aportado por el proyecto para contextualizar las misiones internacionales.",
    )
    st.write(
        "España participa en misiones internacionales en el marco de organizaciones y acuerdos como "
        "Naciones Unidas, la Unión Europea y la OTAN. Este mapa sirve para situar visualmente esa "
        "dimensión exterior; la confirmación de cada misión, su mandato y su fecha debe consultarse "
        "en la fuente institucional correspondiente."
    )
    mission_summary, mission_summary_error = load_external_catalog(
        external_data_revision(),
        "international_missions_map_summary.csv",
        frozenset({"region", "country_or_mission", "mission", "period", "public_context", "framework", "source_url", "notes"}),
    )
    st.markdown("#### Misiones internacionales vigentes: resumen de 2026")
    st.warning(
        "Tabla formada solo por las entradas actuales aportadas y contrastadas con EMAD. EUTM RCA se ha "
        "eliminado por haber finalizado en 2024. Los efectivos por misión no se muestran porque requieren "
        "una publicación específica y fechada para cada caso."
    )
    if mission_summary.empty:
        st.info(mission_summary_error or "No hay resumen de misiones disponible.")
    else:
        region_labels = {
            "Europa": "🇪🇺 Europa",
            "Africa": "🌍 África",
            "Asia": "🌏 Asia",
            "America Central y del Sur": "🌎 América Central y del Sur",
            "Maritimo": "🌊 Marítimo",
        }
        mission_summary = mission_summary.copy()
        mission_summary["zona"] = mission_summary["region"].map(region_labels)
        st.dataframe(
            mission_summary.rename(
                columns={
                    "zona": "Zona",
                    "country_or_mission": "País o misión",
                    "mission": "Misión",
                    "period": "Desde - hasta",
                    "public_context": "Qué hacen",
                    "framework": "Marco",
                }
            )[["Zona", "País o misión", "Misión", "Desde - hasta", "Qué hacen", "Marco"]],
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Fuente de la estructura: Ejército de Tierra — Misiones Internacionales. "
            "[Abrir fuente oficial](https://ejercito.defensa.gob.es/misiones/)"
        )
        st.caption(
            "La tabla resume 15 entradas operativas; EMAD comunica 17 participaciones en total para 2026. "
            "La diferencia se debe a que esta tabla agrupa SNMG/SNMCMG en una sola fila y no desagrega "
            "todos los componentes de la actividad de Irak."
        )
        mission_totals, mission_totals_error = load_external_catalog(
            external_data_revision(),
            "international_mission_totals_2026.csv",
            frozenset({"framework", "missions", "description", "source_url", "source_date", "notes"}),
        )
        st.markdown("#### Operaciones en el exterior previstas para 2026")
        if mission_totals.empty:
            st.info(mission_totals_error or "No hay resumen de misiones disponible.")
        else:
            total = mission_totals.set_index("framework")
            total_column, un_column, coalition_column, _ = st.columns([0.9, 0.9, 1.1, 1.4])
            total_column.metric("Total de misiones", total.loc["Total", "missions"])
            un_column.metric("Bajo mandato ONU", total.loc["ONU", "missions"])
            coalition_column.metric("Coalición internacional", total.loc["Coalicion internacional", "missions"])
            st.caption(
                "Desglose del listado vigente: 8 contribuciones OTAN, 4 de la Unión Europea, 2 bajo "
                "mandato ONU, 1 de la coalición internacional en Irak y 2 en otros marcos."
            )
            st.dataframe(
                mission_totals.rename(
                    columns={
                        "framework": "Marco",
                        "missions": "Misiones",
                        "description": "Descripción",
                        "source_url": "Fuente EMAD",
                        "source_date": "Fecha de fuente",
                        "notes": "Nota",
                    }
                ),
                column_config={"Fuente EMAD": st.column_config.LinkColumn("Fuente EMAD", display_text="Abrir fuente")},
                width="stretch",
                hide_index=True,
            )
    render_leadership_context(external_data_revision())
    atlantic_26, atlantic_26_error = load_external_catalog(
        external_data_revision(),
        "atlantic_26_deployment.csv",
        frozenset({"indicator", "value", "scope", "source_url", "source_date", "notes"}),
    )
    st.markdown("#### Despliegue Atlántico 26: composición comunicada por EMAD")
    if atlantic_26.empty:
        st.info(atlantic_26_error or "No hay datos del despliegue disponibles.")
    else:
        atlantic_metrics = dict(zip(atlantic_26["indicator"], atlantic_26["value"]))
        ships, people, vehicles, aircraft, landing_craft = st.columns(5)
        ships.metric("Buques", atlantic_metrics["Buques"])
        people.metric("Personal", atlantic_metrics["Personal"])
        vehicles.metric("Vehículos", atlantic_metrics["Vehiculos"])
        aircraft.metric("Aeronaves", atlantic_metrics["Aeronaves"])
        landing_craft.metric("Embarcaciones anfibias", atlantic_metrics["Embarcaciones anfibias"])
        st.write(
            "**Buques comunicados:** Juan Carlos I, Castilla, Blas de Lezo, Reina Sofía y Patiño."
        )
        st.dataframe(
            atlantic_26.rename(
                columns={
                    "indicator": "Indicador",
                    "value": "Cantidad",
                    "scope": "Ámbito",
                    "source_url": "Fuente EMAD",
                    "source_date": "Fecha de publicación",
                    "notes": "Límite de interpretación",
                }
            ),
            column_config={"Fuente EMAD": st.column_config.LinkColumn("Fuente EMAD", display_text="Abrir fuente")},
            width="stretch",
            hide_index=True,
        )
    missions, missions_error = load_external_catalog(
        external_data_revision(),
        "international_missions.csv",
        frozenset({"mission_or_framework", "geographic_context", "public_role", "source_url", "source_status", "notes"}),
    )
    st.markdown("#### Fuentes institucionales para las misiones representadas")
    if missions.empty:
        st.info(missions_error or "No hay fuentes de misiones disponibles.")
        return
    st.dataframe(
        missions.rename(
            columns={
                "mission_or_framework": "Misión o marco",
                "geographic_context": "Contexto geográfico",
                "public_role": "Descripción pública",
                "source_url": "Fuente institucional",
                "source_status": "Estado de fuente",
                "notes": "Límite de interpretación",
            }
        ),
        column_config={
            "Fuente institucional": st.column_config.LinkColumn(
                "Fuente institucional",
                display_text="Abrir fuente",
            ),
        },
        width="stretch",
        hide_index=True,
    )


def render_leadership_context(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show time-sensitive senior leadership with dated institutional sources."""
    leadership, leadership_error = load_external_catalog(
        external_revision,
        "public_defence_leadership.csv",
        frozenset({"office", "officeholder", "rank", "status", "source_url", "source_date", "notes"}),
    )
    st.markdown("#### Mandos superiores confirmados en fuentes institucionales de 2026")
    if leadership.empty:
        st.info(leadership_error or "No hay mandos documentados todavía.")
        return
    st.dataframe(
        leadership.rename(
            columns={
                "office": "Cargo",
                "officeholder": "Titular",
                "rank": "Empleo",
                "status": "Estado",
                "source_url": "Fuente",
                "source_date": "Fecha de fuente",
                "notes": "Nota",
            }
        ),
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Los cargos son sensibles al tiempo: cada fila muestra la fecha de la fuente y debe "
        "verificarse de nuevo antes de reutilizarse fuera de ese corte."
    )


def render_personnel_context(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show personnel context, keeping estimated figures separate from official planning data."""
    official_personnel, official_error = load_external_catalog(
        external_revision,
        "public_defence_personnel.csv",
        frozenset({"indicator", "value", "reference_date", "status", "source_url", "source_date", "notes"}),
    )
    personnel_context, context_error = load_external_catalog(
        external_revision,
        "international_missions_personnel_context.csv",
        frozenset({"indicator", "approximate_value", "scope", "source_status", "source_url", "notes"}),
    )
    st.markdown("#### Personal militar y reservistas")
    st.warning(
        "Las cifras aproximadas se presentan como contexto orientativo, no como datos oficiales "
        "verificados de 2026. Se sustituirán por la estadística anual de personal cuando esté disponible."
    )
    if official_personnel.empty:
        st.info(official_error or "No hay datos oficiales de planificación de personal.")
    else:
        st.dataframe(
            official_personnel.rename(
                columns={
                    "indicator": "Indicador",
                    "value": "Dato",
                    "reference_date": "Referencia",
                    "status": "Estado",
                    "source_url": "Fuente",
                    "source_date": "Fecha de fuente",
                    "notes": "Nota",
                }
            ),
            column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
            width="stretch",
            hide_index=True,
        )
    if personnel_context.empty:
        st.info(context_error or "No hay contexto de personal disponible.")
        return

    active, deployed, national, _ = st.columns([0.9, 0.9, 0.9, 1.3])
    active.metric("Activos (aprox.)", "~117.000")
    deployed.metric("Exterior (aprox.)", "~3.000")
    national.metric("Nacional (aprox.)", "~114.000")
    st.dataframe(
        personnel_context.rename(
            columns={
                "indicator": "Indicador",
                "approximate_value": "Estimación",
                "scope": "Qué representa",
                "source_status": "Estado de fuente",
                "source_url": "Fuente para verificar",
                "notes": "Límite de interpretación",
            }
        ),
        column_config={"Fuente para verificar": st.column_config.LinkColumn("Fuente para verificar", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )
    visual = personnel_context[
        personnel_context["indicator"].isin(
            [
                "Militares de carrera en servicio activo",
                "Militares temporales en servicio activo",
                "Desplegados en misiones internacionales",
                "Reservistas voluntarios",
                "Reservistas de especial disponibilidad",
            ]
        )
    ].copy()
    visual["cantidad_numerica"] = visual["approximate_value"].str.replace("~", "", regex=False).astype(int)
    visual["imagen"] = visual["indicator"].map(
        {
            "Militares de carrera en servicio activo": PERSONNEL_IMAGE_PATHS["active"],
            "Militares temporales en servicio activo": PERSONNEL_IMAGE_PATHS["active"],
            "Desplegados en misiones internacionales": PERSONNEL_IMAGE_PATHS["international"],
            "Reservistas voluntarios": PERSONNEL_IMAGE_PATHS["reserve"],
            "Reservistas de especial disponibilidad": PERSONNEL_IMAGE_PATHS["reserve"],
        }
    )
    chart = px.bar(
        visual.sort_values("cantidad_numerica"),
        x="cantidad_numerica",
        y="indicator",
        orientation="h",
        color="indicator",
        text="approximate_value",
        title="Personal militar y reservistas: estimación orientativa",
    )
    chart.update_traces(textposition="outside", cliponaxis=False, width=0.58)
    chart.update_layout(
        showlegend=False,
        margin={"l": 0, "r": 70, "t": 50, "b": 0},
        height=390,
        xaxis={"rangemode": "tozero", "tickformat": ","},
        yaxis={"title": None},
    )
    st.plotly_chart(chart, width="stretch")
    columns = st.columns(len(visual))
    for column, record in zip(columns, visual.itertuples(index=False)):
        with column:
            st.image(record.imagen, width="stretch")
            st.caption(f"**{record.approximate_value}**\n\n{record.indicator}")


def render_ceuta_melilla_map(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Show a non-tactical geographic reference map for the border analysis."""
    st.divider()
    st.subheader("Mapa de situación — Ceuta")
    st.caption(
        "Mapa de referencia geográfica. No muestra posiciones de fuerzas de seguridad, "
        "rutas individuales ni información táctica."
    )

    border, border_error = data["border"]
    if border.empty:
        map_figure = px.scatter_map(
            CEUTA_MELILLA_REFERENCE_POINTS,
            lat="latitude",
            lon="longitude",
            hover_name="location",
            hover_data={"description": True, "latitude": False, "longitude": False},
            color="location",
            zoom=4.6,
            center={"lat": 35.6, "lon": -4.15},
            map_style="open-street-map",
            height=500,
            title="Ceuta y Melilla: localización de referencia",
        )
        map_figure.update_traces(marker={"size": 14})
        map_figure.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0}, showlegend=True)
        st.plotly_chart(map_figure, width="stretch")
        st.info(
            "Capa de datos operativos: pendiente. Se activará al incorporar "
            "`border_irregular_arrivals.csv` con balances verificables del Ministerio del Interior."
        )
    else:
        ceuta_melilla = border[border["territory"].isin(["Ceuta", "Melilla"])].copy()
        if ceuta_melilla.empty:
            st.info("El CSV de frontera cargado no contiene registros de Ceuta ni Melilla.")
        else:
            ceuta_melilla["arrivals"] = pd.to_numeric(ceuta_melilla["arrivals"], errors="coerce")
            ceuta_melilla = ceuta_melilla.dropna(subset=["arrivals"])
            period_column = "period_end" if "period_end" in ceuta_melilla.columns else "period"
            ceuta_melilla["_period_order"] = pd.to_datetime(
                ceuta_melilla[period_column],
                errors="coerce",
            )
            map_border = ceuta_melilla[ceuta_melilla["entry_route"].eq("Vía terrestre")]
            available_periods = (
                map_border.sort_values("_period_order", ascending=False)["period"]
                .drop_duplicates()
                .tolist()
            )
            selected_period = st.selectbox(
                "Periodo mostrado en el mapa",
                available_periods,
                help="Selecciona periodos con la misma fecha de corte para compararlos.",
            )
            selected_border = map_border[map_border["period"].eq(selected_period)]
            map_records = (
                selected_border.sort_values("_period_order")
                .merge(
                    CEUTA_MELILLA_REFERENCE_POINTS,
                    left_on="territory",
                    right_on="location",
                    how="left",
                )
            )
            map_figure = px.scatter_map(
                map_records,
                lat="latitude",
                lon="longitude",
                size="arrivals",
                color="territory",
                hover_name="territory",
                hover_data={
                    "arrivals": ":,",
                    "period": True,
                    "entry_route": True,
                    "latitude": False,
                    "longitude": False,
                    "location": False,
                    "_period_order": False,
                },
                size_max=58,
                zoom=4.6,
                center={"lat": 35.6, "lon": -4.15},
                map_style="open-street-map",
                height=500,
                title="Llegadas irregulares por vía terrestre: acumulado seleccionado",
                labels={
                    "arrivals": "Llegadas registradas",
                    "period": "Periodo acumulado",
                    "entry_route": "Vía",
                    "territory": "Ciudad",
                },
            )
            map_figure.update_layout(
                margin={"l": 0, "r": 0, "t": 45, "b": 0},
                legend_title_text="Ciudad",
            )
            st.plotly_chart(map_figure, width="stretch")
            selected_period_end = map_records[period_column].iloc[0]
            st.caption(
                f"Tamaño de cada marcador = llegadas registradas. Corte del periodo mostrado: {selected_period_end}. "
                "No representa intentos de entrada, interceptaciones ni rutas individuales."
            )
            st.caption(
                "La tabla siguiente incluye también la vía marítima, identificada en la columna "
                "**Vía de llegada**; esta no se representa en el mapa."
            )
            border_table = ceuta_melilla[ceuta_melilla["territory"].eq("Ceuta")][
                ["period", "territory", "entry_route", "arrivals", "source_document"]
            ].rename(
                columns={
                    "period": "Periodo",
                    "territory": "Territorio",
                    "entry_route": "Vía de llegada",
                    "arrivals": "Llegadas registradas",
                    "source_document": "Documento fuente",
                }
            )
            st.dataframe(
                border_table.sort_values(
                    ["Periodo", "Vía de llegada", "Territorio"],
                    ascending=[False, True, True],
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "La tabla muestra solo los registros de Ceuta incluidos en el CSV y conserva la "
                "definición de su fuente; no equivale a intentos o interceptaciones."
            )
    if border_error:
        st.caption(border_error)


def render_defence(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    st.header("Defensa de España")
    military, error = data["military"]
    if military.empty:
        empty_data_message(error, "D01 y D03")
        return

    spain = military[military["country"].eq("Spain")].copy()
    if spain.empty:
        st.warning("No hay registros de España en el conjunto cargado.")
        return
    spain["year"] = pd.to_numeric(spain["year"], errors="coerce")
    spain = spain.dropna(subset=["year"]).sort_values("year")
    spain = convert_usd_to_eur(
        spain,
        ["military_expenditure_constant_usd"],
    )

    chart = px.line(
        spain,
        x="year",
        y="military_expenditure_constant_usd",
        markers=True,
        labels={
            "year": "Año",
            "military_expenditure_constant_usd": "Gasto militar (EUR constantes aprox.)",
        },
        title="Gráfica 1 — Evolución del gasto militar de España (EUR constantes aprox.)",
    )
    st.plotly_chart(chart, width="stretch")

    ratio_chart = px.line(
        spain,
        x="year",
        y="military_expenditure_pct_gdp",
        markers=True,
        labels={"year": "Año", "military_expenditure_pct_gdp": "Gasto militar (% del PIB)"},
        title="Gráfica 2 — Esfuerzo de defensa de España (% del PIB)",
    )
    st.plotly_chart(ratio_chart, width="stretch")
    st.warning(
        "No se deben comparar directamente un presupuesto nacional y una serie SIPRI u OTAN: "
        "sus coberturas y definiciones pueden ser distintas."
    )
    st.subheader("Cómo interpretar el dinero destinado a defensa")
    st.caption(
        "Las gráficas anteriores usan la serie agregada de SIPRI. Para explicar qué se financia "
        "en programas concretos, presupuesto, adjudicación y ejecución miden fases distintas."
    )
    budget_column, award_column, execution_column = st.columns(3)
    with budget_column:
        st.markdown("**Presupuesto aprobado**")
        st.write(
            "Autoriza un máximo o previsión de gasto para un ejercicio. "
            "Permite afirmar que se planificó destinar una cantidad, no que se gastó."
        )
    with award_column:
        st.markdown("**Adjudicación**")
        st.write(
            "Asigna un contrato a una empresa y fija su importe. "
            "No prueba que todo el importe se haya pagado en ese año."
        )
    with execution_column:
        st.markdown("**Ejecución presupuestaria**")
        st.write(
            "Registra obligaciones reconocidas o pagos realizados durante el año. "
            "Es la medida más próxima a cuánto se gastó en ese ejercicio."
        )
    st.info(
        "Un mismo programa puede adjudicarse por un importe total y pagarse durante varios años. "
        "Por eso el catálogo no suma presupuestos, contratos y pagos como si fueran la misma cifra."
    )
    external_revision = external_data_revision()
    render_public_defence_context(external_revision)
    st.subheader("Programas y contratos: catálogo documental")
    st.caption(
        "Este catálogo identifica programas que requieren desglose documental para 2021-2025. "
        "No presenta importes como gasto hasta vincular cada cifra a presupuesto, ejecución o contrato."
    )
    programmes, programmes_error = load_defence_programmes(external_revision)
    if programmes.empty:
        st.info(programmes_error or "No hay programas documentados todavía.")
        return

    st.dataframe(
        programmes.rename(
            columns={
                "programa": "Programa",
                "sistema_o_capacidad": "Sistema o capacidad",
                "periodo_cobertura": "Periodo",
                "tipo_de_registro": "Tipo",
                "importe_eur": "Importe (EUR)",
                "estado_financiero": "Estado financiero",
                "fuente_id": "Fuente",
                "fuente_url": "Enlace oficial",
                "resumen_modelo": "Descripción",
                "cantidad_aproximada": "Cantidad aproximada",
                "ubicacion_publica": "Ubicación pública",
                "contexto_fuente_url": "Fuente de contexto",
                "image_url": "Imagen",
                "image_attribution_url": "Atribución de imagen",
                "image_credit": "Crédito de imagen",
                "nota_de_lectura": "Nota metodológica",
            }
        ),
        column_config={
            "Enlace oficial": st.column_config.LinkColumn("Enlace oficial", display_text="Abrir fuente"),
            "Fuente de contexto": st.column_config.LinkColumn(
                "Fuente de contexto",
                display_text="Abrir fuente",
            ),
            "Imagen": st.column_config.ImageColumn("Imagen", width="medium"),
            "Atribución de imagen": st.column_config.LinkColumn(
                "Atribución de imagen",
                display_text="Ver atribución",
            ),
            "Importe (EUR)": st.column_config.NumberColumn("Importe (EUR)", format="EUR %.0f"),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Las cantidades y ubicaciones son referencias públicas aproximadas, no inventario en tiempo real. "
        "Las imágenes se muestran solo cuando existe un enlace de atribución."
    )
    render_representative_navy_vessels(external_revision)
    st.subheader("Fichas visuales de las plataformas")
    for programme in programmes.itertuples(index=False):
        with st.container(border=True):
            image_column, details_column = st.columns([1, 2])
            with image_column:
                if pd.notna(programme.image_url) and programme.image_url:
                    st.image(programme.image_url, width=320)
                    if pd.notna(programme.image_attribution_url) and programme.image_attribution_url:
                        st.link_button("Ver atribución de la imagen", programme.image_attribution_url)
                else:
                    st.info("Imagen pendiente de una licencia reutilizable verificada.")
            with details_column:
                st.markdown(f"### {programme.programa}")
                st.write(programme.resumen_modelo)
                st.markdown(f"**Cantidad aproximada:** {programme.cantidad_aproximada}")
                st.markdown(f"**Ubicación pública:** {programme.ubicacion_publica}")
                st.link_button("Consultar fuente de contexto", programme.contexto_fuente_url)
                st.caption(programme.nota_de_lectura)
    st.markdown(
        "**Cómo completar cada importe:** usa D17 para el crédito presupuestado, D18 para obligaciones "
        "reconocidas o pagos y D19 para el importe adjudicado. Guarda el documento primario, la fecha y "
        "el identificador de expediente; no agregues estas tres medidas."
    )
    st.subheader("Equipos y capacidades de referencia del Ejército de Tierra")
    st.caption(
        "Catálogo de equipos publicados por fuentes institucionales. No es un inventario: no informa "
        "de existencias, disponibilidad, munición, asignación por unidad ni ubicación operativa."
    )
    equipment, equipment_error = load_public_defence_equipment(external_revision)
    if equipment.empty:
        st.info(equipment_error or "No hay equipos documentados todavía.")
        return

    categories = sorted(equipment["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Categorías de equipo",
        categories,
        default=categories,
        key="defence_equipment_categories",
    )
    selected_equipment = equipment[equipment["category"].isin(selected_categories)]
    st.dataframe(
        selected_equipment.rename(
            columns={
                "service": "Ejército",
                "category": "Categoría",
                "system_or_family": "Sistema o familia",
                "public_role": "Función pública",
                "public_status": "Estado descrito",
                "official_source_url": "Fuente oficial",
                "accessed_on": "Consultado",
                "notes": "Límite de interpretación",
            }
        ),
        column_config={
            "Fuente oficial": st.column_config.LinkColumn(
                "Fuente oficial",
                display_text="Abrir catálogo",
            ),
        },
        width="stretch",
        hide_index=True,
    )

    render_spain_cyber_breakdown(data)


def render_pegasus_events_table(events: pd.DataFrame) -> None:
    """Shared renderer for the Pegasus-related rows of the events timeline."""
    st.dataframe(
        events.sort_values("event_date").rename(
            columns={
                "event_date": "Fecha",
                "event_title": "Hecho",
                "topic": "Tema",
                "evidence_level": "Nivel de evidencia",
                "source_url": "Fuente",
                "notes": "Nota",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
    )


def render_spain_cyber_breakdown(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Show cyberattacks given and received involving Spain, integrated into its defence tab."""
    st.divider()
    st.subheader("Ciberseguridad: ataques dados y recibidos")
    st.info(
        "**Criterio de evidencia:** la atribución de un ciberataque a un Estado o actor "
        "solo se muestra como hecho si existe evidencia pública concluyente o resolución competente. "
        "Una atribución periodística o de un informe técnico no es una condena judicial."
    )
    events, error = data["events"]
    pegasus = events[events["topic"].eq("Pegasus")] if not events.empty else pd.DataFrame()
    if pegasus.empty:
        empty_data_message(error, "Registro documental de Pegasus")
        return

    given_col, received_col = st.columns(2)
    with given_col:
        st.markdown("#### 🇪🇸 Ataques dados (España como origen señalado)")
        st.markdown(
            "El caso **CatalanGate** (informe de Citizen Lab, abril de 2022) identificó hasta 65 "
            "personas del entorno independentista catalán con indicios de infección por Pegasus o "
            "Candiru entre 2017 y 2020. El CNI admitió ante el Parlamento, en mayo de 2022, haber "
            "espiado con autorización judicial a una parte de esos casos; su directora, Paz Esteban, "
            "fue cesada poco después."
        )
    with received_col:
        st.markdown("#### 🎯 Ataques recibidos (España como objetivo señalado)")
        st.markdown(
            "El Pegasus Project (Forbidden Stories, julio de 2021) incluyó números de dirigentes "
            "internacionales en una lista de posible interés para clientes de NSO Group, y señaló a "
            "Marruecos entre los presuntos clientes. Marruecos niega categóricamente haber usado "
            "Pegasus contra España, Francia o cualquier otro país; no existe una condena judicial "
            "que confirme esta atribución."
        )
    st.subheader("Cronología documental")
    render_pegasus_events_table(pegasus)
    st.caption(
        "Distingue expresamente entre lo admitido oficialmente (CNI, mayo de 2022), lo señalado por "
        "informes técnicos independientes (Citizen Lab) y lo meramente atribuido por prensa sin "
        "confirmación judicial (el caso marroquí)."
    )


def render_morocco_defence(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Show Morocco's verified aggregate military-expenditure indicators."""
    st.header("Defensa de Marruecos")
    military, error = data["military"]
    if military.empty:
        empty_data_message(error, "D01")
        return

    morocco = military[military["country"].eq("Morocco")].copy()
    if morocco.empty:
        st.warning("No hay registros de Marruecos en el conjunto cargado.")
        return

    morocco["year"] = pd.to_numeric(morocco["year"], errors="coerce")
    morocco = morocco.dropna(subset=["year"]).sort_values("year")
    morocco = convert_usd_to_eur(
        morocco,
        [
            "military_expenditure_constant_usd",
            "military_expenditure_per_capita_usd",
        ],
    )
    latest = morocco.iloc[-1]
    expenditure, effort, per_capita = st.columns(3)
    expenditure.metric(
        "Gasto militar",
        format_eur(latest["military_expenditure_constant_usd"]),
    )
    effort.metric(
        "Gasto / PIB",
        (
            f"{latest['military_expenditure_pct_gdp']:.2f}%"
            if pd.notna(latest["military_expenditure_pct_gdp"])
            else "No disponible"
        ),
    )
    per_capita.metric(
        "Gasto per cápita",
        (
            f"EUR {latest['military_expenditure_per_capita_usd']:,.0f}"
            if pd.notna(latest["military_expenditure_per_capita_usd"])
            else "No disponible"
        ),
    )
    st.caption(
        f"Último año disponible para Marruecos: {int(latest['year'])}. "
        "Los importes se muestran en EUR aproximados usando el tipo medio EUR/USD de 2024; "
        "los CSV originales permanecen en USD."
    )

    expenditure_chart = px.line(
        morocco,
        x="year",
        y="military_expenditure_constant_usd",
        markers=True,
        labels={
            "year": "Año",
            "military_expenditure_constant_usd": "Gasto militar (EUR constantes aprox.)",
        },
        title="Evolución del gasto militar de Marruecos (EUR constantes aprox.)",
    )
    st.plotly_chart(expenditure_chart, width="stretch")

    effort_chart = px.line(
        morocco,
        x="year",
        y="military_expenditure_pct_gdp",
        markers=True,
        labels={"year": "Año", "military_expenditure_pct_gdp": "Gasto militar (% del PIB)"},
        title="Esfuerzo de defensa de Marruecos (% del PIB)",
    )
    st.plotly_chart(effort_chart, width="stretch")
    st.warning(
        "La serie SIPRI mide gasto militar agregado; por sí sola no representa la capacidad "
        "militar completa ni debe equipararse directamente a presupuestos nacionales."
    )

    external_revision = external_data_revision()
    render_morocco_public_defence_context(external_revision)
    render_morocco_navy_vessels(external_revision)
    render_morocco_equipment_catalog(external_revision)
    render_morocco_cyber_breakdown(data)


def render_morocco_leadership_context(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show Morocco's senior defence leadership with dated encyclopaedic sources."""
    leadership, leadership_error = load_external_catalog(
        external_revision,
        "public_defence_leadership_morocco.csv",
        frozenset({"office", "officeholder", "rank", "status", "source_url", "source_date", "notes"}),
    )
    st.markdown("#### Mandos superiores confirmados en fuentes enciclopédicas")
    if leadership.empty:
        st.info(leadership_error or "No hay mandos documentados todavía.")
        return
    st.dataframe(
        leadership.rename(
            columns={
                "office": "Cargo",
                "officeholder": "Titular",
                "rank": "Empleo",
                "status": "Estado",
                "source_url": "Fuente",
                "source_date": "Fecha de fuente",
                "notes": "Nota",
            }
        ),
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Marruecos no publica un portal de defensa equivalente a defensa.gob.es: estas fuentes son "
        "referencias enciclopédicas que citan The Military Balance (IISS) y deben verificarse de nuevo "
        "antes de reutilizarse fuera de esta fecha de consulta."
    )


def render_morocco_personnel_context(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show Morocco's personnel estimates, all traced to IISS Military Balance via a secondary source."""
    personnel, personnel_error = load_external_catalog(
        external_revision,
        "public_defence_personnel_morocco.csv",
        frozenset({"indicator", "value", "reference_date", "status", "source_url", "source_date", "notes"}),
    )
    st.markdown("#### Personal militar")
    st.warning(
        "Estas cifras proceden de The Military Balance (IISS) citado por una fuente enciclopédica, no de una "
        "estadística oficial marroquí publicada. Deben sustituirse por una fuente primaria cuando esté disponible."
    )
    if personnel.empty:
        st.info(personnel_error or "No hay datos de personal disponibles.")
        return
    st.dataframe(
        personnel.rename(
            columns={
                "indicator": "Indicador",
                "value": "Dato",
                "reference_date": "Referencia",
                "status": "Estado",
                "source_url": "Fuente",
                "source_date": "Fecha de fuente",
                "notes": "Nota",
            }
        ),
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )


def render_morocco_installations_map(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Render only public, non-operational context for Moroccan defence installations."""
    installations, installations_error = load_external_catalog(
        external_revision,
        "public_defence_installations_morocco.csv",
        frozenset(
            {
                "name",
                "service",
                "city",
                "region",
                "latitude",
                "longitude",
                "public_role",
                "source_url",
                "source_date",
                "notes",
            }
        ),
    )
    st.markdown("#### Instalaciones: contexto público")
    st.caption(
        "El mapa usa municipios de referencia y misiones institucionales generales. No muestra "
        "posiciones precisas, disponibilidad, nivel de alerta, inventario, rutas ni despliegues."
    )
    if installations.empty:
        st.info(installations_error or "No hay instalaciones documentadas todavía.")
        return

    map_figure = px.scatter_map(
        installations,
        lat="latitude",
        lon="longitude",
        color="service",
        hover_name="name",
        hover_data={
            "city": True,
            "region": True,
            "public_role": True,
            "source_date": True,
            "latitude": False,
            "longitude": False,
            "source_url": False,
            "notes": False,
        },
        zoom=4.4,
        center={"lat": 31.5, "lon": -7.5},
        map_style="open-street-map",
        height=500,
        title="Instalaciones con referencia pública de las Fuerzas Armadas Reales",
    )
    map_figure.update_traces(marker={"size": 13})
    map_figure.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0})
    st.plotly_chart(map_figure, width="stretch")
    st.dataframe(
        installations.rename(
            columns={
                "name": "Instalación",
                "service": "Organización",
                "city": "Municipio",
                "region": "Región",
                "public_role": "Misión pública general",
                "source_url": "Fuente",
                "source_date": "Fecha de fuente",
                "notes": "Límite de interpretación",
            }
        )[
            [
                "Instalación",
                "Organización",
                "Municipio",
                "Región",
                "Misión pública general",
                "Fuente",
                "Fecha de fuente",
                "Límite de interpretación",
            ]
        ],
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "La base de Dajla se sitúa en el Sáhara Occidental, territorio no autónomo según Naciones Unidas "
        "y en disputa; su inclusión es un punto de referencia geográfico, no una posición táctica."
    )


def render_morocco_public_defence_context(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Group leadership, installations and personnel context for Morocco, mirroring Spain's layout."""
    st.subheader("Instalaciones, personal y mandos: contexto público")
    render_morocco_installations_map(external_revision)
    render_morocco_leadership_context(external_revision)
    render_morocco_personnel_context(external_revision)


def render_morocco_navy_vessels(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show publicly described Royal Moroccan Navy vessel classes without operational detail."""
    vessels, vessels_error = load_external_catalog(
        external_revision,
        "representative_navy_vessels_morocco.csv",
        frozenset({"vessel_or_class", "type", "public_role", "highlight", "source_url", "notes"}),
    )
    st.subheader("Buques representativos de la Armada Real")
    st.caption(
        "Selección de clases documentadas en fuentes enciclopédicas. No refleja ubicación, "
        "disponibilidad ni misión actual de ninguna unidad."
    )
    if vessels.empty:
        st.info(vessels_error or "No hay buques documentados todavía.")
        return
    st.dataframe(
        vessels.rename(
            columns={
                "vessel_or_class": "Buque o clase",
                "type": "Tipo",
                "public_role": "Función principal",
                "highlight": "Destacado",
                "source_url": "Fuente",
                "notes": "Nota",
            }
        ),
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )


def render_morocco_equipment_catalog(external_revision: tuple[tuple[str, int], ...]) -> None:
    """Show a public equipment catalogue for Morocco without operational inventory details."""
    st.subheader("Equipos de referencia del Ejército de Tierra y las Fuerzas Reales Aéreas")
    st.caption(
        "Catálogo de familias de equipo documentadas en fuentes enciclopédicas y de prensa "
        "especializada. No es un inventario: no informa de existencias, disponibilidad ni ubicación operativa."
    )
    equipment, equipment_error = load_external_catalog(
        external_revision,
        "public_defence_equipment_morocco.csv",
        frozenset(
            {
                "service",
                "category",
                "system_or_family",
                "public_role",
                "public_status",
                "official_source_url",
                "accessed_on",
                "notes",
            }
        ),
    )
    if equipment.empty:
        st.info(equipment_error or "No hay equipos documentados todavía.")
        return

    categories = sorted(equipment["category"].dropna().unique())
    selected_categories = st.multiselect(
        "Categorías de equipo",
        categories,
        default=categories,
        key="morocco_equipment_categories",
    )
    selected_equipment = equipment[equipment["category"].isin(selected_categories)]
    st.dataframe(
        selected_equipment.rename(
            columns={
                "service": "Rama",
                "category": "Categoría",
                "system_or_family": "Sistema o familia",
                "public_role": "Función pública",
                "public_status": "Estado descrito",
                "official_source_url": "Fuente",
                "accessed_on": "Consultado",
                "notes": "Límite de interpretación",
            }
        ),
        column_config={
            "Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente"),
        },
        width="stretch",
        hide_index=True,
    )
    st.warning(
        "Marruecos no publica un catálogo oficial de material equivalente al del Ejército de Tierra "
        "español: estas familias proceden de Wikipedia (a su vez con cita a EDA, UNROCA, SIPRI Trade "
        "Registers e IISS) y de prensa especializada que reporta aprobaciones de venta de EE. UU. (DSCA). "
        "Una aprobación de venta no equivale a una entrega ni a unidades en servicio."
    )


def render_morocco_cyber_breakdown(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Show cyberattacks given and received involving Morocco, integrated into its defence tab."""
    st.divider()
    st.subheader("Ciberseguridad: ataques dados y recibidos")
    st.info(
        "**Criterio de evidencia:** la atribución de un ciberataque a un Estado o actor "
        "solo se muestra como hecho si existe evidencia pública concluyente o resolución competente. "
        "Marruecos niega categóricamente las acusaciones descritas a continuación."
    )
    events, error = data["events"]
    pegasus = events[events["topic"].eq("Pegasus")] if not events.empty else pd.DataFrame()
    if pegasus.empty:
        empty_data_message(error, "Registro documental de Pegasus")
        return

    given_col, received_col = st.columns(2)
    with given_col:
        st.markdown("#### 🎯 Ataques dados (Marruecos como origen señalado)")
        st.markdown(
            "El Pegasus Project (Forbidden Stories, julio de 2021) señaló a Marruecos entre los "
            "presuntos clientes de NSO Group tras aparecer números de dirigentes internacionales "
            "—incluidos los de España y Francia— en una lista filtrada de posible interés. "
            "**Marruecos niega categóricamente** haber usado Pegasus contra estos países; no existe "
            "una condena judicial que confirme la atribución, solo indicios periodísticos."
        )
    with received_col:
        st.markdown("#### 🇲🇦 Ataques recibidos")
        st.info(
            "No se ha localizado en fuentes verificables un registro público y trazable de "
            "ciberataques recibidos por Marruecos, equivalente en detalle al caso CatalanGate "
            "en España. Este apartado permanece pendiente de una fuente primaria marroquí o "
            "internacional que lo documente."
        )
    st.subheader("Cronología documental (compartida con la pestaña de España)")
    render_pegasus_events_table(pegasus)
    st.caption(
        "La misma cronología de hechos vinculados a Pegasus se muestra en ambas pestañas de defensa "
        "porque el caso conecta a España y Marruecos: solo cambia el ángulo narrativo (origen vs. objetivo)."
    )


def render_representative_navy_vessels(
    external_revision: tuple[tuple[str, int], ...],
) -> None:
    """Show publicly described Navy vessels without operational detail."""
    vessels, vessels_error = load_external_catalog(
        external_revision,
        "representative_navy_vessels.csv",
        frozenset({"vessel_or_class", "type", "public_role", "highlight", "source_url", "notes"}),
    )
    st.subheader("Buques representativos de la Armada")
    st.caption(
        "Selección de plataformas destacadas por su función pública. Las imágenes se añadirán "
        "posteriormente; esta tabla no refleja ubicación, disponibilidad ni misión actual."
    )
    if vessels.empty:
        st.info(vessels_error or "No hay buques documentados todavía.")
        return
    st.dataframe(
        vessels.rename(
            columns={
                "vessel_or_class": "Buque o clase",
                "type": "Tipo",
                "public_role": "Función principal",
                "highlight": "Destacado",
                "source_url": "Fuente Armada",
                "notes": "Nota",
            }
        ),
        column_config={"Fuente Armada": st.column_config.LinkColumn("Fuente Armada", display_text="Abrir fuente")},
        width="stretch",
        hide_index=True,
    )
    st.info(
        "El Juan Carlos I es el mayor buque construido en España y actúa como plataforma "
        "multipropósito. En el Despliegue Atlántico 26 participaron Juan Carlos I, Castilla, "
        "Blas de Lezo, Reina Sofía y Patiño."
    )


def render_individual_equipment_profiles(
    external_revision: tuple[tuple[str, int], ...],
) -> None:
    """Show user-provided individual equipment ranges before vehicle platform cards."""
    equipment, equipment_error = load_external_catalog(
        external_revision,
        "individual_equipment_profiles.csv",
        frozenset({"profile", "category", "equipment_or_model", "price_range_eur", "source_status", "notes"}),
    )
    st.subheader("Equipo individual: perfiles y precios orientativos")
    st.warning(
        "Estas cinco tablas proceden de estimaciones aportadas para el proyecto. Sus rangos no son "
        "precios oficiales de Defensa, adjudicaciones ni costes de inventario."
    )
    if equipment.empty:
        st.info(equipment_error or "No hay perfiles de equipo disponibles.")
        return
    profile_labels = {
        "Perfil con fusil G36": "🇪🇸 1. Infantería — Fusilero",
        "Perfil ametrallador": "🔥 2. Ametrallador ligero",
        "Perfil tirador de precision": "🎯 3. Tirador de precisión",
        "Perfil paracaidista": "🪂 4. Paracaidista — BRIPAC",
        "Perfil con fusil G36E": "🟢 5. La Legión — Fusilero",
    }
    profile_order = list(profile_labels)
    common_rows = pd.DataFrame(
        [
            ("Pistola", "HK USP - Heckler & Koch (según puesto)", "~600"),
            ("Cargador pistola", "HK USP - 15 cartuchos", "~20-40"),
            ("Munición pistola", "9x19 mm", "~0.30-0.80 por cartucho"),
            ("Casco", "COBAT 01 / COBAT GTH - FECSA", "~60-200"),
            ("Chaleco", "FECSA - protección balística", "~944"),
            ("Uniforme", "Uniforme de campaña", "~42-100"),
            ("Camisetas", "Ropa interior militar", "~10-20"),
            ("Ropa interior", "Prenda interior", "~10-20"),
            ("Calcetines", "Calcetines militares", "~5-15"),
            ("Botas", "ITURRI / FAL / Robusta", "~100-180"),
            ("Gafas", "Protección ocular", "~30-100"),
            ("Guantes", "Guantes de combate", "~20-50"),
            ("Radio", "PR4G V3 / PNR-500", "No publicado"),
            ("Baterías", "Equipos electrónicos", "~20-100"),
            ("Linterna", "Equipo de iluminación", "~20-60"),
            ("Botiquín", "Material sanitario", "~50-150"),
            ("Higiene", "Neceser/material de aseo", "~20-50"),
            ("Hidratación", "Sistema de hidratación", "~20-60"),
            ("Ración", "RIC - Ración Individual de Combate, 24 h", "No publicado"),
        ],
        columns=["category", "equipment_or_model", "price_range_eur"],
    )
    for profile in profile_order:
        profile_rows = equipment[equipment["profile"].eq(profile)]
        if profile_rows.empty:
            continue
        common_for_profile = common_rows.copy()
        if profile == "Perfil con fusil G36":
            common_for_profile.loc[common_for_profile["category"].eq("Uniforme"), "equipment_or_model"] = (
                "Uniforme de campaña pixelado"
            )
            common_for_profile.loc[common_for_profile["category"].eq("Botiquín"), "equipment_or_model"] = (
                "Material sanitario individual"
            )
            common_for_profile.loc[common_for_profile["category"].eq("Hidratación"), "equipment_or_model"] = (
                "Cantimplora/sistema de hidratación"
            )
        if profile == "Perfil paracaidista":
            common_for_profile.loc[common_for_profile["category"].eq("Botiquín"), "equipment_or_model"] = (
                "Material sanitario"
            )
        common_for_profile["profile"] = profile
        common_for_profile["source_status"] = "Estimacion aportada para el proyecto"
        common_for_profile["notes"] = "Precio no oficial; no representa contrato publico."
        profile_rows = pd.concat([profile_rows, common_for_profile], ignore_index=True)
        with st.expander(profile_labels[profile], expanded=False):
            st.dataframe(
                profile_rows.rename(
                    columns={
                        "category": "Categoría",
                        "equipment_or_model": "Equipo / modelo / marca",
                        "price_range_eur": "Precio aprox.",
                        "source_status": "Estado de fuente",
                        "notes": "Nota",
                    }
                )[
                    [
                        "Categoría",
                        "Equipo / modelo / marca",
                        "Precio aprox.",
                        "Estado de fuente",
                        "Nota",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )


def render_public_defence_context(
    external_revision: tuple[tuple[str, int], ...],
) -> None:
    """Render only public, non-operational context for Spanish defence installations."""
    st.subheader("Instalaciones, personal y mandos: contexto público")
    st.caption(
        "El mapa usa municipios de referencia y misiones institucionales generales. No muestra "
        "posiciones precisas, disponibilidad, nivel de alerta, inventario, rutas ni despliegues."
    )
    installations, installations_error = load_external_catalog(
        external_revision,
        "public_defence_installations.csv",
        frozenset(
            {
                "name",
                "service",
                "city",
                "province",
                "latitude",
                "longitude",
                "public_role",
                "source_url",
                "source_date",
                "notes",
            }
        ),
    )
    if installations.empty:
        st.info(installations_error or "No hay instalaciones documentadas todavía.")
    else:
        map_figure = px.scatter_map(
            installations,
            lat="latitude",
            lon="longitude",
            color="service",
            hover_name="name",
            hover_data={
                "city": True,
                "province": True,
                "public_role": True,
                "source_date": True,
                "latitude": False,
                "longitude": False,
                "source_url": False,
                "notes": False,
            },
            zoom=4.2,
            center={"lat": 39.6, "lon": -3.7},
            map_style="open-street-map",
            height=500,
            title="Instalaciones y sedes institucionales con referencia pública",
        )
        map_figure.update_traces(marker={"size": 13})
        map_figure.update_layout(margin={"l": 0, "r": 0, "t": 45, "b": 0})
        st.plotly_chart(map_figure, width="stretch")
        st.dataframe(
            installations.rename(
                columns={
                    "name": "Instalación",
                    "service": "Organización",
                    "city": "Municipio",
                    "province": "Provincia",
                    "public_role": "Misión pública general",
                    "source_url": "Fuente oficial",
                    "source_date": "Fecha de fuente",
                    "notes": "Límite de interpretación",
                }
            )[
                [
                    "Instalación",
                    "Organización",
                    "Municipio",
                    "Provincia",
                    "Misión pública general",
                    "Fuente oficial",
                    "Fecha de fuente",
                    "Límite de interpretación",
                ]
            ],
            column_config={
                "Fuente oficial": st.column_config.LinkColumn(
                    "Fuente oficial",
                    display_text="Abrir fuente",
                ),
            },
            width="stretch",
            hide_index=True,
        )

    render_personnel_context(external_revision)

    render_individual_equipment_profiles(external_revision)
    capabilities, capabilities_error = load_external_catalog(
        external_revision,
        "public_defence_capabilities.csv",
        frozenset(
            {
                "category",
                "system",
                "public_quantity",
                "status",
                "public_context",
                "source_url",
                "source_date",
                "notes",
            }
        ),
    )
    st.markdown("#### Actualizaciones públicas de capacidades")
    if capabilities.empty:
        st.info(capabilities_error or "No hay actualizaciones de capacidades documentadas todavía.")
    else:
        st.dataframe(
            capabilities.rename(
                columns={
                    "category": "Categoría",
                    "system": "Sistema",
                    "public_quantity": "Cantidad pública",
                    "status": "Estado",
                    "public_context": "Contexto",
                    "source_url": "Fuente",
                    "source_date": "Fecha de fuente",
                    "notes": "Límite de interpretación",
                }
            ),
            column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "La tabla separa expresamente unidades previstas, primeras fases contratadas y sistemas "
            "publicados. Ninguna de esas categorías equivale por sí sola a disponibilidad operativa."
        )


def render_spain_morocco_head_to_head(military: pd.DataFrame) -> None:
    """Show a dedicated Spain vs Morocco comparison: the project's main geopolitical pairing."""
    st.subheader("España vs Marruecos: cara a cara")
    st.caption(
        "Comparación directa de los dos países centrales del proyecto. Usa la misma serie SIPRI "
        "que el resto del panel; no mezcla metodologías OTAN ni presupuestos nacionales."
    )
    both = military[military["country"].isin(["Spain", "Morocco"])].copy()
    if both.empty:
        st.info("No hay registros de España ni Marruecos en el conjunto cargado.")
        return
    both["year"] = pd.to_numeric(both["year"], errors="coerce")
    both = both.dropna(subset=["year"]).sort_values("year")
    both = convert_usd_to_eur(
        both,
        ["military_expenditure_constant_usd", "military_expenditure_per_capita_usd"],
    )
    latest_both = latest_by_country(military)
    latest_both = convert_usd_to_eur(
        latest_both,
        ["military_expenditure_constant_usd", "military_expenditure_per_capita_usd"],
    )
    latest_both = latest_both[latest_both["country"].isin(["Spain", "Morocco"])]
    spain_row = latest_both[latest_both["country"].eq("Spain")]
    morocco_row = latest_both[latest_both["country"].eq("Morocco")]

    spain_col, morocco_col = st.columns(2)
    with spain_col:
        st.markdown("#### 🇪🇸 España")
        if spain_row.empty:
            st.info("Sin registro de España.")
        else:
            record = spain_row.iloc[0]
            st.metric("Gasto militar", format_eur(record["military_expenditure_constant_usd"]))
            st.metric(
                "Gasto / PIB",
                f"{record['military_expenditure_pct_gdp']:.2f}%" if pd.notna(record["military_expenditure_pct_gdp"]) else "No disponible",
            )
            st.metric(
                "Gasto per cápita",
                f"EUR {record['military_expenditure_per_capita_usd']:,.0f}" if pd.notna(record["military_expenditure_per_capita_usd"]) else "No disponible",
            )
            st.caption(f"Último año disponible: {int(record['year'])}.")
    with morocco_col:
        st.markdown("#### 🇲🇦 Marruecos")
        if morocco_row.empty:
            st.info("Sin registro de Marruecos.")
        else:
            record = morocco_row.iloc[0]
            st.metric("Gasto militar", format_eur(record["military_expenditure_constant_usd"]))
            st.metric(
                "Gasto / PIB",
                f"{record['military_expenditure_pct_gdp']:.2f}%" if pd.notna(record["military_expenditure_pct_gdp"]) else "No disponible",
            )
            st.metric(
                "Gasto per cápita",
                f"EUR {record['military_expenditure_per_capita_usd']:,.0f}" if pd.notna(record["military_expenditure_per_capita_usd"]) else "No disponible",
            )
            st.caption(f"Último año disponible: {int(record['year'])}.")

    expenditure_col, effort_col = st.columns(2)
    with expenditure_col:
        expenditure_chart = px.line(
            both,
            x="year",
            y="military_expenditure_constant_usd",
            color="country",
            markers=True,
            labels={
                "year": "Año",
                "military_expenditure_constant_usd": "Gasto militar (EUR constantes aprox.)",
                "country": "País",
            },
            title="Gasto militar: España vs Marruecos",
        )
        st.plotly_chart(expenditure_chart, width="stretch")
    with effort_col:
        effort_chart = px.line(
            both,
            x="year",
            y="military_expenditure_pct_gdp",
            color="country",
            markers=True,
            labels={"year": "Año", "military_expenditure_pct_gdp": "Gasto militar (% del PIB)", "country": "País"},
            title="Esfuerzo de defensa (% del PIB): España vs Marruecos",
        )
        st.plotly_chart(effort_chart, width="stretch")

    per_capita_chart = px.bar(
        latest_both.sort_values("country"),
        x="country",
        y="military_expenditure_per_capita_usd",
        color="country",
        labels={
            "country": "País",
            "military_expenditure_per_capita_usd": "Gasto militar per cápita (EUR aprox.)",
        },
        title="Gasto militar per cápita: España vs Marruecos (último año disponible de cada país)",
    )
    st.plotly_chart(per_capita_chart, width="stretch")
    st.warning(
        "España y Marruecos pueden tener el último año disponible en fechas distintas en la serie SIPRI; "
        "verifica el año de cada barra antes de citar la comparación per cápita como simultánea."
    )
    st.divider()


def render_comparison(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    st.header("Comparación internacional")
    military, error = data["military"]
    if military.empty:
        empty_data_message(error, "D01 y D04")
        return

    render_spain_morocco_head_to_head(military)

    latest = latest_by_country(military)
    latest = convert_usd_to_eur(
        latest,
        ["military_expenditure_constant_usd", "military_expenditure_per_capita_usd"],
    )
    st.subheader("Comparación multipaís")
    available = [country for country in COUNTRY_ORDER if country in set(latest["country"])]
    chosen = st.multiselect("Países", available, default=available)
    selected = latest[latest["country"].isin(chosen)].copy()
    if selected.empty:
        st.info("Selecciona al menos un país para comparar.")
        return

    ranking = px.bar(
        selected.sort_values("military_expenditure_pct_gdp", ascending=False),
        x="country",
        y="military_expenditure_pct_gdp",
        color="country",
        labels={"country": "País", "military_expenditure_pct_gdp": "Gasto militar (% del PIB)"},
        title="Gráfica 3 — Gasto militar como porcentaje del PIB por país",
    )
    st.plotly_chart(ranking, width="stretch")

    macro, macro_error = data["macro"]
    comparable = military.merge(macro, on=["country", "year"], how="inner")
    comparable = latest_by_country(comparable)
    comparable = convert_usd_to_eur(
        comparable,
        ["gdp_current_usd", "military_expenditure_constant_usd"],
    )
    comparable = comparable[comparable["country"].isin(chosen)]
    if comparable.empty:
        empty_data_message(macro_error, "D01 y D04")
        return
    scatter = px.scatter(
        comparable,
        x="gdp_current_usd",
        y="military_expenditure_constant_usd",
        color="country",
        size="population",
        hover_data=["year", "military_expenditure_pct_gdp"],
        labels={
            "gdp_current_usd": "PIB (EUR aprox., conversión fija de 2024)",
            "military_expenditure_constant_usd": "Gasto militar (EUR constantes aprox. de 2024)",
            "population": "Población",
        },
        title="Gráfica 4 — PIB, población y gasto militar (EUR aprox.)",
    )
    st.plotly_chart(scatter, width="stretch")
    st.caption(
        "El tamaño representa la población. Los importes se convierten con un tipo fijo de 2024; "
        "el gráfico compara indicadores económicos, no capacidad militar total."
    )


def render_abraham_accords_alliance(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Explain Morocco's Abraham Accords alignment and its arms-supply consequences."""
    st.header("Alianza — Acuerdos de Abraham")
    st.info(
        "Esta sección diferencia acuerdos anunciados, contratos, entregas y despliegues "
        "verificados. Una afirmación de un fabricante o de prensa no prueba por sí sola la "
        "operatividad de un sistema."
    )
    st.subheader("Cronología: normalización con Israel y giro sobre el Sáhara Occidental")
    events, error = data["events"]
    alliance_events = events[events["topic"].eq("Alianza")] if not events.empty else pd.DataFrame()
    if alliance_events.empty:
        empty_data_message(error, "Registro documental de la alianza")
    else:
        st.dataframe(
            alliance_events.sort_values("event_date").rename(
                columns={
                    "event_date": "Fecha",
                    "event_title": "Hecho",
                    "topic": "Tema",
                    "evidence_level": "Nivel de evidencia",
                    "source_url": "Fuente",
                    "notes": "Nota",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        )
    st.markdown(
        "El 10 de diciembre de 2020, Marruecos se convirtió en el cuarto país árabe en normalizar "
        "relaciones con Israel dentro de los Acuerdos de Abraham. A cambio, Estados Unidos reconoció "
        "la soberanía marroquí sobre el Sáhara Occidental mediante una proclamación presidencial "
        "(24 de diciembre de 2020), un paso que **España, la Unión Europea y la ONU no han "
        "reconocido** de la misma forma. El 14 de marzo de 2022, Pedro Sánchez giró la posición "
        "histórica de España al respaldar el plan de autonomía marroquí en una carta al rey "
        "Mohammed VI, una decisión unilateral criticada por su propio socio de gobierno."
    )
    st.warning(
        "El Tribunal de Justicia de la UE ha resuelto en varias ocasiones (2016, 2018, 2021) que los "
        "acuerdos comerciales UE-Marruecos no se aplican al Sáhara Occidental sin el consentimiento "
        "de su población. La postura española de 2022 no representa la posición común de la UE."
    )

    st.subheader("Cooperación militar Israel-Marruecos y rearme de EE. UU.")
    st.markdown(
        "Marruecos e Israel firmaron su primer acuerdo de defensa bilateral en noviembre de 2021, "
        "seguido en 2023 de memorandos de cooperación en aeronáutica, inteligencia artificial, "
        "seguridad militar y ciberseguridad. En paralelo, Estados Unidos aprobó ventas de armamento "
        "de gran volumen a Marruecos por la Agencia de Cooperación en Seguridad de Defensa (DSCA), "
        "incluida la aprobación de 2019 por 4.250 millones de USD para helicópteros de ataque "
        "AH-64E Apache."
    )
    st.caption(
        "Sistemas específicos reportados por prensa especializada (drones Blue Bird/ThunderB, "
        "sistema antidrones Skylock, defensa aérea Barak MX) no han podido verificarse de forma "
        "directa en fuentes primarias en este proyecto; deben tratarse como reportados, no confirmados, "
        "hasta contrastarlos con Defense News, Janes o Africa Intelligence."
    )

    arms, arms_error = data["arms"]
    if arms.empty:
        empty_data_message(arms_error, "D02")
        return

    focused = arms[
        (arms["recipient"].eq("Morocco")) | (arms["supplier"].eq("Israel"))
    ].copy()
    if focused.empty:
        st.info("No hay transferencias de Marruecos o Israel en el archivo cargado.")
        return
    chart = px.bar(
        focused.groupby(["year", "supplier"], as_index=False)["sipri_tiv"].sum(),
        x="year",
        y="sipri_tiv",
        color="supplier",
        labels={"year": "Año", "sipri_tiv": "Valor SIPRI TIV", "supplier": "Proveedor"},
        title="Transferencias de armas relacionadas con Marruecos o Israel",
    )
    st.plotly_chart(chart, width="stretch")
    st.caption("El TIV de SIPRI no es un precio contractual ni un presupuesto militar.")


def render_spain_nato(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    """Expand Spain's NATO role beyond Rota: Madrid Summit, KFOR, and 2% GDP commitment."""
    st.header("España-OTAN")
    st.success(
        "Hecho confirmado — OTAN: en la Cumbre de Madrid de 2022 se acordó ampliar de 4 a 6 los "
        "destructores estadounidenses AEGIS con capacidad BMD en la Base Naval de Rota, con 600 "
        "militares adicionales; el acuerdo de extensión se firmó en mayo de 2023."
    )
    stages = [
        "España",
        "Base Naval de Rota",
        "4 buques AEGIS (2015-2022)",
        "6 buques AEGIS (desde 2023)",
        "Defensa antimisiles de la OTAN",
    ]
    figure = go.Figure(
        go.Sankey(
            node={"label": stages, "pad": 30, "thickness": 25},
            link={"source": [0, 1, 1, 2, 3], "target": [1, 2, 3, 4, 4], "value": [1, 1, 1, 1, 1]},
        )
    )
    figure.update_layout(
        title="Papel documentado de Rota dentro de la arquitectura BMD de la OTAN",
        height=380,
    )
    st.plotly_chart(figure, width="stretch")

    st.subheader("Cronología: compromisos, Rota y la Cumbre de Madrid")
    events, error = data["events"]
    nato_events = events[events["topic"].eq("OTAN")] if not events.empty else pd.DataFrame()
    if nato_events.empty:
        empty_data_message(error, "Registro documental OTAN")
    else:
        st.dataframe(
            nato_events.sort_values("event_date").rename(
                columns={
                    "event_date": "Fecha",
                    "event_title": "Hecho",
                    "topic": "Tema",
                    "evidence_level": "Nivel de evidencia",
                    "source_url": "Fuente",
                    "notes": "Nota",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        )

    st.subheader("Compromiso del 2% del PIB")
    st.markdown(
        "España firmó en la Cumbre de Gales (2014) el compromiso de acercar su gasto en defensa "
        "al 2% del PIB junto al resto de aliados; en ese momento España gastaba en torno al 0,9% "
        "del PIB. Es históricamente uno de los aliados con menor porcentaje de gasto respecto al PIB "
        "dentro de la OTAN. La Cumbre de Madrid (2022) reiteró el objetivo y en el proyecto se maneja "
        "un horizonte de 2029 para varios aliados rezagados, en línea con el objetivo de planificación "
        "de efectivos ya documentado en la pestaña de Defensa de España."
    )
    st.warning(
        "Las cifras exactas de cumplimiento anual de España deben verificarse en las estadísticas de "
        "gasto en defensa de la OTAN (nato.int) y en los presupuestos del Ministerio de Defensa; "
        "este proyecto no fija una cifra concreta sin esa fuente primaria."
    )

    st.subheader("Otras contribuciones de España a la OTAN")
    contributions = pd.DataFrame(
        [
            (
                "KFOR (Kosovo)",
                "España contribuye desde el inicio de la misión en junio de 1999, bajo la resolución 1244 de la ONU. KFOR cuenta actualmente con unos 4.500 efectivos de aliados y socios.",
            ),
            (
                "Rota — buques AEGIS BMD",
                "4 destructores desde 2015 (anunciados por el secretario de Defensa de EE. UU., Leon Panetta, en 2011); ampliación a 6 acordada en la Cumbre de Madrid de 2022 y firmada en mayo de 2023.",
            ),
            (
                "Operation Sea Guardian",
                "Participación con activos navales y aéreos en el Mediterráneo, operación activa desde noviembre de 2016.",
            ),
            (
                "Policía Aérea del Báltico / flanco Este",
                "Despliegues de caza españoles reportados en misiones de vigilancia aérea aliada; verificar despliegue vigente en defensa.gob.es antes de citarlo como activo.",
            ),
        ],
        columns=["Contribución", "Descripción"],
    )
    st.dataframe(contributions, width="stretch", hide_index=True)
    st.caption(
        "Fuentes: NATO — Ballistic Missile Defence (nato.int), Wikipedia (Naval Station Rota, "
        "2022 Madrid NATO summit) y páginas de operaciones de la OTAN (nato.int/en/what-we-do/"
        "operations-and-missions)."
    )


def render_international_alliance_comparison(
    data: dict[str, tuple[pd.DataFrame, str | None]],
) -> None:
    """Compare Spain's 2026 missions across international frameworks."""
    st.header("Comparación alianzas internacionales")
    st.info(
        "Contraste de la participación española por marco internacional. El número de misiones "
        "no mide efectivos, presupuesto ni capacidad militar."
    )
    totals, totals_error = load_external_catalog(
        external_data_revision(),
        "international_mission_totals_2026.csv",
        frozenset({"framework", "missions", "description", "source_url", "source_date", "notes"}),
    )
    if totals.empty:
        empty_data_message(totals_error, "Fuente institucional EMAD")
        return

    totals = totals.copy()
    totals["missions"] = pd.to_numeric(totals["missions"], errors="coerce")
    totals = totals.dropna(subset=["missions"])
    total_row = totals[totals["framework"].eq("Total")]
    frameworks = totals[~totals["framework"].eq("Total")].copy()
    if not total_row.empty:
        st.metric("Misiones españolas previstas para 2026", int(total_row.iloc[0]["missions"]))

    st.subheader("Desglose por alianza o marco")
    chart = px.bar(
        frameworks.sort_values("missions"),
        x="missions",
        y="framework",
        orientation="h",
        text="missions",
        labels={"missions": "Misiones españolas", "framework": "Marco internacional"},
        title="Participación de España por marco internacional en 2026",
    )
    chart.update_traces(textposition="outside")
    st.plotly_chart(chart, width="stretch")

    display_totals = frameworks.rename(
        columns={
            "framework": "Marco internacional",
            "missions": "Misiones de España",
            "description": "Qué representa el dato",
            "source_date": "Fecha de la fuente",
            "notes": "Nota de contraste",
            "source_url": "Fuente",
        }
    )
    st.dataframe(
        display_totals[
            ["Marco internacional", "Misiones de España", "Qué representa el dato",
             "Fecha de la fuente", "Nota de contraste", "Fuente"]
        ],
        width="stretch",
        hide_index=True,
        column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
    )

    st.subheader("España frente a los marcos internacionales")
    comparison = pd.DataFrame(
        [
            ("España", "Contribuyente", "Aporta personal, medios y mandatos a operaciones concretas.", "17 misiones previstas"),
            ("OTAN", "Alianza de defensa colectiva", "Coordina contribuciones aliadas y operaciones de la Alianza.", "8 misiones identificadas"),
            ("Unión Europea", "Marco político y operativo europeo", "Incluye misiones militares de la UE, incluida Atalanta.", "4 misiones identificadas"),
            ("ONU", "Mandato internacional", "Agrupa las contribuciones bajo mandato de Naciones Unidas.", "2 misiones identificadas"),
            ("Coalición internacional", "Cooperación específica", "Agrupa la contribución a la operación de apoyo a Irak.", "1 misión identificada"),
        ],
        columns=["Actor o marco", "Tipo", "Papel contrastado", "Indicador disponible"],
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.warning(
        "El total de 17 es un recuento del listado EMAD, no 17 despliegues independientes. "
        "Algunas actividades se agrupan y otras no se desagregan por componentes. No permite "
        "concluir qué alianza es más fuerte."
    )
    st.caption(
        "No se comparan presupuestos o efectivos de las alianzas porque este proyecto no ofrece "
        "una serie homogénea para esos indicadores."
    )


def render_risk_matrix(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    st.header("Escenarios de riesgo")
    st.warning(
        "Matriz analítica creada para este proyecto. Las puntuaciones no son probabilidades "
        "oficiales ni predicciones; sirven para explicar una metodología transparente."
    )
    scenarios = pd.DataFrame(
        [
            ("Ciberataque a servicios", 4, 4),
            ("Corte eléctrico prolongado", 3, 5),
            ("Caída de comunicaciones", 3, 4),
            ("Desinformación", 5, 3),
            ("Fenómeno meteorológico extremo", 4, 4),
            ("Problemas de suministro", 2, 4),
            ("Emergencia fronteriza / crisis migratoria masiva", 3, 3),
            ("Evacuación local", 2, 5),
            ("Conflicto militar directo", 1, 5),
        ],
        columns=["Escenario", "Probabilidad analítica", "Impacto analítico"],
    )
    scenarios["Riesgo"] = scenarios["Probabilidad analítica"] * scenarios["Impacto analítico"]
    scenarios["Nivel"] = pd.cut(
        scenarios["Riesgo"],
        bins=[0, 4, 9, 16, np.inf],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
    )
    chart = px.scatter(
        scenarios,
        x="Probabilidad analítica",
        y="Impacto analítico",
        size="Riesgo",
        color="Nivel",
        text="Escenario",
        range_x=[0.5, 5.5],
        range_y=[0.5, 5.5],
        title="Matriz de riesgo experimental",
    )
    chart.update_traces(textposition="top center")
    st.plotly_chart(chart, width="stretch")
    st.dataframe(scenarios, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Caso documentado: crisis migratoria masiva (Ceuta, mayo de 2021)")
    st.caption(
        "Este caso ilustra el escenario 'Emergencia fronteriza / crisis migratoria masiva' con hechos "
        "verificados, no una simulación. Distingue lo confirmado por fuentes oficiales de lo meramente "
        "reportado o no verificado en esta investigación."
    )
    events, error = data["events"]
    border_events = events[events["topic"].eq("Frontera")] if not events.empty else pd.DataFrame()
    if border_events.empty:
        empty_data_message(error, "Registro documental de la frontera")
    else:
        st.dataframe(
            border_events.sort_values("event_date").rename(
                columns={
                    "event_date": "Fecha",
                    "event_title": "Hecho",
                    "topic": "Tema",
                    "evidence_level": "Nivel de evidencia",
                    "source_url": "Fuente",
                    "notes": "Nota",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={"Fuente": st.column_config.LinkColumn("Fuente", display_text="Abrir fuente")},
        )
    st.markdown(
        "**Cómo se desencadenó (mayo de 2021):** una disputa diplomática (la hospitalización en España "
        "del líder del Frente Polisario, Brahim Ghali) coincidió con la retirada de controles fronterizos "
        "marroquíes. En 72 horas cruzaron a Ceuta unas 10.000 personas, muchas a nado, entre ellas miles "
        "de menores. Human Rights Watch documentó devoluciones sumarias inmediatas, incluidas de menores "
        "no acompañados, hasta que el Defensor del Pueblo y un tribunal local ordenaron su cese."
    )
    st.markdown(
        "**El papel de la desinformación:** según fuentes periodísticas (sin una URL primaria de "
        "verificador que se haya podido confirmar en esta investigación), circularon mensajes de audio "
        "por WhatsApp y Telegram en darija y francés afirmando que la frontera estaba abierta, lo que "
        "actuó como efecto llamada. En la tragedia de la valla de Melilla (24 de junio de 2022), un "
        "ultimátum de 24 horas de las autoridades marroquíes a personas migrantes asentadas en los "
        "montes de Nador tuvo un efecto similar: una señal ambigua sobre una 'oportunidad' de cruce que "
        "impulsó un intento masivo y coordinado, con al menos 23 muertos confirmados."
    )
    st.warning(
        "**Lección para la ciudadanía:** un mensaje viral sobre 'fronteras abiertas' o 'permisos "
        "especiales' que no proceda de un canal oficial (112, Delegación del Gobierno, Ministerio del "
        "Interior o de Exteriores) debe tratarse como no verificado. Verifica siempre en la fuente "
        "oficial antes de actuar o reenviar, y consulta www.maldita.es o www.newtral.es para contrastar bulos."
    )


def render_civil_preparedness() -> None:
    st.header("Preparación civil")
    st.warning(
        "Índice experimental desarrollado para este proyecto. No es una clasificación oficial "
        "del Gobierno de España y no sustituye las instrucciones de Protección Civil o del 112."
    )
    items = [
        "Agua disponible",
        "Alimentación básica",
        "Iluminación alternativa",
        "Energía / carga de dispositivos",
        "Comunicaciones alternativas",
        "Power bank cargado",
        "Documentación accesible",
        "Dinero o medios de pago alternativos",
        "Medicación habitual",
        "Plan de movilidad",
        "Canales de información oficial",
        "Plan familiar de contacto",
    ]
    selected = [item for item in items if st.checkbox(item, key=f"preparedness_{item}")]
    score = round(100 * len(selected) / len(items))
    if score <= 25:
        level = "Preparación baja"
    elif score <= 50:
        level = "Preparación básica"
    elif score <= 75:
        level = "Preparación media"
    else:
        level = "Preparación alta"

    st.subheader("Índice experimental de preparación civil")
    st.metric("Índice experimental de preparación civil", f"{score}/100", level)
    st.progress(score)
    st.caption("Cada elemento tiene el mismo peso para mantener el modelo comprensible y modificable en clase.")
    st.markdown(
        "Para recomendaciones vigentes, consulta [Protección Civil](https://www.proteccioncivil.es/), "
        "[INCIBE](https://www.incibe.es/) y los servicios 112 de tu comunidad autónoma."
    )

    st.divider()
    st.subheader("Cuánto almacenar: kit de emergencia por persona")
    st.caption(
        "Referencia orientativa inspirada en campañas europeas de autoprotección (por ejemplo, la "
        "recomendación de '72 horas' de autonomía difundida por distintas agencias de protección civil "
        "europeas, y guías nórdicas como 'Si viene una crisis o una guerra'). No es una tabla oficial "
        "española; verifica siempre la recomendación vigente en proteccioncivil.es antes de actuar."
    )
    kit = pd.DataFrame(
        [
            ("Agua potable", "2 litros por persona y día", "Mínimo 3 días (72 h); ideal 7 días si hay espacio."),
            ("Alimentos no perecederos", "Aporte calórico básico por persona y día", "Conservas, frutos secos, barritas; rota el stock antes de la caducidad."),
            ("Botiquín básico", "1 por hogar", "Incluye medicación habitual de cada miembro de la familia, con margen de varios días."),
            ("Linterna y pilas / power bank", "1 por persona", "Evita velas por riesgo de incendio en interiores."),
            ("Radio con pilas o manivela", "1 por hogar", "Para recibir avisos oficiales si falla la cobertura móvil."),
            ("Documentación e efectivo", "Copias y una cantidad pequeña en metálico", "Copias de DNI/pasaporte y tarjeta sanitaria; el pago electrónico puede fallar sin suministro eléctrico."),
            ("Silbato", "1 por persona", "Señal acústica de auxilio que gasta menos energía que gritar."),
        ],
        columns=["Elemento", "Cantidad orientativa", "Nota"],
    )
    st.dataframe(kit, width="stretch", hide_index=True)

    st.subheader("Principios oficiales de evacuación")
    st.markdown(
        "- **No te autoevacúes sin indicación oficial** salvo peligro inmediato y evidente: sigue los "
        "canales de Protección Civil, 112 y ayuntamiento; una evacuación desordenada puede ser más "
        "peligrosa que quedarse en un lugar seguro conocido.\n"
        "- **Conoce el plan de emergencia de tu municipio** (puntos de encuentro, refugios, rutas "
        "señalizadas) antes de que ocurra cualquier suceso; consúltalo en la web de tu ayuntamiento.\n"
        "- **Ten un plan familiar de contacto** con un punto de encuentro y un contacto fuera de la zona, "
        "por si las redes locales se saturan.\n"
        "- **Lleva encima solo lo esencial**: documentación, medicación, agua, algo de comida y el móvil "
        "cargado; prioriza moverte rápido y ligero sobre cargar equipaje.\n"
        "- **Verifica antes de compartir**: un mensaje viral sobre una ruta, una frontera o un refugio "
        "que no proceda de una fuente oficial puede ser un bulo; contrástalo en el 112 o en verificadores "
        "como Maldita.es o Newtral.es antes de actuar o reenviarlo."
    )

    st.subheader("Orientación básica sin GPS: el método de la sombra")
    st.markdown(
        "Si te quedas sin batería o cobertura y necesitas saber dónde está el norte aproximado:\n"
        "1. Clava un palo recto y vertical en el suelo, en una zona despejada de sol.\n"
        "2. Marca con una piedra el extremo de la sombra que proyecta.\n"
        "3. Espera entre 10 y 15 minutos: la sombra se habrá desplazado. Marca el nuevo extremo.\n"
        "4. Traza una línea recta entre ambas marcas: esa línea señala aproximadamente **este-oeste** "
        "(la primera marca queda al oeste, la segunda al este en el hemisferio norte).\n"
        "5. Una línea perpendicular a esa marca indica el eje **norte-sur**.\n\n"
        "Alternativa con reloj analógico (hemisferio norte): apunta la aguja de las horas hacia el sol; "
        "la bisectriz entre esa aguja y las 12 señala aproximadamente el sur."
    )
    st.caption(
        "Es una técnica general de orientación al aire libre, útil ante un simple corte de suministro o "
        "pérdida de cobertura; no sustituye un GPS, una brújula ni la señalización oficial de evacuación."
    )


def render_sources(data: dict[str, tuple[pd.DataFrame, str | None]]) -> None:
    st.header("Fuentes y trazabilidad")
    st.markdown(
        "Cada gráfico debe poder rastrearse hasta un dataset, documento o comunicado. "
        "Registra la fecha de consulta cada vez que se actualicen datos."
    )
    sources = load_sources()
    if sources.empty:
        st.warning("No se encontró el registro de fuentes.")
        return
    st.dataframe(
        sources,
        width="stretch",
        hide_index=True,
        column_config={"url": st.column_config.LinkColumn("URL")},
    )
    st.divider()
    st.subheader("CSV utilizados: datos, cálculos y conclusiones")
    st.caption(
        "Este resumen explica qué se ha utilizado realmente en cada archivo. "
        "Los CSV pendientes se muestran como tales: no generan conclusiones ni gráficos hasta disponer de datos verificables."
    )

    csv_summary = [
        {
            "key": "military",
            "filename": "military_expenditure.csv",
            "description": "Serie internacional de gasto militar de SIPRI.",
            "data_used": (
                "`country`, `year`, gasto militar en USD constantes de 2024, "
                "gasto militar como % del PIB y gasto per cápita."
            ),
            "calculations": (
                "Conversión del gasto SIPRI desde millones a USD; conversión de la cuota del PIB "
                "de proporción a porcentaje; selección del último año disponible por país."
            ),
            "conclusion": (
                "Permite describir la evolución del gasto de España y compararla con los países seleccionados. "
                "No mide por sí sola la capacidad militar completa."
            ),
        },
        {
            "key": "macro",
            "filename": "macro_economic.csv",
            "description": "Indicadores macroeconómicos del Banco Mundial.",
            "data_used": "`country`, `year`, PIB en USD corrientes y población.",
            "calculations": (
                "Unión por país y año con el CSV SIPRI; selección del último año común disponible "
                "para la gráfica de dispersión."
            ),
            "conclusion": (
                "Permite contextualizar el gasto militar por tamaño económico y demográfico. "
                "El PIB corriente no se suma ni se compara directamente con gasto a precios constantes."
            ),
        },
        {
            "key": "arms",
            "filename": "arms_transfers.csv",
            "description": "Transferencias de armas convencionales mayores de SIPRI.",
            "data_used": "`year`, `supplier`, `recipient` y `sipri_tiv`.",
            "calculations": "Suma del TIV por año y proveedor para transferencias relacionadas con Marruecos o Israel.",
            "conclusion": (
                "Permitirá describir patrones de transferencias documentadas. El TIV no es un precio de compra "
                "ni un presupuesto militar."
            ),
        },
        {
            "key": "border",
            "filename": "border_irregular_arrivals.csv",
            "description": "Llegadas irregulares registradas por territorio.",
            "data_used": "`period`, `territory` y `arrivals`, preservando la definición de cada balance oficial.",
            "calculations": "Comparación visual por periodo y territorio; no se combinan llegadas, intentos e interceptaciones.",
            "conclusion": (
                "Permitirá describir variaciones temporales y presión operativa, pero no probará "
                "intencionalidad política o coordinación."
            ),
        },
        {
            "key": "events",
            "filename": "security_events_timeline.csv",
            "description": (
                "Cronología documental de Pegasus, frontera (Ceuta/Melilla), la alianza "
                "Marruecos-Israel-EE. UU. y la OTAN."
            ),
            "data_used": "`event_date`, `event_title`, `topic`, `evidence_level`, `source_url` y `notes`.",
            "calculations": "Ordenación temporal y filtrado por tema (`Pegasus`, `Frontera`, `Alianza`, `OTAN`); no se calculan causalidades.",
            "conclusion": (
                "Separa expresamente hechos confirmados, información oficial, información periodística, "
                "atribuciones e hipótesis. Alimenta el desglose de ciberseguridad en ambas pestañas de "
                "defensa, la pestaña de la Alianza, España-OTAN y el caso documentado en Escenarios de riesgo."
            ),
        },
    ]
    for item in csv_summary:
        frame, error = data[item["key"]]
        status = f"{len(frame):,} filas cargadas" if not frame.empty else "Pendiente de cargar"
        with st.expander(f"`{item['filename']}` — {status}"):
            st.markdown(f"**Descripción:** {item['description']}")
            st.markdown(f"**Datos utilizados:** {item['data_used']}")
            st.markdown(f"**Cálculos realizados:** {item['calculations']}")
            st.markdown(f"**Conclusión permitida:** {item['conclusion']}")
            if error:
                st.caption(error)


def render_presentation_script() -> None:
    st.header("Guion de presentación — 15 minutos")
    st.caption("Este guion se puede leer aquí y también está disponible en `reports/presentation_script.md`.")
    st.markdown(load_presentation_script())


def main() -> None:
    st.set_page_config(
        page_title="Spain Security & Defense Analytics",
        page_icon="🛡️",
        layout="wide",
    )
    st.title("España ante el nuevo escenario de seguridad")
    st.subheader("Análisis de defensa, ciberseguridad, geopolítica y resiliencia civil")
    st.caption("Proyecto académico basado en fuentes trazables. Última actualización de datos: la indicada en cada fuente.")

    data = load_project_data(processed_data_revision())
    tabs = st.tabs(
        [
            "Overview",
            "Defensa de España",
            "Defensa de Marruecos",
            "Comparación internacional",
            "Alianza — Acuerdos de Abraham",
            "España-OTAN",
            "Comparación alianzas internacionales",
            "Escenarios de riesgo",
            "Preparación civil",
            "Fuentes",
            "Guion 15 min",
        ]
    )
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_defence(data)
    with tabs[2]:
        render_morocco_defence(data)
    with tabs[3]:
        render_comparison(data)
    with tabs[4]:
        render_abraham_accords_alliance(data)
    with tabs[5]:
        render_spain_nato(data)
    with tabs[6]:
        render_international_alliance_comparison(data)
    with tabs[7]:
        render_risk_matrix(data)
    with tabs[8]:
        render_civil_preparedness()
    with tabs[9]:
        render_sources(data)
    with tabs[10]:
        render_presentation_script()


if __name__ == "__main__":
    main()
