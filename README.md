# Spain Security & Defense Analytics

Proyecto académico para analizar defensa, ciberseguridad, geopolítica y resiliencia civil con datos verificables.

## Ejecutar la aplicación

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app\streamlit_app.py
```

El guion de la presentación está disponible en `reports/presentation_script.md` y en la pestaña **Guion 15 min** de la aplicación.

## Datos: reglas y estructura

No hay cifras de ejemplo en el repositorio. La aplicación solo dibuja gráficos si existen CSV procesados reales en `data/processed/`; si faltan, explica el archivo y la fuente requerida.

Conserva los archivos originales en `data/raw/`, no los modifiques y documenta toda transformación. Consulta el registro `data/external/sources.csv`.

### `data/processed/military_expenditure.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `country` | texto | Nombre de país, por ejemplo `Spain` o `Morocco`. |
| `year` | entero | Año de la observación. |
| `military_expenditure_constant_usd` | número | Gasto en USD constantes, con año base documentado. |
| `military_expenditure_pct_gdp` | número | Gasto militar como porcentaje del PIB (por ejemplo, `2.1` representa 2,1 %). |
| `military_expenditure_per_capita_usd` | número | Gasto militar per cápita en USD, según la hoja `Per capita` de SIPRI. |

Fuente primaria recomendada: D01. No combines esta tabla con cifras OTAN en el mismo gráfico sin declarar la metodología.

El pipeline reproducible está en `src/data_cleaning.py`. Con el libro original de SIPRI en `data/raw/sipri/`, ejecútalo así:

```powershell
py -m src.data_cleaning
```

La misma ejecución produce `data/processed/macro_economic.csv` a partir de las respuestas originales de la API del Banco Mundial para PIB y población.

### `data/processed/macro_economic.csv`

Columnas: `country`, `year`, `gdp_current_usd`, `population`, `source_id`, `source_dataset`.

La fuente es D04. El PIB está en USD corrientes y se utiliza para contextualizar el gasto; no se mezcla con la serie de gasto SIPRI a precios constantes.

### `data/processed/arms_transfers.csv`

Columnas requeridas: `year`, `supplier`, `recipient`, `sipri_tiv`.

Fuente primaria: D02. El indicador `sipri_tiv` no representa el precio de compra ni el gasto presupuestario.

### `data/external/defence_programmes.csv`

Catálogo documental de programas de defensa que necesitan desglose para 2021-2025. No contiene
cifras de ejemplo: `importe_eur` permanece vacío hasta verificar un documento primario.

Para cada importe incorporado, documenta si es:

- crédito presupuestado (D17);
- obligación reconocida o pago ejecutado (D18); o
- importe adjudicado de un expediente (D19).

No sustituyas una de esas medidas por otra ni deduzcas gasto anual a partir del coste total de un
programa. El catálogo identifica programas y la fuente institucional para iniciar la extracción,
pero no prueba una anualidad ni una ejecución financiera.

### `data/external/public_defence_equipment.csv`

Catálogo de referencia de equipos publicados por fuentes institucionales, con la categoría y la
función pública general. No es un inventario de las Fuerzas Armadas: no incluye cantidades,
disponibilidad, munición, estado de alerta, distribución por unidad ni ubicaciones operativas.

Cada fila incluye el enlace institucional y la fecha de consulta. D20 permite identificar material
publicado por el Ejército de Tierra y D21 permite diferenciar sistemas en un programa de
modernización de equipos entregados o disponibles.

### Contexto público de defensa

Los archivos `public_defence_installations.csv`, `public_defence_leadership.csv` y
`public_defence_personnel.csv` alimentan el mapa y las tablas de contexto de la pestaña
**Defensa de España**.

### Contexto público de defensa — Marruecos

La pestaña **Defensa de Marruecos** replica la misma estructura que **Defensa de España**
(mandos, instalaciones, personal, buques y equipos), pero Marruecos no publica un portal
institucional de defensa equivalente a `defensa.gob.es`. Por eso estos archivos citan fuentes
enciclopédicas (Wikipedia, con referencia a *The Military Balance* del IISS, EDA, UNROCA y SIPRI
Trade Registers) y prensa especializada (Defense News) en lugar de documentos institucionales
marroquíes de primera mano:

- `public_defence_leadership_morocco.csv`: mandos superiores de las FAR (Rey, Ministro Delegado
  de la Administración de la Defensa Nacional, Inspector General de las FAR, Armada y Fuerzas
  Reales Aéreas).
- `public_defence_installations_morocco.csv`: bases navales y aéreas con municipio y región de
  referencia. La base de Dajla se sitúa en el Sáhara Occidental, territorio no autónomo según
  Naciones Unidas y en disputa.
- `public_defence_personnel_morocco.csv`: estimaciones de personal por rama, todas trazadas a
  *The Military Balance* (IISS) vía Wikipedia, no a una estadística oficial marroquí.
- `representative_navy_vessels_morocco.csv`: clases principales de la Armada Real (FREMM,
  Floréal, Sigma, Descubierta, Rais Bargach, Lazaga, BATRAL).
- `public_defence_equipment_morocco.csv`: familias de equipo del Ejército de Tierra y las
  Fuerzas Reales Aéreas, incluyendo una aprobación de venta DSCA (AH-64E Apache) que no equivale
  a una entrega confirmada.

Las fuentes D35-D41 en `data/external/sources.csv` documentan cada uno de estos archivos.

- Las instalaciones se sitúan mediante un municipio de referencia y describen únicamente su
  misión institucional general. No incluyas coordenadas tácticas, posiciones, disponibilidad,
  rutas, inventario ni despliegues.
- Los mandos deben incluir una URL institucional y la fecha de la fuente. La titularidad es
  sensible al tiempo: no extrapoles datos de una fecha de consulta a otra.
- Las cifras de personal y reservistas deben tener un periodo de referencia y una definición
  oficial. Si no existe una cifra primaria verificable, deja el valor vacío y documenta el límite
  en lugar de estimarlo.
- `public_defence_capabilities.csv` distingue cantidades previstas, contratadas, entregadas y
  publicadas como material. Nunca las presentes como disponibilidad operativa ni las sumes entre sí.
- `international_missions_personnel_context.csv` contiene estimaciones aportadas para el proyecto,
  no cifras oficiales. Debe mostrarse con esa etiqueta y sustituirse por una estadística institucional
  con fecha de referencia cuando esté disponible.

### `data/processed/border_irregular_arrivals.csv`

Columnas requeridas: `period`, `territory`, `arrivals`.

Incluye una columna adicional de metadatos en el pipeline para conservar fuente, fecha de publicación y definición estadística. No agregues llegadas, intentos e interceptaciones como si fueran la misma variable.

El archivo incluido contiene el balance D07 n.º 15 de 2026: datos acumulados provisionales
del 1 de enero al 15 de agosto de 2026 y exclusivamente llegadas por vía terrestre. En
el mapa, el tamaño de cada marcador representa los registros de Ceuta y Melilla de ese
periodo. No debe interpretarse como un conteo de intentos, interceptaciones o rutas.

### `data/processed/security_events_timeline.csv`

Columnas requeridas: `event_date`, `event_title`, `topic`, `evidence_level`, `source_url` (además de
`notes` para el límite de interpretación de cada fila).

Valores admitidos para `evidence_level`:

- `Hecho confirmado`
- `Información oficial`
- `Información periodística`
- `Atribución`
- `Hipótesis`
- `Interpretación`

No uses una atribución como un hecho confirmado.

Este archivo alimenta varias pestañas mediante el filtro de `topic`:

- `Pegasus`: casos CatalanGate (España) y Pegasus Project (acusación contra Marruecos), mostrados
  con el desglose "Ciberseguridad: ataques dados y recibidos" en **Defensa de España** y
  **Defensa de Marruecos**. La antigua pestaña independiente de Ciberseguridad se eliminó.
- `Frontera`: crisis migratoria de Ceuta (mayo de 2021) y tragedia de la valla de Melilla
  (junio de 2022), usados como caso documentado en **Escenarios de riesgo**. La antigua pestaña
  "Marruecos / Ceuta / Melilla" se eliminó por ser redundante con el mapa ya mostrado en Overview.
- `Alianza`: cronología de los Acuerdos de Abraham y el reconocimiento de EE. UU. sobre el Sáhara
  Occidental, en la pestaña **Alianza — Acuerdos de Abraham** (antes "Israel / Marruecos").
- `OTAN`: hitos de la Base Naval de Rota y la Cumbre de Madrid de 2022, en la pestaña
  **España-OTAN** (antes "Rota / OTAN").

## Estado del proyecto

- App Streamlit: creada y preparada para datos verificados.
- Registro de fuentes: creado.
- Guion de 15 minutos: creado.
- Descarga, limpieza y EDA de datasets: pendientes.
