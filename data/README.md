# Descarga del dataset — dunnhumby "The Complete Journey"

Poné acá (`data/`) los siguientes archivos CSV, sin renombrarlos (~808MB en
total; no se suben a git porque son públicos y muy pesados):

- `transaction_data.csv`
- `product.csv`
- `hh_demographic.csv`
- `campaign_desc.csv`
- `campaign_table.csv`
- `coupon.csv`
- `coupon_redempt.csv`
- `causal_data.csv`

## Opción A — Kaggle (recomendada, más rápida)

1. Creá una cuenta gratuita en https://www.kaggle.com si no tenés.
2. Buscá el dataset **"dunnhumby - The Complete Journey"**.
3. Descargalo manualmente desde la web (botón *Download*) y descomprimí el ZIP acá,
   **o** usá la API de Kaggle:

   ```bash
   pip install kaggle
   # Configurá tu API token: Kaggle > Account > Create New Token,
   # y colocá el kaggle.json descargado en C:\Users\<tu_usuario>\.kaggle\kaggle.json
   kaggle datasets download -d frtgnn/dunnhumby-the-complete-journey -p data --unzip
   ```

   (El slug del dataset puede variar según el mirror que uses en Kaggle; si ese
   comando da 404, buscá el dataset en la web de Kaggle y copiá el comando
   `kaggle datasets download -d ...` que te da el botón de los 3 puntos > *Copy API command*.)

## Opción B — Portal oficial dunnhumby

1. Andá a https://www.dunnhumby.com/source-files/ y registrate con tu email.
2. Buscá *"The Complete Journey"* y descargá el ZIP.
3. Descomprimí los CSV acá.

## Después de descargar

Corré el pipeline (`python -m src.pipeline`) o `python -m src.cargar_datos
--validar-completo` — si falta algún archivo, el script avisa explícitamente
cuál antes de fallar con un error críptico. Luego `python -m
src.calcular_precios_catalogo` y `python -m src.preparar_demo` para que la
app (`streamlit run app/app.py`) tenga datos para mostrar.
