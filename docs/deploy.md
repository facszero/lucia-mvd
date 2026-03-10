# 🚀 Deploy en Streamlit Cloud — LUCÍA-MVD

## Pasos para tener la app en línea en 5 minutos

### 1. Ir a Streamlit Cloud
https://share.streamlit.io  
(loguearse con la cuenta de GitHub: **facszero**)

### 2. New app
Hacer click en **"New app"** y completar:

| Campo | Valor |
|-------|-------|
| Repository | `facszero/lucia-mvd` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL (opcional) | `lucia-mvd` → genera `lucia-mvd.streamlit.app` |

### 3. Advanced settings (opcional)
- Python version: **3.11**
- Secrets: no se necesitan para esta versión

### 4. Deploy
Click en **"Deploy!"** — el build tarda ~3-5 minutos la primera vez
(instala dependencias del sistema vía `packages.txt` y Python vía `requirements.txt`)

### URL final
```
https://lucia-mvd.streamlit.app
```
o con el slug personalizado que elijas.

---

## Archivos de configuración incluidos

| Archivo | Propósito |
|---------|-----------|
| `packages.txt` | Dependencias del sistema Ubuntu (libgdal, libgeos, etc.) |
| `requirements.txt` | Dependencias Python con versiones fijadas |
| `.streamlit/config.toml` | Tema oscuro + configuración del servidor |

---

## Alternativa: Deploy con Docker

```bash
# Build
docker build -t lucia-mvd .

# Run local
docker run -p 8501:8501 lucia-mvd

# Deploy en Railway / Render / Fly.io
# (subir imagen y exponer puerto 8501)
```

---

## Verificar que el deploy funcionó

La app debe:
1. Mostrar el header **LUCÍA-MVD** con el badge de ONU Mujeres
2. Cargar el mapa de Montevideo con heatmap de riesgo
3. Responder al selector de franja horaria
4. Mostrar el simulador de intervenciones funcional

Si hay error en el primer deploy, revisar los logs en Streamlit Cloud:
- Error de `libgdal` → revisar `packages.txt`
- Error de `import` → revisar `requirements.txt`
- Error de datos → el pipeline se ejecuta automáticamente al iniciar
