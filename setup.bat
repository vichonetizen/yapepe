@echo off
echo ============================================
echo   PENTAMODAL 3.0 - Instalacion
echo ============================================
echo.
echo Instalando dependencias Python...
pip install -r requirements.txt
echo.
echo Copiando archivo de configuracion...
if not exist .env (
    copy .env.example .env
    echo Archivo .env creado. Edita la clave ANTHROPIC_API_KEY antes de iniciar.
) else (
    echo Archivo .env ya existe.
)
echo.
echo ============================================
echo   Instalacion completa!
echo   1. Edita el archivo .env con tu API key
echo   2. Ejecuta run.bat para iniciar
echo ============================================
pause
