# start_env.ps1 🚀
# Script para configurar el entorno Conda y ejecutar la verificación.

$envFile = "environment.yml"
$ErrorActionPreference = "Stop" # Salir del script si un comando falla (excepto donde se maneje explícitamente)

# Función auxiliar para intentar obtener el PATH de Python global/base (ya no se usa si nbstripout se elimina)
# function Get-GlobalPythonPath { ... } # Podrías eliminar esta función si no la necesitas para nada más

try {
    # --- Sección 1: Configuración del Entorno Conda ---
    Write-Host "--- Configurando Entorno Conda ---" -ForegroundColor Yellow
    $script:envName = $null # Definir a nivel de script para accesibilidad si fuera necesario

    $envNameLine = Select-String -Path $envFile -Pattern "^name:\s*(.+)$"
    if (-not $envNameLine) {
        Write-Error "❌ No se pudo encontrar el nombre del entorno en '$envFile'"
        exit 1
    }
    $script:envName = $envNameLine.Matches[0].Groups[1].Value.Trim()
    Write-Host "📦 Nombre del entorno Conda: '$($script:envName)'"

    $envExists = conda env list | Select-String -Pattern "^\s*$($script:envName)\s" -ErrorAction SilentlyContinue
    if (-not $envExists) {
        Write-Host "⚠️ Entorno '$($script:envName)' no encontrado. Creando desde '$envFile'..."
        conda env create -f $envFile
        Write-Host "✅ Entorno '$($script:envName)' creado." -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Entorno '$($script:envName)' ya existe. Actualizando desde '$envFile'..."
        conda env update --name $script:envName --file $envFile --prune
        Write-Host "✅ Entorno '$($script:envName)' actualizado." -ForegroundColor Green
    }

    Write-Host "🧪 Activando entorno Conda: '$($script:envName)' (para operaciones del script)..."
    conda activate $script:envName
    Write-Host "✅ Entorno '$($script:envName)' activado." -ForegroundColor Green

    # --- Sección 2: Verificación del Entorno y Ejecución de Tareas del Proyecto ---
    # Asumimos que tu proyecto se instala a través de `environment.yml` con `-e .` si es necesario.
    Write-Host "--- Verificando Entorno y Ejecutando Tareas del Proyecto ---" -ForegroundColor Yellow
    
    Write-Host "🐍 Ejecutando script de verificación: env/verify_env.py..."
    python env/verify_env.py # Usa el python del entorno activado
    Write-Host "✅ Verificación del entorno completada." -ForegroundColor Green
    
    # Si tenías `pip install -e .` aquí y no lo moviste a environment.yml,
    # podrías necesitarlo de vuelta. Pero es mejor práctica en environment.yml.
    # Write-Host "🛠️ Instalando el proyecto en modo editable (si no está en environment.yml)..."
    # pip install -e . 
    # Write-Host "✅ Proyecto instalado en modo editable." -ForegroundColor Green

}
catch {
    Write-Error "💥 Se produjo un error durante la ejecución del script:"
    Write-Error $_.Exception.Message
    if ($_.InvocationInfo) {
        Write-Error "En la línea $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line)"
    }
    exit 1
}

Write-Host "-----------------------------------------------------" -ForegroundColor Cyan
Write-Host "🎉 Script de inicio de entorno y verificación completado." -ForegroundColor Magenta
Write-Host "-----------------------------------------------------" -ForegroundColor Cyan