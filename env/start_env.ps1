# start_env.ps1 🚀

$envFile = "environment.yml"
$ErrorActionPreference = "Stop" # Salir del script si un comando falla (excepto donde se maneje explícitamente)

try {
    # --- Sección 1: Configuración del Entorno Conda ---
    Write-Host "--- Configurando Entorno Conda ---" -ForegroundColor Yellow

    # 📄 Leer nombre del entorno desde environment.yml
    $envNameLine = Select-String -Path $envFile -Pattern "^name:\s*(.+)$"
    if (-not $envNameLine) {
        Write-Error "❌ No se pudo encontrar el nombre del entorno en '$envFile'"
        exit 1 # Salir explícitamente aquí ya que $ErrorActionPreference="Stop" podría no cubrir esto
    }
    $envName = $envNameLine.Matches[0].Groups[1].Value.Trim()
    Write-Host "📦 Nombre del entorno Conda: '$envName'"

    # 🔍 Comprobar si el entorno existe
    $envExists = conda env list | Select-String -Pattern "^\s*$($envName)\s" # Corregido: usar $envName correctamente entrecomillado

    if (-not $envExists) {
        Write-Host "⚠️ Entorno '$envName' no encontrado. Creando desde '$envFile'..."
        conda env create -f $envFile
        # $LASTEXITCODE se comprueba automáticamente por $ErrorActionPreference="Stop"
        Write-Host "✅ Entorno '$envName' creado." -ForegroundColor Green
    }
    else {
        Write-Host "ℹ️ Entorno '$envName' ya existe. Actualizando desde '$envFile'..."
        conda env update --name $envName --file $envFile --prune # Usar --name y --file para claridad
        Write-Host "✅ Entorno '$envName' actualizado." -ForegroundColor Green
    }

    # ⚙️ Activar entorno
    Write-Host "🧪 Activando entorno Conda: '$envName'..."
    # Nota: `conda activate` en un script de PS tiene matices.
    # Su efecto principal es modificar el PATH para esta sesión de PowerShell.
    # Los comandos subsecuentes DEBERÍAN usar los ejecutables del entorno activado.
    conda activate $envName
    Write-Host "✅ Entorno '$envName' activado (para esta sesión de script)." -ForegroundColor Green

    # --- Sección 2: Verificación y Instalación del Proyecto ---
    Write-Host "--- Verificando e Instalando Proyecto ---" -ForegroundColor Yellow

    # ✅ Ejecutar la verificación del entorno
    Write-Host "🐍 Ejecutando script de verificación: env/verify_env.py..."
    python env/verify_env.py # Asume que `python` ahora es el del entorno activado
    Write-Host "✅ Verificación del entorno completada." -ForegroundColor Green

    # 🔄 Instalar el proyecto en modo editable
    Write-Host "🛠️ Instalando el proyecto en modo editable (pip install -e .)..."
    pip install -e . # Asume que `pip` ahora es el del entorno activado
    Write-Host "✅ Proyecto instalado en modo editable." -ForegroundColor Green


    # --- Sección 3: Configuración de nbstripout ---
    Write-Host "--- Configurando nbstripout para Git ---" -ForegroundColor Yellow

    # 1. Verificar nbstripout (ahora que el entorno está activo y el proyecto instalado)
    Write-Host "Verificando nbstripout..."
    if (-not (Get-Command nbstripout -ErrorAction SilentlyContinue)) {
        Write-Warning "nbstripout no encontrado en el entorno '$envName'."
        Write-Warning "Asegúrate de que 'nbstripout' esté listado como dependencia en '$envFile' (sección pip)"
        Write-Warning "o instálalo manualmente en el entorno activado: pip install nbstripout"
        # Podrías optar por salir si nbstripout es crucial:
        # throw "nbstripout es necesario y no se encontró."
    } else {
        Write-Host "  nbstripout encontrado:" -ForegroundColor Green
        nbstripout --version # Se ejecutará la versión del entorno activado
    }

    # 2. Configurar filtros de Git
    Write-Host "Configurando filtros Git para nbstripout..."
    # `nbstripout --install` modifica .git/config local al repo
    nbstripout --install
    Write-Host "  Configuración de Git para nbstripout completada o ya existente." -ForegroundColor Green

    # 3. Asegurar .gitattributes
    Write-Host "Asegurando .gitattributes..."
    $gitattributesPath = ".gitattributes"
    $expectedContent = @(
        "*.ipynb filter=nbstripout diff=nbstripout",
        "*.zpln filter=nbstripout"
    )

    $currentContent = if (Test-Path $gitattributesPath) { Get-Content $gitattributesPath -ErrorAction SilentlyContinue } else { $null }

    if ($null -eq $currentContent -or (Compare-Object -ReferenceObject $expectedContent -DifferenceObject $currentContent)) {
        Set-Content -Path $gitattributesPath -Value ($expectedContent | Out-String) -Encoding utf8NoBOM
        Write-Host "  .gitattributes creado/actualizado." -ForegroundColor Green
        $gitattributesStatus = git status --porcelain -- $gitattributesPath
        if ($gitattributesStatus -match "^\?\? " -or $gitattributesStatus -match "^ M ") {
            Write-Warning "IMPORTANTE: '.gitattributes' ha sido modificado. Por favor, añádelo a Git: git add .gitattributes"
        }
    } else {
        Write-Host "  .gitattributes ya está configurado correctamente." -ForegroundColor Green
    }

    # 4. Verificar configuración final de nbstripout
    Write-Host "Verificando configuración final con 'nbstripout --status'..."
    if (Get-Command nbstripout -ErrorAction SilentlyContinue) {
        nbstripout --status
    }

    Write-Host "-----------------------------------------------------"
    Write-Host "Notas de nbstripout:"
    Write-Host " - Si '.gitattributes' fue modificado, no olvides hacer 'git add .gitattributes' y 'git commit'."
    Write-Host " - Limpia notebooks existentes si es necesario: Get-ChildItem -Path . -Include '*.ipynb','*.zpln' -Recurse | ForEach-Object { nbstripout \$_.FullName }"
    Write-Host "   Luego: git add . && git commit -m 'Clean existing notebooks'"
    Write-Host "nbstripout configurado." -ForegroundColor Green

}
catch {
    Write-Error "💥 Se produjo un error durante la ejecución del script:"
    Write-Error $_.Exception.Message
    if ($_.InvocationInfo) {
        Write-Error "En la línea $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line)"
    }
    exit 1
}

Write-Host "🎉 Script de inicio de entorno y configuración completado." -ForegroundColor Magenta