# Proyecto: Framework QUBO 

Framework para implementar hamiltonianos derivados de modelos QUBO y ejecutarlos de manera cuántica.

## Estructura del Repositorio

Una breve explicación de las carpetas principales:
*   `config/`: Para archivos de configuración.
*   `data/`: Contiene los datos de entrada necesarios para el proyecto.
*   `docs/`: (Si existe) Documentación detallada.
*   `env/`: Scripts de activación de entorno.
*   `notebooks/`: Scripts para automatizar tareas (ej. descarga de datos, preprocesamiento).
*   `results/`: Figuras, tablas y otros resultados generados.
*   `scripts/`: Para archivos y utilidades que no son parte del nucleo lógico de src pero que son importantes para el flujo del trabajo (mismo rol que main).
*   `src/`: Código fuente Python modularizado (funciones, clases y modulos). Código reutilizable en notebooks, scripts. Reside la lódgica central del proyecto.
*   `tests/`: Contiene pruebas unitarias.
*   `.gitignore`: Los tipos de archivos explicitados en el, son ignorados por git.
*   `environment.yml`: Recoge los requisitos necesarios del entorno de desarrollo.
*   `pyproject.toml`: Especificación y gestión de las dependecias de construcción y los metadatos del proyecto. Posibilita la istalación de paquetes y por ende la importación del paquete desde otros puntos del proyecto.


## Requisitos Previos

Antes de empezar, asegúrate de tener instalado:
*   [Git](https://git-scm.com/)
*   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) o [Anaconda](https://www.anaconda.com/products/individual)
*   [Python](https://www.python.org/downloads/) (versión 3.X)

## Instalación

1.  **Clona el repositorio (elige licencia):**
    ```bash
    git clone https://github.com/IkerEtxeQ/qubo-optimization-wrapper.git
    cd tu_proyecto
    ```
2.  **Rellena la plantilla**

    1. environment.yml: Poner nombre al entorno env -> qubo-framework-env
    2. pyproject.toml:  Poner nombre [project].name al paquete de código fuente, lo que se importara. 
                        Rellenar metadatos.
    3. readme.md: Rellenar datos.
    
2.  **Crea y activa el entorno:**
    Opción1:
    *   **Usando script**
        Abre una terminal en la raiz del proyecto:
        ``.\env\start_env.ps1´´
        Crea el entorno (si no existe, si existe lo actualiza) y verifica que todos los paquetes espeficados en el environment.yml y pyproject.toml estan instalados.
        Instala código fuente: Para poder importar los módulos dentro de src.
        ** Si se hacen modificaciones al archivo start_env.ps1 no olvidar guardarlo en la codificación utf-8 con BOM para los acentos y los emojis. En VSCode barra inferior UTF8 -> Guardar codificación -> UTF-8 with BOM.

    Opción 2:
    *   **Manualmente**
        Abre una terminal en la raiz del proyecto:
        ```bash
        conda init powershell
        conda env create -f environment.yml
        conda activate qubo-framework-env
        ```
    * **Instalar paquete src en modo editable ()**
      **Usa `pyproject.toml`:**
        `` pip install -e .´´ 
        Despues de ejecutar esto, se podran importar modulos dentro de la carpeta src desde cualquier otro script o notebook del entrono con import mi_modulo. 
        Refleja los cambios que se hagan en src instantaneamente sin necesidad de reinstalar.

## Integrar nuevas librerías

Si estoy desarrollando y necesito hacer uso de una librería que no esta en el entorno:
    1. Si mi código esta dentro de src: 
        * Ejecutar en terminal: ``conda install nombre_pkg´´ si solo esta en PyPI ``pip install nombre_pkg´´
        * Añadir nueva librería a pyproject.toml y environment.yml.
    2. Si mi código esta fuera de src.
        * Ejecutar en terminal: ``conda install nombre_pkg´´ si solo esta en PyPI ``pip install nombre_pkg´´
        * Añadir nueva librería a environment.yml.

## Distribuir software.

    `` python -m build --wheel´´
*   Crea un wheel de src: distribución binaria precompilada.

    `` python -m build --sdist´´
*   Crea un sdist de src: distribución fuente.

## Resultados

Describe brevemente qué tipo de resultados se generan y dónde se guardan.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Si tienes preguntas, puedes contactar a Iker Etxebarria en ietxebarriao@ayesa.com
