# Proyecto Científico: [Nombre de tu Proyecto]

Breve descripción de una o dos frases sobre qué trata el proyecto, su objetivo principal y el tipo de ciencia que se desarrolla.

## Estructura del Repositorio

Una breve explicación de las carpetas principales:
*   `config/`: Para archivos de configuración.
*   `data/`: Contiene los datos de entrada necesarios para el proyecto.
*   `docs/`: (Si existe) Documentación detallada.
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
    git clone https://github.com/tu_usuario/tu_proyecto.git
    cd tu_proyecto
    ```
2.  **Rellena la plantilla**

    1. environment.yml: Poner nombre al entorno env -> mi-proyecto-env
    2. pyproject.toml:  Poner nombre [project].name al paquete de código fuente, lo que se importara. 
                        Rellenar metadatos.
    3. readme.md: Rellenar datos.
    
2.  **Crea y activa el entorno:**

    *   **Usando Conda (`environment.yml`):**
        ```bash
        conda env create -f environment.yml
        conda activate nombre-del-entorno-definido-en-yml
        ```

3. **Instalar paquete src en modo editable ()**

    *   **Usa `pyproject.toml`:**
        `` pip install -e .´´ 
*       Despues de ejecutar esto, se podran importar modulos dentro de la carpeta src desde cualquier otro script o notebook del entrono con import mi_modulo. 
*       Refleja los cambios que se hagan en src instantaneamente sin necesidad de reinstalar.

## Distribuir software con el código fuente compilado.

    `` python -m build --wheel´´
*   Crea un wheel de src: distribución binaria precompilada.

    `` python -m build --sdist´´
*   Crea un sdist de src: distribución fuente.

## Resultados

Describe brevemente qué tipo de resultados se generan y dónde se guardan.

## Licencia

Este proyecto está bajo la Licencia [Nombre de la Licencia - ej. MIT, Apache 2.0]. Ver el archivo `LICENSE` para más detalles.

## Contacto

Si tienes preguntas, puedes contactar a Iker Etxebarria en ietxebarriao@ayesa.com