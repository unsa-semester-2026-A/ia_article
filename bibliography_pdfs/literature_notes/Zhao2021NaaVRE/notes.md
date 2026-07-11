# Notebook-as-a-VRE (NaaVRE): From private notebooks to a collaborative cloud virtual research environment

- **Key**: Zhao2021NaaVRE
- **Year**: 2022
- **Venue**: Software: Practice and Experience

## Resumen
Los Entornos Virtuales de Investigación (VRE) proporcionan soporte centrado en el usuario durante el ciclo de vida de las actividades científicas (búsqueda de datos, modelado, flujos de trabajo, gestión de infraestructura y colaboración). Sin embargo, su adopción suele verse obstaculizada por la alta inversión de tiempo necesaria para aprender nuevas tecnologías rígidas. Por otro lado, los entornos de cuadernos como Jupyter son muy populares para la creación rápida de prototipos interactivos, pero carecen de soporte nativo para ejecutar cálculos a gran escala en infraestructura remota o compartir fragmentos de código de forma colaborativa. Este artículo investiga la brecha entre los cuadernos y los VREs y propone una solución integrada en Jupyter llamada Notebook-as-a-VRE (NaaVRE). NaaVRE expone componentes modulares a través de un catálogo (marketplace) que permite a los científicos encapsular fragmentos de código (*cells*) como contenedores Docker (usando FAIR-Cells), componer flujos de trabajo lógicos distribuidos (usando Argo/CWL), automatizar recursos multi-nube (usando SDIA/Cloud-Cells) y registrar la procedencia y el historial en un libro mayor distribuido (blockchain). Se demuestra la eficacia de NaaVRE con un caso de estudio ecológico (LidarAirCloud) escalando el procesamiento de terabytes de nubes de puntos LiDAR del territorio holandés (AHN3).

## Secciones y Subsecciones

### I. Introducción
Introduce la necesidad de grandes volúmenes de datos, modelos avanzados y computación distribuida para resolver retos científicos globales. Define los VREs y contrasta su rigidez frente al dinamismo ligero de los cuadernos de Jupyter.
* **Problemas atacados**: La dificultad que tienen los científicos para pasar de prototipos locales ligeros en Jupyter a ejecuciones en la nube a gran escala de manera colaborativa, sin tener que abandonar su interfaz cotidiana ni aprender tecnologías complejas de orquestación.
* **Limitaciones de ese entonces**: Los VREs tradicionales son monolíticos, requieren costosos registros web (p. ej., D4Science) y software cliente pesado. Los cuadernos Jupyter facilitan prototipar pero no interactúan fluidamente con infraestructuras en la nube remotas ni permiten descubrir funciones modulares en repositorios de código monolíticos de Git.
* **Soluciones alcanzadas**: Propuesta conceptual de NaaVRE, que expande el entorno de cuadernos Jupyter para convertirlo en un VRE integrado y modular.

### II. Descripción del Problema y Trabajo Relacionado
Analiza las interacciones complejas de los tres ciclos en las investigaciones científicas modernas: investigación científica, gestión de datos (basado en el modelo de referencia ENVRI) y utilización de infraestructura.

#### II.A. Entornos Virtuales de Investigación y Ciclo de Vida de la Investigación
Detalla los requisitos funcionales de un VRE para gestionar activos, experimentos y colaboración.
* **Problemas atacados**: Integrar flujos de trabajo heterogéneos y dinámicos propios del ciclo de vida científico (que es iterativo por naturaleza).
* **Limitaciones de ese entonces**: Las soluciones tradicionales no son flexibles, carecen de reproducibilidad distribuida y de una capa de confianza sobre los productos de datos intermedios provenientes de terceros.
* **Soluciones alcanzadas**: Modelar el soporte informático para automatizar procesos estandarizados, asegurar la procedencia de datos y abstraer las APIs de recursos físicos.

#### II.B. Límites del Entorno Jupyter
Describe los tres problemas fundamentales de Jupyter como plataforma científica abierta.
* **Problemas atacados**: La falta de granularidad, reusabilidad de bloques lógicos y escalabilidad en cuadernos de ciencia de datos.
* **Limitaciones de ese entonces**: 
  1. Los cuadernos se comparten en repositorios Git como archivos de texto monolíticos JSON (.ipynb), lo que impide buscar o aislar funciones individuales (*cells*) fácilmente.
  2. Acoplamiento implícito de variables de entorno y librerías que entorpece reutilizar fragmentos de código en otros flujos de trabajo.
  3. Los servidores JupyterHub están acoplados a una capacidad de nube preconfigurada fija, impidiendo la paralelización en clusters dinámicos bajo demanda.
* **Soluciones alcanzadas**: Identificación de requisitos para modularizar fragmentos de código de Jupyter (*FAIR-Cells*) y enlazarlos a orquestadores en la nube.

#### II.C. Retos en el Desarrollo de VREs
Discute los problemas de sostenibilidad de software académico y la interoperabilidad de APIs.
* **Problemas atacados**: La baja sostenibilidad de interfaces VRE diseñadas a medida para una única infraestructura o comunidad científica.
* **Limitaciones de ese entonces**: La falta de interoperabilidad de metadatos y APIs entre diversas infraestructuras de investigación obliga a rediseñar integraciones personalizadas complejas.
* **Soluciones alcanzadas**: Adopción de una arquitectura de VRE embebida en la interfaz Jupyter que hereda las prácticas cotidianas del usuario.

#### II.D. Trabajos Relacionados
Examina extensiones previas de Jupyter para acceso a datos (como EODAG), portabilidad (Binder) y reproducibilidad.
* **Problemas atacados**: Soporte específico para tareas puntuales en cuadernos científicos.
* **Limitaciones de ese entonces**: Las soluciones existentes se limitan a integrar recursos de datos únicos, restaurar el orden de celdas por análisis de dependencias estáticas o empaquetar entornos virtuales fijos, pero no abordan el ciclo de vida completo de un VRE.
* **Soluciones alcanzadas**: Posicionamiento de NaaVRE como una suite integral que cubre procedencia, orquestación en la nube, blockchain y componibilidad de flujos de trabajo.

### III. Notebook-as-a-VRE (NaaVRE)
Detalla la arquitectura de NaaVRE como un ecosistema modular de microservicios y extensiones de Jupyter.

#### III.A. Requisitos
Define los cinco pilares de desarrollo del sistema.
* **Problemas atacados**: Compatibilidad con las prácticas del usuario, modularidad de servicios y descentralización.
* **Limitaciones de ese entonces**: Los desarrollos de VRE imponen metodologías de trabajo intrusivas a los investigadores.
* **Soluciones alcanzadas**: Requisitos de integración nativa, extensibilidad para componentes comunitarios, reconfiguración personalizada, tolerancia a fallos y uso de arquitecturas descentralizadas.

#### III.B. Arquitectura del Sistema
Describe los 12 componentes funcionales del marketplace de NaaVRE (desde base de datos de conocimiento A hasta DevOps L).
* **Problemas atacados**: La orquestación y el desacoplamiento de los módulos necesarios para la computación científica remota.
* **Limitaciones de ese entonces**: La falta de abstracción entre la interfaz de usuario (cuaderno) y los motores de infraestructura en la nube o bases de procedencia de datos.
* **Soluciones alcanzadas**: Diseño de servicios específicos:
  - *FAIR-Cells (Component Containerizer)*: Extrae e interactúa con celdas seleccionadas para empaquetarlas automáticamente como microservicios RESTful dentro de contenedores Docker.
  - *Experiment Manager*: Diseña y compone flujos de trabajo usando especificaciones como Common Workflow Language (CWL).
  - *Cloud-Cells (Infrastructure Automator)*: Utiliza SDIA para aprovisionar dinámicamente recursos en la nube de diversos proveedores (Azure, AWS, EGI) y configurar clusters Kubernetes para los contenedores de las celdas.
  - *Distributed Workflow Bus (Argo/Hyperledger)*: Orquesta la ejecución del flujo y registra metadatos inmutables en una blockchain empresarial (Hyperledger Fabric) junto con almacenamiento distribuido (IPFS/WebDAV) y procedencia PROV-O.

#### III.C. ¿Cómo Puede Operar NaaVRE?
Muestra los flujos de operación tanto para laboratorios virtuales organizados como PaaS, como para usuarios individuales.
* **Problemas atacados**: Flexibilidad para el despliegue del sistema tanto en instituciones centralizadas como en estaciones de trabajo personales descentralizadas.
* **Limitaciones de ese entonces**: Los frameworks científicos requieren administradores de sistemas dedicados e infraestructuras locales pesadas para su despliegue.
* **Soluciones alcanzadas**: Doble modelo operativo. Un operador de plataforma (PaaS) puede instanciar laboratorios virtuales (VL) personalizados basados en JupyterHub. De forma alternativa, un investigador individual puede instalar componentes locales desde el marketplace e integrarse directamente en la red peer-to-peer (IPFS/Blockchain).

### IV. Implementación y Estado Actual
Describe la metodología ágil de desarrollo colaborativo y los componentes implementados en el prototipo de código abierto disponible en GitHub.
* **Problemas atacados**: Desarrollo incremental de las herramientas de procedencia, búsqueda y DevOps científico.
* **Limitaciones de ese entonces**: Integración de estándares abiertos heterogéneos (PROV-O, Docker, Kubernetes, Blockchain) en una sola interfaz estable.
* **Soluciones alcanzadas**: Prototipado exitoso del motor de búsqueda (ElasticSearch), el containerizador de celdas, el compositor de Argo, la integración con SDIA para clouds híbridos, almacenamiento WebDAV/IPFS, el ledger Hyperledger Fabric y un visualizador interactivo de gráficos de procedencia (PROV-O) alineado con los registros de rendimiento del hardware.

### V. Caso de Estudio: Laboratorio Virtual LidarAirCloud
Demuestra la solución migrando y escalando el pipeline de extracción de características LiDAR 'Laserchicken' de entornos locales a la nube.
* **Problemas atacados**: El procesamiento ineficiente a nivel local de múltiples terabytes de nubes de puntos LiDAR provenientes de escaneos aéreos nacionales (AHN3 en los Países Bajos).
* **Limitaciones de ese entonces**: La herramienta Laserchicken solo podía ejecutarse en la máquina local o infraestructura fija del científico mediante interacción manual paso a paso (load, compute neighbors, features, export), limitando la escala geográfica de los análisis ecológicos.
* **Soluciones alcanzadas**: Creación de un flujo de trabajo modularizado en NaaVRE. Las celdas se empaquetan en dockers, se compone el grafo lógico con operadores paralelos de división y fusión (*split* y *merge*), y el Workflow Bus automatiza el despliegue y la paralelización de múltiples instancias sobre máquinas virtuales y recursos dinámicos en la nube (Microsoft Azure y EOSC). Esto permitió procesar conjuntos masivos de terabytes de LiDAR en paralelo para mapear variables de estructura del ecosistema a escala nacional.

### VI. Discusión
Analiza la interoperabilidad y el soporte para integrarse con otros sistemas gestores de flujos de trabajo tradicionales.
* **Problemas atacados**: El aislamiento de NaaVRE respecto a motores de ejecución heredados populares en la ciencia.
* **Limitaciones de ese entonces**: La dificultad de forzar a las comunidades a migrar sus motores de ejecución existentes a una nueva solución de VRE.
* **Soluciones alcanzadas**: Desacoplamiento del Backend de NaaVRE. La cola de tareas (Workflow Bus) puede llamar a motores externos bajo demanda si estos exponen APIs adecuadas para invocaciones remotas, y la interfaz de cuadernos puede exportar grafos compatibles con herramientas de terceros.

### VII. Resumen
Resume las lecciones aprendidas y los proyectos de financiación del proyecto.
* **Problemas atacados**: Consolidar el desarrollo de la ciencia de datos abierta en proyectos multidisciplinares.
* **Limitaciones de ese entonces**: Falta de adopción práctica en dominios reales fuera de las pruebas sintéticas.
* **Soluciones alcanzadas**: NaaVRE se implementa activamente en consorcios europeos como ENVRI-FAIR y LifeWatch, sirviendo a comunidades de ecología, ciencias marinas y adaptándose para flujos de machine learning distribuido en investigación médica.
