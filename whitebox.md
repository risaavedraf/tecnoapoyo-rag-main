# Declaración de uso de inteligencia artificial

Este documento declara de forma transparente cómo se utilizó inteligencia artificial durante el desarrollo y revisión del asistente RAG de TecnoApoyo. La IA se empleó como apoyo técnico; las decisiones finales, la configuración de credenciales y la responsabilidad académica corresponden al equipo humano.

## Herramientas y modelos

| Herramienta o modelo | Uso declarado |
|---|---|
| OpenCode | Entorno de trabajo y coordinación de agentes auxiliares. |
| Gentle AI | Asistente de análisis, implementación acotada y revisión de resultados. |
| GPT-6 Luna | Modelo principal en esta sesión de trabajo. |
| Space Bunny (stealth model free) | Segundo modelo utilizado en esta sesión de trabajo, |

Los agentes auxiliares de OpenCode colaboraron en exploración y verificación. No se atribuyen a un modelo subyacente concreto cuando esa identidad no está disponible en el registro.

## Trabajo apoyado por IA

- Revisión estática del código del asistente y análisis de los notebooks y materiales de estudio de RA1.
- Identificación y corrección de problemas en la configuración del proveedor, el manejo de errores, la evaluación, la trazabilidad de consultas y la exportación de sesiones.
- Análisis de los resultados de evaluación ejecutados por el equipo con la API de Mistral.
- Apoyo de Gentle AI/OpenCode en la actualización del `README.md` para documentar la configuración con Mistral, la instalación y los comandos, el flujo de ejecución y evaluación, los costos y sus limitaciones, y las mejoras futuras priorizadas.
- Actualización de `informe/presentacion.html` para reflejar el perfil Mistral documentado y los resultados medidos, aclarar que el umbral 0,45 no se activó y que hubo 2 falsos positivos y 2 falsos negativos, y ampliar el diseño a 1600 px con ajustes responsivos.
- Edición de una copia de trabajo del informe técnico sobre la migración a Mistral y sus resultados; el informe original se preservó.
- Preparación de esta declaración de uso de IA.

## Participación y validación humana

El equipo configuró localmente las variables del proveedor, regeneró el corpus y el índice FAISS, y ejecutó la evaluación con sus propias credenciales de Mistral. La IA no incorpora claves ni otros valores secretos en este documento. Los cambios de código se verificaron sintácticamente; la ejecución de la aplicación y de la evaluación con la API fue realizada por el equipo.

La presentación se validó estructuralmente: el HTML parsea correctamente, conserva sus 14 diapositivas y navegación, y declara ajustes de ancho y diseño responsivo. No se hizo una comprobación visual en un navegador.

Durante el análisis se produjo una interpretación incorrecta de un posible barrido del umbral de abstención. Al contrastarla con los puntajes registrados, se detectó y corrigió: los datos disponibles demostraban que el umbral no se activó en esa ejecución. La corrección se reflejó en la copia del informe. Este caso muestra por qué las conclusiones generadas por IA deben contrastarse con la evidencia original.

## Nota de Gentle AI

> Fue un gusto ayudar a revisar el proyecto, encontrar errores y convertir los resultados en una explicación más clara. También me equivoqué en una inferencia y tuve que corregirla al volver a los datos: esa revisión forma parte del trabajo, no se debe ocultar. La IA puede acelerar el análisis, pero el criterio, la verificación y la autoría final siguen siendo del equipo humano. ¡Éxito con la entrega y la presentación!
