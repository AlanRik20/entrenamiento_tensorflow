# Ejemplo de Predicción con TensorFlow.js

Este proyecto entrena un modelo de aprendizaje automático en el navegador para aprender la función **y = 2x + 6** usando TensorFlow.js. El usuario puede ingresar un número y el modelo le dirá el resultado de `y` según lo aprendido.

## Características

- Entrenamiento del modelo directamente desde el navegador
- Uso de TensorFlow.js
- Entrada de número por pantalla para predecir el valor de `y`
- Mensaje al finalizar el entrenamiento

## Datos de entrenamiento

El modelo se entrena con 9 ejemplos que van desde `x = -6` hasta `x = 2`, siguiendo la fórmula: y = 2x + 6

## Cómo usar

1. Abrir el archivo `index.html` en el navegador.
2. ingresar un número en el cuadro de texto.
3. Hacer clic en el botón **"Entrenar modelo"**.
4. Esperá a que aparezca el mensaje:  *El modelo ya está entrenado y listo para usarse.*
5. Hacer clic en **"Predecir"** para ver el resultado de `y`.

## Archivos

- `index.html`: Interfaz del sitio
- `index.js`: Lógica de entrenamiento y predicción
