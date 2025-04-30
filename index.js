let numInput = document.getElementById("texto");
let btnEntrenar = document.getElementById("btn_entrenar");
let btnPredict = document.getElementById("predict");
let estado = document.getElementById("estado");
let resultado = document.getElementById("result");

let model; 

btnEntrenar.addEventListener("click", async function () {

  // Crear modelo secuencial
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // Compilar el modelo
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  // Generar datos desde x = -6 con 9 valores (x = -6 a 2)
  const x_vals = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const y_vals = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10],[9, 1]); // y = 2x + 6

  // Entrenar el modelo por al menos 350 épocas
  await model.fit(x_vals, y_vals, {
    epochs: 350,
    callbacks: {
      onTrainEnd: () => {
        estado.innerText =
          "El modelo ya está entrenado y listo para usarse.";
      },
    },
  });
});

btnPredict.addEventListener("click", function () {
  if (!model) {
    resultado.innerText = "es necesario entrenar primero al modelo.";
    return;
  }

  const valorX = parseFloat(numInput.value);
  if (isNaN(valorX)) {
    resultado.innerText = "el número ingresado es inválido.";
    return;
  }

  const inputTensor = tf.tensor2d([valorX], [1, 1]);
  const prediccion = model.predict(inputTensor);
  const valorY = prediccion.dataSync()[0];

  resultado.innerText = `Para x = ${valorX}, la predicción de y es: ${valorY.toFixed(
    2
  )}`;
});
