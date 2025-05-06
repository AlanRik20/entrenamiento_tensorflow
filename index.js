let numInput = document.getElementById("texto");
let btnEntrenar = document.getElementById("btn_entrenar");
let btnPredict = document.getElementById("predict");
let estado = document.getElementById("estado");
let resultado = document.getElementById("result");
let infoPerdida = document.getElementById("info_perdida");

let model;
const hist_epoch = [];

btnEntrenar.addEventListener("click", async function () {
  model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const x_vals = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const y_vals = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]); // y = 2x + 6

  hist_epoch.length = 0;

  await model.fit(x_vals, y_vals, {
    epochs: 350,
    callbacks: {
      onTrainEnd: () => {
        estado.innerText = "Estado: Modelo entrenado correctamente";
        dibujarGrafico();
        mostrarInfoPerdida();
      },
      onEpochEnd: async (epochs, logs) => {
        hist_epoch.push({ epochs, loss: logs.loss });
        console.log(epochs + ': ' + logs.loss.toFixed(4));
      }
    },
  });
});

btnPredict.addEventListener("click", function () {
  if (!model) {
    resultado.innerText = "Es necesario entrenar primero al modelo.";
    return;
  }

  const valores = numInput.value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
  if (valores.length === 0) {
    resultado.innerText = "Ingresa al menos un valor numérico válido.";
    return;
  }

  const inputTensor = tf.tensor2d(valores, [valores.length, 1]);
  const predicciones = model.predict(inputTensor).dataSync();

  let resHtml = "<h4>Resultados:</h4><ul>";
  valores.forEach((x, i) => {
    resHtml += `<li>Para x = ${x}: y = ${predicciones[i].toFixed(2)}</li>`;
  });
  resHtml += "</ul>";
  resultado.innerHTML = resHtml;
});

// Gráfico con Chart.js
function dibujarGrafico() {
  const ctx = document.getElementById('lossChart').getContext('2d');
  const epochs = hist_epoch.map(p => p.epochs);
  const losses = hist_epoch.map(p => p.loss);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [{
        label: 'Pérdida (Loss)',
        data: losses,
        borderColor: 'cyan',
        backgroundColor: 'lightblue',
        tension: 0.1,
        pointRadius: 2
      }]
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'Época' } },
        y: { title: { display: true, text: 'Valor de pérdida' } }
      }
    }
  });
}

function mostrarInfoPerdida() {
  if (hist_epoch.length < 2) return;
  const inicial = hist_epoch[0].loss;
  const final = hist_epoch[hist_epoch.length - 1].loss;
  const reduccion = 100 * (1 - final / inicial);
  infoPerdida.innerText = `Pérdida inicial: ${inicial.toFixed(4)}, Pérdida final: ${final.toFixed(4)} (Reducción: ${reduccion.toFixed(2)}%)`;
}
