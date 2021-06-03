let net;

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');


async function app() {
    console.log('cargando mobilenet..');
  
    // cargando el model.
    net = await mobilenet.load();
    console.log('Modelo cargado correctamente');
  
    // Tensorflow.js  APU de datos que podría capturar la imagen 
    // desde la cámara web como Tensor. 
    const webcam = await tf.data.webcam(webcamElement);
  
    // Captura una imagen de la cámara web y la asocia con una clase específica
    const addExample = async classId => {
      // captura imagen de la camara
      const img = await webcam.capture();
  
      // algoritmo KNN clasificador mas cercano.
      const activation = net.infer(img, true);
  
      // Pase la activación intermedia al clasificador.
      classifier.addExample(activation, classId);
  
      img.dispose();
    };
  
    //button
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
  
    while (true) {
      if (classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
  
        //activation from mobilenet
        //modulo clasificador
        const activation = net.infer(img, 'conv_preds');        
        const result = await classifier.predictClass(activation);
  
        const classes = ['A', 'B', 'C'];
        document.getElementById('console').innerText = `
          prediccion: ${classes[result.label]}\n
          provabilidad: ${result.confidences[result.label]}
        `;
  
        // tensor disponible
        img.dispose();
      }
  
      await tf.nextFrame();
    }
  }

app();
