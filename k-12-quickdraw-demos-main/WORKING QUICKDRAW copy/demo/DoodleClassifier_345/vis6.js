// Trained on 1 filters - 8 filters of 3 by 3 
const IMAGE_SIZE = 784;
const CLASSES = ['cat','sheep','apple','door','cake','triangle']
const k = 10;
let model;
let cnv;

async function loadMyModel() {
  model = await tf.loadLayersModel('model3/model.json');
  model.summary();
}

function setup() {
  loadMyModel();

  // creates a canvas to draw on 
  cnv = createCanvas(280, 280);

  // background color is white
  background(255);


  // each time the mouse is released on the canvas, the guess function will be issued
  cnv.mouseReleased(guess);
  cnv.parent('canvasContainer');

  
  let clearButton = select('#clear');
  clearButton.mousePressed(() => {
    background(255);
    select('#outputLayer').html('');
  });
}

function guess() {
  // Get input image from the canvas
  const inputs = getInputImage();

  // Predict
  let guess = model.predict(tf.tensor([inputs]));

  // Format res to an array
  const rawProb = Array.from(guess.dataSync());

  // Get top K res with index and probability
  const rawProbWIndex = rawProb.map((probability, index) => {
    return {
      index,
      probability
    }
  });

  const sortProb = rawProbWIndex.sort((a, b) => b.probability - a.probability);
  const topKClassWIndex = sortProb.slice(0, k);
  const topKRes = topKClassWIndex.map(i => `<br>${CLASSES[i.index]} (${(i.probability.toFixed(2) * 100)}%)`);
  select('#outputLayer').html(` ${topKRes.toString()}`);
  
  const layer1 = model.getLayer('conv2d_2');
  console.log("layer1 =")
  console.log(layer1);
  
  var IdImage = [];
  for(var i = 0; i < inputs.length; i++)
  {
    IdImage = IdImage.concat(inputs[i]);
  }
  inputImage=tf.tensor2d(IdImage, [784, 1]);
  inputImage1=inputImage.reshape([1,28,28,1])
  nameL=[]
  nameL.push("conv2d_2")
  positionA=[]
  positionA.push("filters1")
  positionA.push("filters2")
  positionA.push("filters3")
  positionA.push("filters4")
  positionA.push("filters5")
  positionA.push("filters6")
  positionA.push("filters7")
  positionA.push("filters8")
  positionF=[]
  positionF.push("featureMaps1")
  positionF.push("featureMaps2")
  positionF.push("featureMaps3")
  positionF.push("featureMaps4")
  positionF.push("featureMaps5")
  positionF.push("featureMaps6")
  positionF.push("featureMaps7")
  positionF.push("featureMaps8")
 
  
  const { filters, filterActivations } = getActivationTable(inputImage1,nameL[0]);
  console.log("filters in guess function")
  console.log(filters)
  console.log("activations in guess function")
  //console.log(filterActivations)
  for (let i = 0; i < 8; i++) { 
    console.log(filterActivations[0][i])
    renderImage(positionA[i], filters[i], { width: 40, height: 40 })
    renderImage(positionF[i], filterActivations[0][i], { width: 40, height: 40 })

  }
  
  
}

 async function renderImage(container, tensor, imageOpts) {
     console.log("tensor")
     console.log(tensor)
    const resized = tf.tidy(() =>
      tf.image.resizeNearestNeighbor(tensor,
        [imageOpts.height, imageOpts.width]).clipByValue(0.0, 1.0)
    );
    const childs =document.getElementById(container); 
    
    
    const canvas =  childs.querySelector('canvas') || document.createElement('canvas');
    canvas.width = imageOpts.width;
    canvas.height = imageOpts.height;
    canvas.style = `margin: 4px; :${imageOpts.width}px; height:${imageOpts.height}px`;
    childs.appendChild(canvas);
    console.log("rsized")
    console.log(resized)
   
  
    await tf.browser.toPixels(resized, canvas);
    resized.dispose();
  }





function getActivationTable(image1,layerName) {
    const exampleImageSize = 28;

    const layer = model.getLayer(layerName);

    let filters = tf.tidy(() => layer.kernel.val.transpose([3, 0, 1, 2]).unstack());

    console.log("filters")
    console.log(filters)
    console.log(filters[0].shape[2])
    
    console.log("filters1")
    console.log(filters)

    // Get the activations
    const activations = tf.tidy(() => {
      return getActivation(image1, model, layer).unstack();
    });
    const activationImageSize = activations[0].shape[0]; // e.g. 24
    const numFilters = activations[0].shape[2]; // e.g. 8
    console.log("activationImage size=")
    console.log(activationImageSize)
    console.log("numFilters=")
    console.log(numFilters)


    const filterActivations = activations.map((activation, i) => {
      // activation has shape [activationImageSize, activationImageSize, i];
      const unpackedActivations = Array(numFilters).fill(0).map((_, i) =>
        activation.slice([0, 0, i], [activationImageSize, activationImageSize, 1])
      );

      // prepend the input image
      const inputExample = tf.tidy(() =>
      inputImage1.slice([i], [1]).reshape([exampleImageSize, exampleImageSize, 1]));

      //unpackedActivations.unshift(inputExample);
      return unpackedActivations;
    });

    return {
      filters,
      filterActivations,
    };
}

function getActivation(input, model, layer) {
    const activationModel = tf.model({
      inputs: model.input,
      outputs: layer.output,
    });

    return activationModel.predict(input);
  }

function getInputImage() {
  let inputs = [];
  // p5 function, get image from the canvas
  let img = get();
  img.resize(28, 28);
  img.loadPixels();

  // Group data into [[[i00] [i01], [i02], [i03], ..., [i027]], .... [[i270], [i271], ... , [i2727]]]]
  let oneRow = [];
  for (let i = 0; i < IMAGE_SIZE; i++) {
    let bright = img.pixels[i * 4];
    let onePix = [parseFloat((255 - bright) / 255)];
    oneRow.push(onePix);
    if (oneRow.length === 28) {
      inputs.push(oneRow);
      oneRow = [];
    }
  }

  return inputs;
}

function draw() {
  strokeWeight(10);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}



