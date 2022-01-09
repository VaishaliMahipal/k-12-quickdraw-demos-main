const IMAGE_SIZE = 784;
const CLASSES = ['cat','sheep','apple','door','cake','triangle']
const k = 10;
let model;
let cnv;

async function loadMyModel() {
  model = await tf.loadLayersModel('model/model.json');
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

  // guessButton is the html button with the identifier as guess
  let guessButton = select('#guess');
  guessButton.mousePressed(guess);


  // clearButton is the html button with identifies as clear
  // when it is pressed, it will make the background color white
  let clearButton = select('#clear');
  clearButton.mousePressed(() => {
    background(255);
    select('#res').html('');
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
  select('#res').html(`I see: ${topKRes.toString()}`);
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



