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
    const topKRes = topKClassWIndex.map(i => `<br>${CLASSES[i.index]}(${(i.probability.toFixed(2) * 100)}%)`);
    select('#res').html(`I see: ${topKRes.toString()}`);


    const feedImage = getInputImage();

    console.log(feedImage)
    var input = []
    console.log(input)
    input.push(tf.tidy(() => { return tf.expandDims(feedImage,0).asType('float32')}));
    console.log(input);
    for (var i = 1; i <= 10; i++)
    {
        input.push(model.layers[i]);

    }
    
    console.log(input.length)
    const firstconv = input[1];
    const secondconv = input[3];
    const thirdconv = input[5];
    console.log(firstconv)
    const firstconv_list = tf.tidy(() => { return tf.unstack(firstconv.reshape([24,24,64]),2)});
    const secondconv_list = tf.tidy(() => { return tf.unstack(secondconv.reshape([10,10,64]),2)});
    const thirdconv_list = tf.tidy(() => { return tf.unstack(thirdconv.reshape([3,3,128]),2)})
    console.log(firstconv_list[1])

    for(var i = 0 ; i < 64 ; i++)
    {
        const reverse_img = tf.reverse2d(firstconv_list[i]);
        drawconv_map(Array.from(reverse_img.dataSync()),"fc_"+i,24,24,28,28);
        reverse_img.dispose();
    }

    for(var i = 0 ; i < 64 ; i++)
    {
        const reverse_img = tf.reverse2d(secondconv_list[i]);
        drawconv_map(Array.from(reverse_img.dataSync()),"sc_"+i,10,10,24,24);
        reverse_img.dispose();
    }

    for(var i = 0 ; i < 128 ; i++)
    {
        const reverse_img = tf.reverse2d(thirdconv_list[i]);
        drawconv_map(Array.from(reverse_img.dataSync()),"tc_"+i,3,3,10,10);
        reverse_img.dispose();

    }

    feedImage.dispose();
    firstconv.dispose();
    secondconv.dispose();
    thirdconv.dispose();

    for(var i = 1 ; i < 64 ; i++)
    {
        firstconv_list[i].dispose();
    }
    for (var i = 0; i < 64; i++)
    {
        secondconv_list[i].dispose();

    }
    for (var i = 0; i < 128; i++)
    {
        thirdconv_list[i].dispose();
    }

    for (var i = input.length - 1; i >= 0; i--) {
        input[i].dispose();

    }
}

function getInputImage() {

    let inputs = [];
    // p5 function, get image from the canvas
    let img = get();
    img.resize(28, 28);
    img.loadPixels();
    // Group data into [[[i00] [i01], [i02], [i03], ..., [i027]], .... [[i270], [i271], ... , [i2727]]]]
    let oneRow = [];
    for (let i = 0; i < IMAGE_SIZE; i++)
    {
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


/*Array.prototype.reshape = function(rows, cols)
{

var copy = this.slice(0); 

this.length = 0; 




for (var r = 0; r < rows; r++)
{

var row = [];

for (var c = 0; c < cols; c++)
{

var i = r * cols + c;

if (i < copy.length) {

row.push(copy[i]);

}

}

this.push(row);

}

};*/







function drawconv_map(x,elements_name,reshape_x,reshape_y,width_plot,height_plot)

{


x.reshape(reshape_x,reshape_y);

var data = [

{

z: x,

type: 'heatmap',

colorscale: 'YIGnBu',

showlegend: false,

showarrow: false,

showscale: false,

showgrid : false

}

];




var layout = {

autosize: false,

width: width_plot,

height: height_plot,

margin: {

l: 10,

r: 10,

b: 10,

t: 10,

pad: 4

},

paper_bgcolor: 'rgba(0,0,0,0)',

plot_bgcolor: 'rgba(0,0,0,0)',

showlegend: false,

xaxis: {visible: false},

yaxis: {visible: false},




};




Plotly.newPlot(elements_name, data , layout);

}







