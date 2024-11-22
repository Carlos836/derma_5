// After loading the model we want to make a prediction on the default image
// So the user will see predictions when the page first loads

function simulateClick(tabID) {
    document.getElementById(tabID).click();
}

function predictOnLoad() {
    // Simulate a click on the predict button
    setTimeout(simulateClick.bind(null,'predict-button'), 500);
}

$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#prediction-list").empty();
    }
    
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
    
    // Simulate a click on the predict button
    // This introduces a 0.5 second delay before the click
    // Without this delay, the model loads but can't predict automatically
    setTimeout(simulateClick.bind(null,'predict-button'), 500);
});

let model;
(async function () {
    try {
        // Updated model loading for TF.js 2.18.0
        model = await tf.loadLayersModel('model_kerasnative_v4/model.json');
        $("#selected-image").attr("src", "assets/samplepic.jpg");
        
        // Hide the model loading spinner
        $('.progress-bar').hide();
        
        // Simulate a click on the predict button
        predictOnLoad();
    } catch (error) {
        console.error('Error loading the model:', error);
    }
})();

$("#predict-button").click(async function () {
    try {
        let image = $('#selected-image').get(0);
        
        // Pre-process the image
        image.width = 600;
        image.height = 450;

        // Updated image processing for TF.js 2.18.0
        let tensor = tf.browser.fromPixels(image)
            .resizeNearestNeighbor([224,224])
            .toFloat();
        
        let offset = tf.scalar(127.5);
        
        tensor = tensor.sub(offset)
            .div(offset)
            .expandDims();

        // Updated prediction handling for TF.js 2.18.0
        let predictions = await model.predict(tensor).data();
        
        // Memory management: dispose of tensor when done
        tensor.dispose();
        offset.dispose();

        let top5 = Array.from(predictions)
            .map(function (p, i) {
                return {
                    probability: p,
                    className: TARGET_CLASSES[i]
                };
            }).sort(function (a, b) {
                return b.probability - a.probability;
            }).slice(0, 6);

        $("#prediction-list").empty();
        top5.forEach(function (p) {
            $("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
        });
    } catch (error) {
        console.error('Error during prediction:', error);
        $("#prediction-list").empty();
        $("#prediction-list").append(`<li>Error during prediction: ${error.message}</li>`);
    }
});

