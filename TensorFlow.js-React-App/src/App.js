import React, { useState } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs'; 

function App() {
    const [model, setModel] = useState(null);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState('');
    const [predictionResult, setPredictionResult] = useState('');
    const [numHiddenUnits, setNumHiddenUnits] = useState(1);
    const [learningRate, setLearningRate] = useState(0.1);
    const [numEpochs, setNumEpochs] = useState(100);
    const [inputValue, setInputValue] = useState('');

    async function trainModel() {
        setIsTraining(true);

        // Example training data (linear regression: y = 2x)
        const xs = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [8, 1]);
        const ys = tf.tensor2d([2, 4, 6, 8, 10, 12, 14, 16], [8, 1]); 

        // Create and compile the model
        const newModel = tf.sequential();
        newModel.add(tf.layers.dense({ units: parseInt(numHiddenUnits), activation: 'relu', inputShape: [1] }));
        newModel.add(tf.layers.dense({ units: 1 }));
        newModel.compile({
            optimizer: tf.train.sgd(parseFloat(learningRate)),
            loss: 'meanSquaredError'
        });

        // Start training with callbacks to update status
        try {
            const history = await newModel.fit(xs, ys, {
                epochs: parseInt(numEpochs),
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(5)}`);
                        setTrainingStatus(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(5)}`);
                    }
                }
            });
            console.log('Training Complete!');
            console.log(history.history.loss);
            setTrainingStatus('Training Complete!');
            setModel(newModel);
        } catch (error) {
            console.error('Training Error:', error);
            setTrainingStatus('Training Failed! Check console.');
        } finally {
            setIsTraining(false);
        }
    }
    
    async function predict() {
        if (!model) {
            console.log('Model not trained yet. Please train the model first.');
            return;
        }

        const input = parseFloat(inputValue);
        if (isNaN(input)) {
            console.log('Invalid input');
            setPredictionResult('Invalid input');
            return;
        }

        const prediction = model.predict(tf.tensor2d([[input]]));
        const output = await prediction.data();
        const predictedValue = output[0].toFixed(2);
        console.log('Prediction:', predictedValue);
        setPredictionResult(`Prediction: ${predictedValue}`);
    }

    return (
        <div className="container">
            <div>
                <label htmlFor="numHiddenUnits">Number of Hidden Units:</label>
                <input type="number" id="numHiddenUnits" value={numHiddenUnits} onChange={(e) => setNumHiddenUnits(e.target.value)} />
            </div>
            <div>
                <label htmlFor="learningRate">Learning Rate:</label>
                <input type="number" id="learningRate" step="0.01" value={learningRate} onChange={(e) => setLearningRate(e.target.value)} />
            </div>
            <div>
                <label htmlFor="numEpochs">Number of Epochs:</label>
                <input type="number" id="numEpochs" value={numEpochs} onChange={(e) => setNumEpochs(e.target.value)} />
            </div>
            <div>
                <button onClick={trainModel} disabled={isTraining}>Train model</button>
                <button onClick={predict}>Predict</button>
            </div>
            <div>
                <label htmlFor="input">Input:</label>
                <input type="number" id="input" value={inputValue} onChange={(e) => setInputValue(e.target.value)} />
            </div>
            <div id="predictionResult">{predictionResult}</div>
            <div id="trainingStatus">{trainingStatus}</div>
        </div>
    );
}

export default App;