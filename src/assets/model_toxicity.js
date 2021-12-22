import * as tf from '@tensorflow/tfjs';
import * as toxicity from '@tensorflow-models/toxicity';

const threshold = 0.8;

export const load = () => {
	return loadModel();
};

export const predict = (sentence, model) => {
	console.log('predict using Toxicity model');
	return predictResults(sentence, model);
};

const loadModel = async () => {
	const model = await toxicity.load(threshold);
	return model;
};

const predictResults = async (sentence, model) => {
	console.log('Prediction Started');

	const predictionResult = await model.classify(sentence.toLowerCase().trim());
	console.log(predictionResult);

	let predictedClasses = [];
	predictionResult.forEach((e) => {
		if (e['results'][0]['match'] == true) {
			predictedClasses.push(e['label']);
		}
	});
	if (predictedClasses.length <= 0) {
		predictedClasses.push('non-toxic');
	}

	console.log(predictedClasses);
	return predictedClasses;
};
