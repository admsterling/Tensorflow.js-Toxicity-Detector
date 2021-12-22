import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

let EMBEDDING_SIZE = 6000;
const BATCH_SIZE = 16;
const render = true;
const TRAINING_EPOCHS = 15;
const MODEL_ID = 'toxicity-detector-tfidf';
const IDF_STORAGE_ID = 'toxicity-idfs';
const DICTIONARY_STORAGE_ID = 'toxicity-tfidf-dictionary';

const csvUrl = require('./data/train_tut.csv');
const file = require('./stop_words.txt');

export const train = () => {
	tf.tidy(() => {
		run();
	});
};

export const load = () => {
	return loadModel();
};

export const predict = async (sentence, model) => {
	console.log('predict using custom model');
	const stopwords = await readStopWords(file);
	return predictResults(sentence, stopwords, model);
};

const readStopWords = async (file) => {
	return await fetch(file).then((response) =>
		response.text().then((text) => text.split(/\r|\n/))
	);
};

let tmpDictionary = {};

const readRawData = async () => {
	return await tf.data.csv(csvUrl, {
		hasHeader: true,
		columnConfigs: {
			toxic: {
				isLabel: true,
			},
		},
	});
};

const plotOutputLabelCounts = (labels) => {
	const labelCounts = labels.reduce((acc, label) => {
		acc[label] = acc[label] === undefined ? 1 : (acc[label] += 1);
		return acc;
	}, {});

	const barChartData = [];
	Object.keys(labelCounts).forEach((key) => {
		barChartData.push({
			index: key,
			value: labelCounts[key],
		});
	});

	tfvis.render.barchart(
		{
			tab: 'Exploration',
			name: 'Toxic Output Labels',
		},
		barChartData
	);
};

const tokenize = (sentence, stopwords, isCreateDict = false) => {
	const tmpTokens = sentence.split(/\s+/g);
	const tokens = tmpTokens.filter(
		(token) => !stopwords.includes(token) && token.length > 0
	);
	if (isCreateDict) {
		tokens.reduce((acc, token) => {
			acc[token] = acc[token] === undefined ? 1 : (acc[token] += 1);
			return acc;
		}, tmpDictionary);
	}
	return tokens;
};

const sortDictionaryByValue = (dict) => {
	const items = Object.keys(dict).map((key) => {
		return [key, dict[key]];
	});
	return items.sort((first, second) => {
		return second[1] - first[1];
	});
};

const getInverseDocumentFrequency = (documentTokens, dictionary) => {
	return dictionary.map(
		(token) =>
			1 +
			Math.log(
				documentTokens.length /
					documentTokens.reduce(
						(acc, curr) => (curr.includes(token) ? acc + 1 : acc),
						0
					)
			)
	);
};

const encoder = (sentence, stopwords, dictionary, idfs) => {
	const tokens = tokenize(sentence, stopwords);
	const tfs = getTermFrequency(tokens, dictionary);
	const tfidfs = getTfIdf(tfs, idfs);
	return tfidfs;
};

const getTermFrequency = (tokens, dictionary) => {
	return dictionary.map((token) =>
		tokens.reduce((acc, curr) => (curr == token ? acc + 1 : acc), 0)
	);
};

const getTfIdf = (tfs, idfs) => {
	return tfs.map((element, index) => element * idfs[index]);
};

const prepareData = (stopwords, dictionary, idfs) => {
	const preprocess = ({ xs, ys }) => {
		const comment = xs['comment_text'];
		const trimmedComment = comment.toString().toLowerCase().trim();
		const encoded = encoder(trimmedComment, stopwords, dictionary, idfs);

		return {
			xs: tf.tensor2d([encoded], [1, dictionary.length]),
			ys: tf.tensor2d([ys['toxic']], [1, 1]),
		};
	};

	const readData = tf.data
		.csv(csvUrl, {
			columnConfigs: {
				toxic: {
					isLabel: true,
				},
			},
		})
		.map(preprocess);

	return readData;
};

const prepareDataUsingGenerator = (
	stopwords,
	comments,
	labels,
	dictionary,
	idfs
) => {
	function* getFeatures() {
		for (let i = 0; i < comments.length; i++) {
			const encoded = encoder(comments[i], stopwords, dictionary, idfs);
			yield tf.tensor2d([encoded], [1, dictionary.length]);
		}
	}
	function* getLabels() {
		for (let i = 0; i < labels.length; i++) {
			yield tf.tensor2d([labels[i]], [1, 1]);
		}
	}

	const xs = tf.data.generator(getFeatures);
	const ys = tf.data.generator(getLabels);
	const ds = tf.data.zip({ xs, ys });
	return ds;
};

const trainValTestSplit = (ds, nrows) => {
	const trainingValidationCount = Math.round(nrows * 0.7);
	const trainingCount = Math.round(nrows * 0.6);
	const SEED = 7687547;

	const trainingValidationData = ds
		.shuffle(nrows, SEED)
		.take(trainingValidationCount);

	const testDataset = ds
		.shuffle(nrows, SEED)
		.skip(trainingValidationCount)
		.batch(BATCH_SIZE);

	const trainingDataset = trainingValidationData
		.take(trainingCount)
		.batch(BATCH_SIZE);

	const validationDataset = trainingValidationData
		.skip(trainingCount)
		.batch(BATCH_SIZE);

	return {
		trainingDataset,
		validationDataset,
		testDataset,
	};
};

const buildModel = () => {
	console.log('Building...');
	const model = tf.sequential();
	model.add(
		tf.layers.dense({
			inputShape: [EMBEDDING_SIZE],
			activation: 'relu',
			units: 5,
		})
	);
	model.add(tf.layers.dense({ activation: 'sigmoid', units: 1 }));
	model.compile({
		loss: 'binaryCrossentropy',
		optimizer: tf.train.adam(0.06),
		metrics: ['accuracy'],
	});
	model.summary();
	return model;
};

const trainModel = async (model, trainingDataset, validationDataset) => {
	console.log('Training...');
	const history = [];
	const surface = { name: 'onEpochEnd Performance', tab: 'Training' };

	const batchHistory = [];
	const batchSurface = { name: 'onBatchEnd Performance', tab: 'Training' };

	const messageCallback = new tf.CustomCallback({
		onEpochEnd: async (epoch, logs) => {
			history.push(logs);
			console.log(`Epoch: ${epoch} / Loss: ${logs.loss}`);
			if (render) {
				tfvis.show.history(surface, history, [
					'loss',
					'val_loss',
					'acc',
					'val_acc',
				]);
			}
		},
		onBatchEnd: async (batch, logs) => {
			batchHistory.push(logs);
			if (render) {
				tfvis.show.history(batchSurface, batchHistory, ['loss', 'acc']);
			}
		},
	});

	const earlyStoppingCallback = tf.callbacks.earlyStopping({
		monitor: 'val_acc',
		minDelta: 0.3,
		patience: 5,
		verbose: 1,
	});

	const trainResult = await model.fitDataset(trainingDataset, {
		epochs: TRAINING_EPOCHS,
		validationData: validationDataset,
		callbacks: [messageCallback, earlyStoppingCallback],
	});
	return model;
};

const evaluateModel = async (model, testDataset) => {
	console.log('Evaluating...');
	const modelResult = await model.evaluateDataset(testDataset);

	const testLoss = modelResult[0].dataSync()[0];
	const testAcc = modelResult[1].dataSync()[0];

	console.log(`Loss on Test Dataset : ${testLoss.toFixed(4)}`);
	console.log(`Accuracy on Test Dataset : ${testAcc.toFixed(4)}`);
};

const getMoreEvaluationSummaries = async (model, testDataset) => {
	const allActualLables = [];
	const allPredictedLables = [];

	await testDataset.forEachAsync((row) => {
		const actualLabels = row['ys'].dataSync();
		actualLabels.forEach((x) => allActualLables.push(x));

		const features = row['xs'];
		const predict = model.predictOnBatch(tf.squeeze(features, 1));
		const predictLables = tf.round(predict).dataSync();
		predictLables.forEach((x) => allPredictedLables.push(x));
	});

	const allActualLablesTensor = tf.tensor1d(allActualLables);
	const allPredictedLablesTensor = tf.tensor1d(allPredictedLables);

	const accuracyResult = await tfvis.metrics.accuracy(
		allActualLablesTensor,
		allPredictedLablesTensor
	);

	console.log(`Accuracy result: ${accuracyResult}`);

	const perClassAccuracyResult = await tfvis.metrics.perClassAccuracy(
		allActualLablesTensor,
		allPredictedLablesTensor
	);

	console.log(
		`Per Class Accuracy result: ${JSON.stringify(
			perClassAccuracyResult,
			null,
			2
		)}`
	);

	const confusionMatrixResult = await tfvis.metrics.confusionMatrix(
		allActualLablesTensor,
		allPredictedLablesTensor
	);
	const confusionMatrixVizResult = {
		values: confusionMatrixResult,
	};

	const surface = {
		tab: 'Evaluation',
		name: 'Confusion Matrix',
	};
	if (render) {
		tfvis.render.confusionMatrix(surface, confusionMatrixVizResult);
	}
};

const exportModel = async (
	model,
	modelID,
	idfs,
	idfsStorageID,
	dictionary,
	dictionaryStorageID
) => {
	console.log('Exporting...');
	const modelPath = `localstorage://${modelID}`;
	const saveModelResults = await model.save(modelPath);
	console.log('Model Exported');

	localStorage.setItem(dictionaryStorageID, JSON.stringify(dictionary));
	localStorage.setItem(idfsStorageID, JSON.stringify(idfs));
	console.log('dictionary and idfs exported');

	return saveModelResults;
};

const loadModel = async () => {
	const modelPath = `localstorage://${MODEL_ID}`;
	console.log(modelPath);
	const models = await tf.io.listModels();
	if (models[modelPath]) {
		console.log('model exists');
		const model_loaded = await tf.loadLayersModel(modelPath);
		return model_loaded;
	} else {
		console.log('No model available');
		return null;
	}
};

const predictResults = (sentence, stopwords, model) => {
	console.log('Prediction started...');

	const idfObject = localStorage.getItem(IDF_STORAGE_ID);
	const idfs = JSON.parse(idfObject);

	const dictionaryObject = localStorage.getItem(DICTIONARY_STORAGE_ID);
	const dictionary = JSON.parse(dictionaryObject);

	const encoded = encoder(
		sentence.toString().toLowerCase().trim(),
		stopwords,
		dictionary,
		idfs
	);
	const encodedTensor = tf.tensor2d([encoded], [1, dictionary.length]);

	const predictionResult = model.predict(encodedTensor);
	const predictionScore = predictionResult.dataSync();

	const predictedClass = predictionScore >= 0.5 ? 'toxic' : 'non-toxic';
	const resultMessage = `Probability: ${predictionScore}, Class: ${predictedClass}`;
	console.log(resultMessage);
	return predictedClass;
};

const run = async () => {
	const rawDataResult = await readRawData();
	const stopwords = await readStopWords(file);

	const labels = [];

	const comments = [];
	const documentTokens = [];

	await rawDataResult.forEachAsync((row) => {
		const comment = row['xs']['comment_text'];
		const trimmedComment = comment.toString().trim().toLowerCase();
		comments.push(trimmedComment);
		documentTokens.push(tokenize(trimmedComment, stopwords, true));
		labels.push(row['ys']['toxic']);
	});

	if (render) {
		plotOutputLabelCounts(labels);
	}

	const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
	if (sortedTmpDictionary.length <= EMBEDDING_SIZE) {
		EMBEDDING_SIZE = sortedTmpDictionary.length;
	}
	const dictionary = sortedTmpDictionary.slice(0, EMBEDDING_SIZE).map((row) => {
		return row[0];
	});

	const idfs = getInverseDocumentFrequency(documentTokens, dictionary);

	// Prepare Data
	// const ds = prepareData(stopwords, dictionary, idfs);
	// await ds.forEachAsync((e) => {
	// 	console.log(e);
	// });

	// Prepare with Generator
	const ds = prepareDataUsingGenerator(
		stopwords,
		comments,
		labels,
		dictionary,
		idfs
	);

	const { trainingDataset, validationDataset, testDataset } = trainValTestSplit(
		ds,
		documentTokens.length
	);

	let model = buildModel();
	if (render) {
		tfvis.show.modelSummary(
			{
				name: 'Model Summary',
				tab: 'Model',
			},
			model
		);
	}

	model = await trainModel(model, trainingDataset, validationDataset);

	await evaluateModel(model, testDataset);
	await getMoreEvaluationSummaries(model, testDataset);

	const exportResult = await exportModel(
		model,
		MODEL_ID,
		idfs,
		IDF_STORAGE_ID,
		dictionary,
		DICTIONARY_STORAGE_ID
	);
};
