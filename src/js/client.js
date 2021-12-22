import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import '@tensorflow/tfjs-backend-wasm';
import 'regenerator-runtime/runtime';

import * as model from '../assets/model';
import * as modelTransfer from '../assets/model_transfer';
import * as modelToxicity from '../assets/model_toxicity';

import $ from 'jquery';
import 'materialize-css';
import 'material-icons';

M.AutoInit();

const init = async () => {
	await tf.ready();

	const init_message = `Powered by Tensorflow.js - version: ${
		tf.version.tfjs
	} with backed : ${tf.getBackend()}`;
	$('#init').text(init_message);
};

init();

let modelOptionSelect = $('select');
let modelOption = 1;
modelOptionSelect.on('change', (e) => {
	modelOption = parseInt(e.target.value);
});

$('#btn_visor').on('click', () => {
	tfvis.visor().toggle();
});

$('#btn_train').on('click', async () => {
	switch (modelOption) {
		case 1:
			$('#btn_train').addClass('disabled');
			M.toast({ html: 'Training Started' });
			console.log('Training custom model with TFIDF features');
			await model.train();
			break;
		case 2:
			$('#btn_train').addClass('disabled');
			M.toast({ html: 'Training Started' });
			console.log('Training custom model on pre-trained USE Model');
			await modelTransfer.train();
			break;
		case 3:
			M.toast({ html: 'No training needed' });
			break;
		default:
			break;
	}
});

let model_loaded;

$('#btn_load').on('click', async () => {
	switch (modelOption) {
		case 1:
			console.log('Loading TF.js trained model');
			model_loaded = await model.load();
			model_loaded.summary();
			if (model_loaded) {
				M.toast({ html: 'Model Loaded' });
			}
			break;
		case 2:
			console.log('Loading pre-trained transfer learning model');
			model_loaded = await modelTransfer.load();
			model_loaded.summary();
			if (model_loaded) {
				M.toast({ html: 'Model Loaded' });
			}
			break;
		case 3:
			console.log('Loading pre-trained toxicity model');
			model_loaded = await modelToxicity.load();
			console.log('Loaded');
			break;
		default:
			break;
	}

	$('#btn_load').addClass('disabled');
	$('#btn_predict').removeClass('disabled');
});

$('#btn_predict').on('click', async () => {
	const message = $('#textarea-message').val();
	if (message.trim().length <= 0) {
		M.toast({ html: 'Empty Message: Nothing to predict' });
		return;
	}
	console.log(`Here is the message: ${message}`);

	$('#chip_result').empty();
	$('#btn_predict').addClass('disabled');
	let predictedClass;
	switch (modelOption) {
		case 1:
			predictedClass = await model.predict(message, model_loaded);
			$('#chip_result').append(
				`Predicted Label: <div class="chip pink-text">${predictedClass}</div>`
			);
			break;
		case 2:
			predictedClass = await modelTransfer.predict(message, model_loaded);
			$('#chip_result').append(
				`Predicted Label: <div class="chip pink-text">${predictedClass}</div>`
			);
			break;
		case 3:
			predictedClass = await modelToxicity.predict(message, model_loaded);
			$('#chip_result').append('<span>Predicted Label(s): </span>');
			predictedClass.forEach((element) => {
				$('#chip_result').append(
					`<div class="chip pink-text">${element}</div>`
				);
			});

			break;
		default:
			break;
	}

	$('#btn_predict').removeClass('disabled');
});
