const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const PORT = 9000;

const express = require('express');
const app = express();

app.get('/train', (req, res) => {
	console.log(tf.version);

	tf.ready().then(() => {
		const message = `Loaded TF.js - version: ${tf.version.tfjs} with backend`;
		console.log(message);
		res.send(message);
	});
});

app.listen(PORT, (req, res) => {
	console.log(`App started on: ${PORT}`);
});
