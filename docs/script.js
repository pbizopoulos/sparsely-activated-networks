"use strict";
(() => {
	const motifMaxNum = 3;
	const neuronMaxNum = 3;
	tf.env().set("WEBGL_DELETE_TEXTURE_THRESHOLD", 0);
	const activationFunctionArray = ["none", "extrema"];
	const arrowMargin = 5;
	const circleRadius = 15;
	const exampleObject = {
		noisyOneMotif: noisyOneMotif,
		sanConvEncoder: sanConvEncoder,
		sanResize: sanResize,
	};
	const height = 150;
	const inputColor = "#1f77b4";
	const inputMotifTypeArray = ["constant", "cos", "random", "sin", "triangle"];
	const inputNoiseTypeObject = {
		normal: tf.randomNormal,
		uniform: tf.randomUniform,
	};
	const inputSizeMax = parseInt(d3.select("#size-input-range").property("max"));
	const kernelInitializationArray = ["constant", "normal", "uniform"];
	const lossFunctionObject = {
		huber: tf.losses.huberLoss,
		mae: tf.losses.absoluteDifference,
		mse: tf.losses.meanSquaredError,
	};
	const motifColorArray = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];
	const motifIndexArray = [...Array(motifMaxNum).keys()];
	const neuronColorArray = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"];
	const neuronIndexArray = [...Array(neuronMaxNum).keys()];
	const optimizerObject = {
		adadelta: tf.train.adadelta,
		adagrad: tf.train.adagrad,
		adam: tf.train.adam,
		adamax: tf.train.adamax,
		rmsprop: tf.train.rmsprop,
		sgd: tf.train.sgd,
	};
	const reconstructionColor = "#ff7f0e";
	const referenceFunctionArray = ["absolute topk", "subsample bilinear", "subsample nn"];
	const resizeFunctionObject = {
		bilinear: tf.image.resizeBilinear,
		nn: tf.image.resizeNearestNeighbor,
	};
	const width = 150;
	let epoch = 0;
	let input = null;
	let inputChannelCurrentIndex = 0;
	let inputReconstructionDescriptionLength = 0;
	let interval = null;
	let learningRateExponent = parseFloat(d3.select("#learning-rate-exponent-input-range").property("value"));
	let lossFunctionKey = d3.select("#loss-function-select").property("value");
	let neuronArray = new Array(neuronMaxNum).fill(null);
	let neuronCurrentIndex = 0;
	let optimizerKey = d3.select("#optimizer-select").property("value");
	let referenceAction = null;
	let referenceArray = null;
	let referenceFunction = null;
	let referenceReconstructionLoss = null;
	let referenceReupsampledReconstructionLossArray = null;
	let sampleFirstCurrent = 0;
	let zeroReconstructionLoss = null;

	function addNeuronKernelWeightAndVisualizations(index) {
		tf.tidy(() => {
			switch (neuronArray[index].kernel.initialization) {
				case "constant":
					neuronArray[index].kernel.weights = tf.variable(tf.fill([neuronArray[index].kernel.size, 1, 1], neuronArray[index].kernel.amplitude));
					break;
				case "normal":
					neuronArray[index].kernel.weights = tf.variable(tf.randomNormal([neuronArray[index].kernel.size, 1, 1], 0, neuronArray[index].kernel.amplitude));
					break;
				case "uniform":
					neuronArray[index].kernel.weights = tf.variable(tf.randomUniform([neuronArray[index].kernel.size, 1, 1], -neuronArray[index].kernel.amplitude, neuronArray[index].kernel.amplitude));
					break;
			}
		});
		activationsSvg
			.append("path")
			.attr("id", `activation-path-${index}`)
			.style("fill", "none")
			.style("stroke", neuronColorArray[index]);
		activationsSvg
			.append("path")
			.attr("id", `similarity-transparent-path-${index}`)
			.style("fill", "none")
			.style("stroke", neuronColorArray[index])
			.style("opacity", "0.4");
		kernelSvg
			.append("path")
			.attr("d", line(neuronArray[index].kernel.weights.arraySync()))
			.attr("id", `kernel-path-${index}`)
			.style("fill", "none")
			.style("stroke", neuronColorArray[index]);
		kernelReconstructionsSvg
			.append("path")
			.attr("id", `kernel-reconstructions-path-${index}`)
			.style("fill", "none")
			.style("stroke", neuronColorArray[index]);
		ndnlSvg
			.append("circle")
			.attr("id", `ndnl-neuron-circle-${index}`)
			.attr("r", 2)
			.style("fill", neuronColorArray[index]);
		ndnlSvg
			.append("line")
			.attr("id", `ndnl-neuron-line-${index}`)
			.style("stroke", neuronColorArray[index]);
		similaritiesSvg
			.append("path")
			.attr("id", `similarity-path-${index}`)
			.style("fill", "none")
			.style("stroke", neuronColorArray[index]);
	}

	function applyDistanceMin(x, distanceMin) {
		const areaAllowed = new Array(x.length).fill(0);
		const sorted = tf.topk(tf.abs(x), x.length);
		const indicesSorted = sorted.indices.arraySync();
		const valuesSorted = tf.gather(x, indicesSorted).arraySync();
		const xWithDistanceMin = tf.buffer([x.length, 1]);
		for (let i = 0; i < indicesSorted.length; i++) {
			const indexSorted = indicesSorted[i];
			if (valuesSorted[i] && areaAllowed[indexSorted] === 0) {
				for (let j = 0; j <= distanceMin; j++) {
					areaAllowed[indexSorted - j] = x[indexSorted];
					if (indexSorted + j < x.length) {
						areaAllowed[indexSorted + j] = x[indexSorted];
					}
				}
				xWithDistanceMin.set(x[indexSorted], indexSorted, 0);
			}
		}
		return [xWithDistanceMin.toTensor(), areaAllowed];
	}

	function extrema1d(x) {
		const dx = tf.sub(x.slice(1, x.size - 1), x.slice(0, x.size - 1));
		const dxPadRightGreater = tf.greater(dx.pad([[0, 1]]), 0);
		const dxPadLeftLessEqual = tf.lessEqual(dx.pad([[1, 0]]), 0);
		const sign = tf.sub(1, tf.sign(x)).asType("bool");
		const valleys = tf.logicalAnd(tf.logicalAnd(dxPadRightGreater, dxPadLeftLessEqual), sign);
		const peaks = tf.logicalAnd(tf.logicalAnd(tf.logicalNot(dxPadRightGreater), tf.logicalNot(dxPadLeftLessEqual)), tf.logicalNot(sign));
		const peaksValleys = tf.logicalOr(peaks, valleys);
		return tf.mul(peaksValleys, x);
	}

	function generateAndProcessReference() {
		tf.dispose(referenceArray);
		tf.dispose(referenceReupsampledReconstructionLossArray);
		tf.tidy(() => {
			referenceArray = [tf.keep(tf.zerosLike(input.data))];
			const referenceDescriptionLengthArray = [0];
			const referenceReconstructionLossArray = [lossFunctionObject[lossFunctionKey](input.data, tf.zerosLike(input.data)).dataSync()];
			const step = 20;
			for (let i = step; i < input.data.size; i += step) {
				let reference = null;
				let downsampled = null;
				const sorted = tf.topk(input.data.abs().squeeze(-1), i);
				switch (referenceFunction) {
					case "absolute topk":
						reference = tf.scatterND(sorted.indices, tf.gather(input.data, sorted.indices).squeeze(-1), [input.data.size]).expandDims(-1).expandDims(-1);
						break;
					case "subsample bilinear":
						downsampled = tf.image.resizeBilinear(input.data.expandDims(-1), [i, 1]);
						reference = tf.image.resizeBilinear(downsampled, input.data.shape);
						break;
					case "subsample nn":
						downsampled = tf.image.resizeNearestNeighbor(input.data.expandDims(-1), [i, 1]);
						reference = tf.image.resizeNearestNeighbor(downsampled, input.data.shape);
						break;
				}
				referenceArray.push(tf.keep(reference));
				referenceDescriptionLengthArray.push(i);
				referenceReconstructionLossArray.push(lossFunctionObject[lossFunctionKey](input.data.expandDims(-1), reference).dataSync());
			}
			referenceArray.push(tf.keep(input.data));
			referenceDescriptionLengthArray.push(input.data.size - 1);
			referenceReconstructionLossArray.push(new Float32Array([0.0]));
			referenceReupsampledReconstructionLossArray = tf.keep(tf.image.resizeBilinear(tf.tensor(referenceReconstructionLossArray).expandDims(-1), input.data.shape, true));
			ndnlY.domain([0, 2 * referenceReupsampledReconstructionLossArray.dataSync()[0]]);
			d3.select("#ndnl-path").attr("d", ndnlLine(referenceReupsampledReconstructionLossArray.dataSync()));
			d3.select("#reference-path").attr("d", null);
			ndnlSvg.selectAll("#ndnl-reference-circle").remove();
			ndnlSvg
				.selectAll("#ndnl-reference-circle")
				.data(referenceDescriptionLengthArray)
				.enter()
				.append("circle")
				.attr("id", "ndnl-reference-circle")
				.attr("fill", "cyan")
				.attr("cx", (d) => {
					return ndnlX(d);
				})
				.attr("cy", (d) => {
					return ndnlY(referenceReupsampledReconstructionLossArray.dataSync()[d]);
				})
				.attr("r", 2)
				.on("mouseover", (event) => {
					d3.select("#reference-path").attr("d", line(referenceArray[Math.round(event.currentTarget.__data__ / step)].dataSync()));
				})
				.on("mouseout", () => {
					d3.select("#reference-path").attr("d", null);
				});
		});
		referenceReconstructionLoss = referenceReupsampledReconstructionLossArray.dataSync()[inputReconstructionDescriptionLength];
		if (inputReconstructionDescriptionLength > input.size) {
			referenceReconstructionLoss = 0;
		}
		zeroReconstructionLoss = referenceReupsampledReconstructionLossArray.dataSync()[0] * input.size;
		referenceAction = zeroReconstructionLoss * inputReconstructionDescriptionLength + referenceReconstructionLoss * input.size * input.size;
		d3.select("#reference-action-text").text(referenceAction.toFixed(4));
		d3.select("#reference-reconstruction-loss-text").text((referenceReconstructionLoss * input.size).toFixed(4));
		d3.select("#reference-zero-reconstruction-loss-text").text(zeroReconstructionLoss.toFixed(4));
	}

	function generateInputChannelData(index, motifDistancesInitialize, motifAmplitudesInitialize) {
		tf.dispose(input.channelArray[index].data);
		const motifAppearencesNum = inputSizeMax;
		tf.tidy(() => {
			if (motifDistancesInitialize) {
				tf.dispose(input.channelArray[index].distances);
				input.channelArray[index].distances = tf.keep(tf.randomUniform([motifAppearencesNum], 0, 1));
			}
			if (motifAmplitudesInitialize) {
				tf.dispose(input.channelArray[index].amplitudes);
				input.channelArray[index].amplitudes = tf.keep(tf.randomUniform(input.channelArray[index].distances.shape, 0, 1));
			}
			const indices = input.channelArray[index].distances
				.mul(input.channelArray[index].distanceMax)
				.add(input.channelArray[index].distanceMin)
				.cumsum()
				.cast("int32");
			const updates = input.channelArray[index].amplitudes.mul(input.channelArray[index].amplitudeMax).add(input.channelArray[index].amplitudeBase);
			const motifPositions = tf.scatterND(indices, updates, [inputSizeMax]).expandDims(-1);
			let motifData = null;
			const elementsNum = Math.round(input.channelArray[index].motifSize / 2);
			switch (input.channelArray[index].motifType) {
				case "constant":
					motifData = tf.fill([input.channelArray[index].motifSize, 1, 1], 1);
					break;
				case "cos":
					motifData = tf
						.cos(tf.linspace(0, 2 * Math.PI, input.channelArray[index].motifSize))
						.expandDims(-1)
						.expandDims(-1);
					break;
				case "random":
					motifData = tf.randomNormal([input.channelArray[index].motifSize, 1, 1]);
					break;
				case "sin":
					motifData = tf
						.sin(tf.linspace(0, 2 * Math.PI, input.channelArray[index].motifSize))
						.expandDims(-1)
						.expandDims(-1);
					break;
				case "triangle":
					motifData = tf
						.linspace(0, 1 - 1 / elementsNum, elementsNum)
						.concat(tf.linspace(1, 1 / elementsNum, elementsNum))
						.expandDims(-1)
						.expandDims(-1);
					break;
			}
			input.channelArray[index].data = tf.keep(tf.conv1d(motifPositions, motifData, 1, "same"));
		});
	}

	function getNeuronDescriptionLength(index) {
		return tf.min([2 * neuronArray[index].activation.data.asType("bool").sum().arraySync(), neuronArray[index].activation.data.size]).arraySync() + neuronArray[index].kernel.size;
	}

	function processInputChannelsData(noiseInitialize) {
		tf.dispose(input.data);
		tf.tidy(() => {
			input.data = tf.zeros([inputSizeMax, 1]);
			for (let i = 0; i < motifMaxNum; i++) {
				if (input.channelArray[i].use && input.channelArray[i].data) {
					input.data = input.data.add(input.channelArray[i].data);
				}
			}
			if (noiseInitialize) {
				tf.dispose(input.dataNoise);
				input.dataNoise = tf.keep(inputNoiseTypeObject[input.noiseTypeKey](input.data.shape, 0, 1));
			}
			if (input.dataNoise) {
				input.data = input.data.add(input.dataNoise.mul(input.noiseSigma));
			}
			if (input.standardize) {
				input.data = input.data.sub(input.data.min());
				input.data = input.data.div(input.data.max());
				input.data = input.data.mul(2).sub(1);
			}
			if (sampleFirstCurrent > input.data.size - input.size) {
				sampleFirstCurrent = 0;
			}
			input.data = input.data.slice(sampleFirstCurrent, input.size);
			sampleFirstCurrent += input.velocity;
			const inputDownsampled = resizeFunctionObject[input.resizeFunctionKey](input.data.expandDims(-1), [Math.ceil(input.data.size * input.resizeMultiplier), 1]);
			const inputReupsampled = resizeFunctionObject[input.resizeFunctionKey](inputDownsampled, [input.data.size, 1]).squeeze(-1);
			input.data = tf.keep(inputReupsampled.mul(input.quantizationStatesNum).round().div(input.quantizationStatesNum));
			d3.select("#input-path").attr("d", line(input.data.arraySync()));
			d3.select("#input-transparent-path").attr("d", line(input.data.arraySync()));
		});
	}

	function removeNeuronKernelWeightAndVisualizations(index) {
		if (neuronArray[index]) {
			tf.dispose(neuronArray[index].kernel.weights);
		}
		d3.select(`#ndnl-neuron-circle-${index}`).remove();
		d3.select(`#ndnl-neuron-line-${index}`).remove();
		d3.select(`#activation-path-${index}`).remove();
		d3.select(`#kernel-path-${index}`).remove();
		d3.select(`#kernel-reconstructions-path-${index}`).remove();
		d3.select(`#similarity-path-${index}`).remove();
		d3.select(`#similarity-transparent-path-${index}`).remove();
	}

	function train() {
		const startTime = performance.now();
		if (input.noiseInitialize || input.velocity !== 0) {
			processInputChannelsData(input.noiseInitialize);
			generateAndProcessReference();
		}
		inputReconstructionDescriptionLength = 0;
		const neuronActivationAreaAllowedArray = [];
		tf.tidy(() => {
			const optimizer = optimizerObject[optimizerKey](10 ** learningRateExponent);
			let inputReconstruction = null;
			if (input.data) {
				inputReconstruction = tf.zerosLike(input.data);
			}
			if (neuronArray.filter((neuron) => neuron.kernel.weights).filter((neuron) => neuron.kernel.weights.trainable).length === 0) {
				return;
			}
			const { value, grads } = tf.variableGrads(() => {
				for (let i = 0; i < neuronMaxNum; i++) {
					if (neuronArray[i].use) {
						neuronArray[i].kernel.weightsResized = resizeFunctionObject[neuronArray[i].kernel.resizeFunction](neuronArray[i].kernel.weights, [Math.ceil(neuronArray[i].kernel.size * neuronArray[i].kernel.resizeMultiplier), 1]);
						if (neuronArray[i].convEncoderUse) {
							neuronArray[i].similarity = tf.conv1d(input.data, neuronArray[i].kernel.weightsResized, neuronArray[i].kernel.stride, "same");
							neuronArray[i].similarity = resizeFunctionObject[neuronArray[i].kernel.strideResizeFunction](neuronArray[i].similarity.expandDims(-1), [input.data.size, 1]).squeeze(-1);
						} else {
							neuronArray[i].similarity = input.data;
						}
						neuronArray[i].activation.data = neuronArray[i].similarity.where(
							tf.logicalOr(neuronArray[i].similarity.greaterEqual(neuronArray[i].similarity.max().mul(neuronArray[i].activation.amplitudeMin)), neuronArray[i].similarity.lessEqual(neuronArray[i].similarity.max().mul(-neuronArray[i].activation.amplitudeMin))),
							tf.zerosLike(neuronArray[i].similarity),
						);
						switch (neuronArray[i].activation.function_) {
							case "none":
								break;
							case "extrema":
								neuronArray[i].activation.data = extrema1d(neuronArray[i].activation.data.squeeze());
								break;
						}
						[neuronArray[i].activation.data, neuronActivationAreaAllowedArray[i]] = applyDistanceMin(neuronArray[i].activation.data.squeeze().arraySync(), neuronArray[i].activation.distanceMin);
					} else {
						neuronActivationAreaAllowedArray[i] = new Array(input.data.size).fill(0);
					}
				}
				const neuronActivationAreaAllowedStackedArray = tf.stack(neuronActivationAreaAllowedArray);
				const neuronActivationAreaAllowedPostArray = tf.unstack(tf.oneHot(tf.argMax(neuronActivationAreaAllowedStackedArray.abs()), neuronMaxNum).expandDims(-1), 1);
				for (let i = 0; i < neuronMaxNum; i++) {
					if (neuronArray[i].use) {
						if (neuronArray[i].activation.regulated) {
							neuronArray[i].activation.data = neuronArray[i].activation.data.mul(neuronActivationAreaAllowedPostArray[i]);
						}
						neuronArray[i].reconstruction = tf.conv1d(neuronArray[i].activation.data, neuronArray[i].kernel.weightsResized, 1, "same");
						inputReconstruction = tf.add(inputReconstruction, neuronArray[i].reconstruction);
						d3.select(`#ndnl-neuron-circle-${i}`)
							.attr("cx", ndnlX(getNeuronDescriptionLength(i)))
							.attr("cy", ndnlY(lossFunctionObject[lossFunctionKey](input.data, neuronArray[i].reconstruction).dataSync()));
						d3.select(`#ndnl-neuron-line-${i}`)
							.attr("x1", ndnlX(getNeuronDescriptionLength(i)))
							.attr("y1", ndnlY(0))
							.attr("x2", ndnlX(getNeuronDescriptionLength(i)))
							.attr("y2", ndnlY(lossFunctionObject[lossFunctionKey](input.data, neuronArray[i].reconstruction).dataSync()));
						d3.select(`#activation-path-${i}`).attr("d", line(neuronArray[i].activation.data.arraySync()));
						d3.select(`#kernel-path-${i}`).attr("d", line(neuronArray[i].kernel.weights.arraySync()));
						d3.select(`#kernel-reconstructions-path-${i}`).attr("d", line(neuronArray[i].reconstruction.arraySync()));
						d3.select(`#similarity-path-${i}`).attr("d", line(neuronArray[i].similarity.arraySync()));
						d3.select(`#similarity-transparent-path-${i}`).attr("d", line(neuronArray[i].similarity.arraySync()));
						inputReconstructionDescriptionLength += getNeuronDescriptionLength(i);
					}
				}
				const inputReconstructionLoss = lossFunctionObject[lossFunctionKey](input.data, inputReconstruction);
				const tmp = lossFunctionObject[lossFunctionKey](inputReconstruction, tf.zerosLike(inputReconstruction)).arraySync();
				const action = tmp * input.size * inputReconstructionDescriptionLength + inputReconstructionLoss.arraySync() * input.size * input.size;
				d3.select("#action-text").text(action.toFixed(4));
				d3.select("#input-reconstruction-loss-text").text((input.size * inputReconstructionLoss.arraySync()).toFixed(4));
				d3.select("#input-reconstruction-path").attr("d", line(inputReconstruction.arraySync()));
				d3.select("#ndnl-circle").attr("cx", ndnlX(inputReconstructionDescriptionLength)).attr("cy", ndnlY(inputReconstructionLoss.arraySync()));
				d3.select("#ndnl-line").attr("x1", ndnlX(inputReconstructionDescriptionLength)).attr("y1", ndnlY(0)).attr("x2", ndnlX(inputReconstructionDescriptionLength)).attr("y2", ndnlY(inputReconstructionLoss.arraySync()));
				d3.select("#input-reconstruction-energy-text").text((tmp * input.size).toFixed(4));
				return inputReconstructionLoss;
			});
			console.log(value);
			d3.select("#description-length-text").text(inputReconstructionDescriptionLength);
			d3.select("#reference-description-length-text").text(inputReconstructionDescriptionLength);
			d3.select("#epoch-text").text(`epoch: ${epoch}`);
			d3.select("#time-per-epoch-text").text(`time/epoch: ${Math.round(performance.now() - startTime)}ms`);
			epoch++;
			optimizer.applyGradients(grads);
			optimizer.dispose();
		});
	}

	const arrowSvg = d3.select("#grid-container-div").append("svg").append("defs").append("marker").attr("id", "arrow").attr("refX", arrowMargin).attr("refY", arrowMargin).attr("markerWidth", width).attr("markerHeight", height).attr("orient", "auto");
	arrowSvg.append("path").attr(
		"d",
		d3.line()([
			[0, 0],
			[0, 2 * arrowMargin],
			[2 * arrowMargin, arrowMargin],
		]),
	);
	const x = d3.scaleLinear().range([0, width]);
	const y = d3.scaleLinear().domain([-1, 1]).range([height, 0]);
	const line = d3
		.line()
		.x((d, i) => x(i))
		.y((d) => y(d));
	const ndnlX = d3.scaleLinear().range([0, width]);
	const ndnlY = d3.scaleLinear().domain([0, 1]).range([height, 0]);
	const ndnlLine = d3
		.line()
		.x((d, i) => ndnlX(i))
		.y((d) => ndnlY(d));
	const activationFunctionSvg = d3.select("#activation-function-div").append("svg").attr("viewBox", [0, 0, width, height]);
	activationFunctionSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, 0],
				[width / 2, height / 4 - circleRadius - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	activationFunctionSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, height / 4 + circleRadius],
				[width / 2, height / 2 - circleRadius - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	activationFunctionSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, height / 2 + circleRadius],
				[width / 2, (3 * height) / 4 - circleRadius - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	activationFunctionSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, (3 * height) / 4 + circleRadius],
				[width / 2, height - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	activationFunctionSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "25%").attr("r", "15px");
	activationFunctionSvg.append("text").attr("x", "50%").attr("y", "25%").text("T\u2090");
	activationFunctionSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "50%").attr("r", "15px");
	activationFunctionSvg.append("text").attr("x", "50%").attr("y", "50%").text("\u03d5");
	activationFunctionSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "75%").attr("r", "15px");
	activationFunctionSvg.append("text").attr("x", "50%").attr("y", "75%").text("T\u2091");
	activationFunctionSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Activation function block. It is responsible for choosing which activations are kept.");
	});
	const activationsSvg = d3.select("#activations-div").append("svg").attr("viewBox", [0, 0, width, height]);
	activationsSvg.append("text").attr("x", "50%").attr("y", "10%").text("\u03B1");
	activationsSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Activations.");
	});
	const convDecoderSvg = d3.select("#conv-decoder-div").append("svg").attr("viewBox", [0, 0, width, height]);
	convDecoderSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2 - circleRadius, height / 2],
				[arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convDecoderSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width, height / 2],
				[width / 2 + circleRadius + arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convDecoderSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, 0],
				[width / 2, height / 2 - circleRadius - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convDecoderSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "50%").attr("r", "15px");
	convDecoderSvg.append("text").attr("x", "50%").attr("y", "50%").text("*");
	convDecoderSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "20%").attr("r", "15px");
	convDecoderSvg.append("text").attr("x", "50%").attr("y", "20%").text("\u2195");
	convDecoderSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Convolution decoder with a kernel resize option.");
	});
	const convEncoderSvg = d3.select("#conv-encoder-div").append("svg").attr("viewBox", [0, 0, width, height]);
	convEncoderSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[0, height / 2],
				[width / 2 - circleRadius - arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convEncoderSvg
		.append("path")
		.attr("id", "convEncoderSvgVisible")
		.attr(
			"d",
			d3.line()([
				[0, height / 2],
				[width - arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convEncoderSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, height],
				[width / 2, height / 2 + circleRadius + arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	convEncoderSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "50%").attr("r", "15px");
	convEncoderSvg.append("text").attr("x", "50%").attr("y", "50%").text("*");
	convEncoderSvg.append("circle").attr("class", "circle-operation").attr("cx", "75%").attr("cy", "50%").attr("r", "15px");
	convEncoderSvg.append("text").attr("x", "75%").attr("y", "50%").text("\u2195");
	convEncoderSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "80%").attr("r", "15px");
	convEncoderSvg.append("text").attr("x", "50%").attr("y", "80%").text("\u2195");
	convEncoderSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Convolutional Encoder with a kernel resize option.");
	});
	const horizontalLineSvg1 = d3.select("#horizontal-line-1-div").append("svg").attr("viewBox", [0, 0, width, height]);
	horizontalLineSvg1.append("line").attr("stroke", "black").attr("x1", "0%").attr("y1", "50%").attr("x2", "100%").attr("y2", "50%");
	horizontalLineSvg1.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "");
	});
	const horizontalLineSvg2 = d3.select("#horizontal-line-2-div").append("svg").attr("viewBox", [0, 0, width, height]);
	horizontalLineSvg2.append("line").attr("stroke", "black").attr("x1", "0%").attr("y1", "50%").attr("x2", "100%").attr("y2", "50%");
	horizontalLineSvg2.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "");
	});
	const inputSvg = d3.select("#input-div").append("svg").attr("viewBox", [0, 0, width, height]);
	inputSvg.append("text").attr("x", "50%").attr("y", "10%").text("x");
	inputSvg.append("path").attr("id", "input-path").style("fill", "none").style("stroke", inputColor);
	inputSvg.on("mouseover", (event) => {
		event.currentTarget.style.cursor = "crosshair";
		d3.select("#help-div").property("innerHTML", "Input data. Click and drag to change. For all plots the x axis range is [0, n] and the y axis range [-1, 1].");
	});
	inputSvg.on("mouseout", (event) => {
		event.currentTarget.style.cursor = "default";
	});
	d3.select("#input-div").call(
		d3.drag().on("start", (event) => {
			event.on("drag", (event) => {
				const buffer = tf.buffer(input.data.shape, input.data.dtype, input.data.dataSync());
				buffer.set(3 - (2 * event.y) / height, Math.round((input.data.size * event.x) / width), 0);
				tf.dispose(input.data);
				input.data = buffer.toTensor();
				d3.select("#input-path").attr("d", line(input.data.arraySync()));
				d3.select("#input-transparent-path").attr("d", line(input.data.arraySync()));
				generateAndProcessReference();
			});
		}),
	);
	const kernelSvg = d3.select("#kernel-div").append("svg").attr("viewBox", [0, 0, width, height]);
	kernelSvg.append("text").attr("x", "50%").attr("y", "10%").text("w\u1d62");
	kernelSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Learned kernels.");
	});
	const kernelReconstructionsSvg = d3.select("#kernel-reconstruction-div").append("svg").attr("viewBox", [0, 0, width, height]);
	kernelReconstructionsSvg.append("text").attr("x", "50%").attr("y", "10%").text("w\u1d62*\u03B1");
	kernelReconstructionsSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Kernel reconstructions.");
	});
	const lossSvg = d3.select("#loss-div").append("svg").attr("viewBox", [0, 0, width, height]);
	lossSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, 0],
				[width / 2, height / 2 - circleRadius - arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	lossSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2, height],
				[width / 2, height / 2 + circleRadius + arrowMargin],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	lossSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2 + circleRadius, height / 2],
				[width - arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	lossSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "50%").attr("r", "15px");
	lossSvg.append("text").attr("x", "50%").attr("y", "50%").text("L");
	lossSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "The loss L of the reconstruction w.r.t. the input.");
	});
	const ndnlSvg = d3.select("#ndnl-div").append("svg").attr("viewBox", [0, 0, width, height]);
	ndnlSvg.append("path").attr("id", "ndnl-path").style("fill", "none").style("stroke", inputColor).style("visibility", "hidden");
	ndnlSvg.append("circle").attr("id", "ndnl-circle").attr("r", 2).style("fill", reconstructionColor);
	ndnlSvg.append("line").attr("id", "ndnl-line").style("stroke", reconstructionColor);
	ndnlSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "||d<sub>0->x\u0302</sub>|| and ||x - x\u0302|| on the x and y axis respectively normalized. Orange corresponds to SAN while cyan to reference functions for various t.");
	});
	const reconstructionSvg = d3.select("#reconstruction-div").append("svg").attr("viewBox", [0, 0, width, height]);
	reconstructionSvg.append("text").attr("x", "50%").attr("y", "10%").text("x\u0302");
	reconstructionSvg.append("path").attr("id", "input-transparent-path").style("fill", "none").style("stroke", inputColor).style("opacity", "0.4");
	reconstructionSvg.append("path").attr("id", "input-reconstruction-path").style("fill", "none").style("stroke", reconstructionColor);
	reconstructionSvg.append("path").attr("id", "reference-path").style("fill", "none").style("stroke", "cyan");
	reconstructionSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Reconstruction of SAN. When mouse is over the cyan circles in the ||d<sub>0->x\u0302</sub>||, ||x - x\u0302|| diagram, the cyan plot is the reference function.");
	});
	const similaritiesSvg = d3.select("#similarities-div").append("svg").attr("viewBox", [0, 0, width, height]);
	similaritiesSvg.append("text").attr("id", "similaritiesTextSvg").attr("x", "50%").attr("y", "10%");
	similaritiesSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Result after the convolution encoder. The physical meaning of the amplitudes is the similarity of the motif with each part of the input.");
	});
	const sumSvg = d3.select("#sum-div").append("svg").attr("viewBox", [0, 0, width, height]);
	sumSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width / 2 + circleRadius, height / 2],
				[arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	sumSvg
		.append("path")
		.attr(
			"d",
			d3.line()([
				[width, height / 2],
				[width / 2 + circleRadius + arrowMargin, height / 2],
			]),
		)
		.attr("stroke", "black")
		.attr("marker-end", "url(#arrow)");
	sumSvg.append("circle").attr("class", "circle-operation").attr("cx", "50%").attr("cy", "50%").attr("r", "15px");
	sumSvg.append("text").attr("x", "50%").attr("y", "50%").text("+");
	sumSvg.on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Summation operator, which is applied to the set of individual kernel reconstructions.");
	});
	d3.select("#activation-amplitude-min-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].activation.amplitudeMin = parseFloat(event.currentTarget.value);
		d3.select("#activation-amplitude-min-text").html(`min amp (T\u2090): ${neuronArray[neuronCurrentIndex].activation.amplitudeMin}`);
	});
	d3.select("#activation-amplitude-min-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the amplitude threshold.");
	});
	d3.select("#activation-distance-min-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].activation.distanceMin = parseFloat(event.currentTarget.value);
		d3.select("#activation-distance-min-text").html(`min dist (T\u2091): ${neuronArray[neuronCurrentIndex].activation.distanceMin}`);
	});
	d3.select("#activation-distance-min-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the minimum distance between the non-zero activations.");
	});
	d3.select("#activation-function-select")
		.selectAll("option")
		.data(activationFunctionArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#activation-function-select").on("input", (event) => {
		neuronArray[neuronCurrentIndex].activation.function_ = event.currentTarget.value;
	});
	d3.select("#activation-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the activation function.");
	});
	d3.select("#activation-regulated-input-checkbox").on("change", (event) => {
		neuronArray[neuronCurrentIndex].activation.regulated = event.currentTarget.checked;
	});
	d3.select("#activation-regulated-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether this kernel's activation is regulated by other regulating kernels.");
	});
	d3.select("#activation-regulates-input-checkbox").on("change", (event) => {
		neuronArray[neuronCurrentIndex].activation.regulates = event.currentTarget.checked;
	});
	d3.select("#activation-regulates-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether this kernel's activation regulates other regulated kernels.");
	});
	d3.select("#advanced-input-checkbox").on("change", (event) => {
		let visibility;
		if (event.currentTarget.checked) {
			visibility = "visible";
		} else {
			visibility = "hidden";
		}
		d3.select("#input-control-1-div").style("visibility", visibility);
		d3.select("#input-control-2-div").style("visibility", visibility);
		d3.select("#input-control-3-div").style("visibility", visibility);
		d3.select("#input-control-4-div").style("visibility", visibility);
		d3.select("#input-control-5-div").style("visibility", visibility);
		d3.select("#neuron-control-1-div").style("visibility", visibility);
		d3.select("#neuron-control-2-div").style("visibility", visibility);
		d3.select("#neuron-control-3-div").style("visibility", visibility);
		d3.select("#loss-control-div").style("visibility", visibility);
		d3.select("#activation-regulates-label").style("visibility", visibility);
		d3.select("#activation-regulated-label").style("visibility", visibility);
		d3.select("#reference-control-div").style("visibility", visibility);
		d3.select("#ndnl-path").style("visibility", visibility);
		d3.selectAll("#ndnl-reference-circle").style("visibility", visibility);
	});
	d3.select("#advanced-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Toggles visibility of simple/advanced view.");
	});
	d3.select("#channel-amplitude-base-input-range").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].amplitudeBase = parseFloat(event.currentTarget.value);
		d3.select("#channel-amplitude-base-text").html(`base amp: ${input.channelArray[inputChannelCurrentIndex].amplitudeBase}`);
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, false, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#channel-amplitude-base-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the base amplitude.");
	});
	d3.select("#channel-amplitude-max-input-range").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].amplitudeMax = parseFloat(event.currentTarget.value);
		d3.select("#channel-amplitude-max-text").html(`max amp: ${input.channelArray[inputChannelCurrentIndex].amplitudeMax}`);
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, false, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#channel-amplitude-max-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the max amplitude.");
	});
	d3.select("#channel-distance-max-input-range").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].distanceMax = parseInt(event.currentTarget.value);
		d3.select("#channel-distance-max-text").html(`max dist: ${input.channelArray[inputChannelCurrentIndex].distanceMax}`);
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, true, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#channel-distance-max-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the maximum distance between motif instances.");
	});
	d3.select("#channel-distance-min-input-range").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].distanceMin = parseInt(event.currentTarget.value);
		d3.select("#channel-distance-min-text").html(`min dist: ${input.channelArray[inputChannelCurrentIndex].distanceMin}`);
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, true, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#channel-distance-min-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the mininum distance between motif instances.");
	});
	d3.select("#channel-motif-size-input-range").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].motifSize = parseInt(event.currentTarget.value);
		d3.select("#channel-motif-size-text").html(`size: ${input.channelArray[inputChannelCurrentIndex].motifSize}`);
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, false, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#channel-motif-size-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the motif size.");
	});
	d3.select("#channel-use-input-checkbox").on("change", (event) => {
		if (event.currentTarget.checked) {
			input.channelArray[inputChannelCurrentIndex].use = true;
			generateInputChannelData(inputChannelCurrentIndex, false, false);
		} else {
			input.channelArray[inputChannelCurrentIndex].use = false;
		}
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#channel-use-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether the current motif is used.");
	});
	d3.select("#conv-encoder-use-input-checkbox").on("change", (event) => {
		if (event.currentTarget.checked) {
			convEncoderSvg.style("visibility", "visible");
			d3.select("#similaritiesTextSvg").text("w\u1d62*x");
		} else {
			convEncoderSvg.style("visibility", "hidden");
			d3.select("#convEncoderSvgVisible").style("visibility", "visible");
			d3.select("#similaritiesTextSvg").text("x");
		}
		neuronArray[neuronCurrentIndex].convEncoderUse = event.currentTarget.checked;
	});
	d3.select("#conv-encoder-use-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether the convolution encoder is used.");
	});
	d3.select("#description-length-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Description length of the SAN representation.");
	});
	d3.select("#loss-description-length-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Description length of the loss of the SAN representation.");
	});
	d3.select("#example-select")
		.selectAll("option")
		.data(Object.keys(exampleObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#example-select").on("change", (event) => {
		for (let i = 0; i < neuronMaxNum; i++) {
			removeNeuronKernelWeightAndVisualizations(i);
		}
		if (input) {
			tf.dispose(input.data);
			for (let i = 0; i < motifMaxNum; i++) {
				tf.dispose(input.channelArray[i].data);
				tf.dispose(input.channelArray[i].amplitudes);
				tf.dispose(input.channelArray[i].distances);
			}
		}
		epoch = 0;
		sampleFirstCurrent = 0;
		d3.select("#description-length-text").text("null");
		d3.select("#epoch-text").text(`epoch: ${epoch}`);
		d3.select("#action-text").text("null");
		d3.select("#input-reconstruction-path").attr("d", null);
		d3.select("#ndnl-circle").attr("cx", null).attr("cy", null);
		d3.select("#ndnl-line").attr("x1", null).attr("y1", null).attr("x2", null).attr("y2", null);
		d3.select("#input-reconstruction-loss-text").text("null");
		d3.select("#reference-action-text").text("null");
		d3.select("#reference-reconstruction-loss-text").text("null");
		d3.select("#time-per-epoch-text").text("time/epoch: 0 ms");
		d3.select("#input-reconstruction-energy-text").text("null");
		({ input, learningRateExponent, lossFunctionKey, neuronArray, optimizerKey, referenceFunction } = exampleObject[event.currentTarget.value]);
		for (let i = 0; i < motifMaxNum; i++) {
			generateInputChannelData(i, true, true);
			d3.select("#input-channel-index-select").property("value", i).dispatch("change");
			d3.select("#channel-use-input-checkbox")
				.property("checked", input.channelArray[i].use)
				.dispatch("change");
			d3.select("#channel-amplitude-base-input-range")
				.property("value", input.channelArray[i].amplitudeBase)
				.dispatch("input");
			d3.select("#channel-amplitude-max-input-range")
				.property("value", input.channelArray[i].amplitudeMax)
				.dispatch("input");
			d3.select("#channel-distance-max-input-range")
				.property("value", input.channelArray[i].distanceMax)
				.dispatch("input");
			d3.select("#channel-distance-min-input-range")
				.property("value", input.channelArray[i].distanceMin)
				.dispatch("input");
			d3.select("#channel-motif-size-input-range")
				.property("value", input.channelArray[i].motifSize)
				.dispatch("input");
			d3.select("#input-channel-motif-type-select")
				.property("value", input.channelArray[i].motifType)
				.dispatch("change");
		}
		d3.select("#input-channel-index-select").property("value", 0).dispatch("change");
		d3.select("#noise-initialize-input-checkbox").property("checked", input.noiseInitialize).dispatch("change");
		d3.select("#standardize-input-checkbox").property("checked", input.standardize).dispatch("change");
		d3.select("#noise-sigma-input-range").property("value", input.noiseSigma).dispatch("input");
		d3.select("#quantization-states-num-input-range").property("value", input.quantizationStatesNum).dispatch("input");
		d3.select("#resize-multiplier-input-range").property("value", input.resizeMultiplier).dispatch("input");
		d3.select("#size-input-range").property("value", input.size).dispatch("click");
		d3.select("#velocity-input-range").property("value", input.velocity).dispatch("input");
		d3.select("#learning-rate-exponent-input-range").property("value", learningRateExponent).dispatch("input");
		d3.select("#input-noise-type-select").property("value", input.noiseTypeKey).dispatch("change");
		d3.select("#input-resize-function-select").property("value", input.resizeFunctionKey).dispatch("change");
		d3.select("#loss-function-select").property("value", lossFunctionKey).dispatch("change");
		d3.select("#optimizer-select").property("value", optimizerKey).dispatch("change");
		d3.select("#reference-function-select").property("value", referenceFunction).dispatch("change");
		for (let i = 0; i < neuronMaxNum; i++) {
			d3.select("#neuron-index-select").property("value", i).dispatch("change");
			d3.select("#activation-regulated-input-checkbox")
				.property("checked", neuronArray[i].activation.regulated)
				.dispatch("change");
			d3.select("#activation-regulates-input-checkbox")
				.property("checked", neuronArray[i].activation.regulates)
				.dispatch("change");
			d3.select("#conv-encoder-use-input-checkbox")
				.property("checked", neuronArray[i].convEncoderUse)
				.dispatch("change");
			d3.select("#activation-amplitude-min-input-range")
				.property("value", neuronArray[i].activation.amplitudeMin)
				.dispatch("input");
			d3.select("#activation-distance-min-input-range")
				.property("value", neuronArray[i].activation.distanceMin)
				.dispatch("input");
			d3.select("#kernel-amplitude-input-range").property("value", neuronArray[i].kernel.amplitude);
			d3.select("#kernel-resize-multiplier-input-range")
				.property("value", neuronArray[i].kernel.resizeMultiplier)
				.dispatch("input");
			d3.select("#kernel-size-input-range").property("value", neuronArray[i].kernel.size);
			d3.select("#kernel-stride-input-range")
				.property("value", neuronArray[i].kernel.stride)
				.dispatch("input");
			d3.select("#activation-function-select")
				.property("value", neuronArray[i].activation.function_)
				.dispatch("change");
			d3.select("#kernel-initialization-select").property("value", neuronArray[i].kernel.initialization);
			d3.select("#kernel-resize-function-select")
				.property("value", neuronArray[i].kernel.resizeFunction)
				.dispatch("change");
			d3.select("#stride-resize-function-select")
				.property("value", neuronArray[i].kernel.strideResizeFunction)
				.dispatch("change");
			d3.select("#neuron-use-input-checkbox")
				.property("checked", neuronArray[i].use)
				.dispatch("change");
		}
		d3.select("#neuron-index-select").property("value", 0).dispatch("change");
		d3.select("#advanced-input-checkbox").dispatch("change");
	});
	d3.select("#example-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Choose a predefined example.");
	});
	d3.select("#action-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Action of the SAN representation.");
	});
	d3.select("#help-div").property("innerHTML", "You can get started by choosing one of the predefined examples (bottom right) and then press the start button.");
	d3.select("#help-div").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Displays help for each UI element.");
	});
	d3.select("#input-channel-index-select")
		.selectAll("option")
		.data(motifIndexArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#input-channel-index-select").on("change", () => {
		inputChannelCurrentIndex = parseInt(d3.select("#input-channel-index-select").property("value"));
		d3.select("#channel-use-input-checkbox").property("checked", input.channelArray[inputChannelCurrentIndex].use);
		d3.select("#channel-amplitude-base-input-range").property("value", input.channelArray[inputChannelCurrentIndex].amplitudeBase);
		d3.select("#channel-amplitude-max-input-range").property("value", input.channelArray[inputChannelCurrentIndex].amplitudeMax);
		d3.select("#channel-distance-max-input-range").property("value", input.channelArray[inputChannelCurrentIndex].distanceMax);
		d3.select("#channel-distance-min-input-range").property("value", input.channelArray[inputChannelCurrentIndex].distanceMin);
		d3.select("#channel-motif-size-input-range").property("value", input.channelArray[inputChannelCurrentIndex].motifSize);
		d3.select("#input-channel-motif-type-select").property("value", input.channelArray[inputChannelCurrentIndex].motifType);
		d3.select("#input-channel-motif-colored-box-svg").style("fill", motifColorArray[inputChannelCurrentIndex]);
		d3.select("#channel-amplitude-base-text").html(`base amp: ${input.channelArray[inputChannelCurrentIndex].amplitudeBase}`);
		d3.select("#channel-amplitude-max-text").html(`max amp: ${input.channelArray[inputChannelCurrentIndex].amplitudeMax}`);
		d3.select("#channel-distance-max-text").html(`max dist: ${input.channelArray[inputChannelCurrentIndex].distanceMax}`);
		d3.select("#channel-distance-min-text").html(`min dist: ${input.channelArray[inputChannelCurrentIndex].distanceMin}`);
		d3.select("#channel-motif-size-text").html(`size: ${input.channelArray[inputChannelCurrentIndex].motifSize}`);
	});
	d3.select("#input-channel-index-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the motif index. It sets the options in this block and the right next to this.");
	});
	d3.select("#input-channel-motif-colored-box-svg").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Indicates the color of the currently selected motif.");
	});
	d3.select("#input-channel-motif-type-select")
		.selectAll("option")
		.data(inputMotifTypeArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#input-channel-motif-type-select").on("input", (event) => {
		input.channelArray[inputChannelCurrentIndex].motifType = event.currentTarget.value;
		if (input.channelArray[inputChannelCurrentIndex].use) {
			generateInputChannelData(inputChannelCurrentIndex, false, false);
			processInputChannelsData(false);
			generateAndProcessReference();
		}
	});
	d3.select("#input-channel-motif-type-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the type of the motif.");
	});
	d3.select("#input-noise-type-select")
		.selectAll("option")
		.data(Object.keys(inputNoiseTypeObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#input-noise-type-select").on("change", (event) => {
		input.noiseTypeKey = event.currentTarget.value;
		processInputChannelsData(true);
		generateAndProcessReference();
	});
	d3.select("#input-noise-type-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the type of the noise.");
	});
	d3.select("#input-resize-function-select")
		.selectAll("option")
		.data(Object.keys(resizeFunctionObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#input-resize-function-select").on("change", (event) => {
		input.resizeFunctionKey = event.currentTarget.value;
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#input-resize-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the input resize function.");
	});
	d3.select("#kernel-amplitude-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.amplitude = parseFloat(event.currentTarget.value);
		d3.select("#kernel-amplitude-text").html(`amplitude: ${neuronArray[neuronCurrentIndex].kernel.amplitude}`);
		if (neuronArray[neuronCurrentIndex].use) {
			removeNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
			addNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
		}
	});
	d3.select("#kernel-amplitude-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel amplitude.");
	});
	d3.select("#kernel-initialization-select")
		.selectAll("option")
		.data(kernelInitializationArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#kernel-initialization-select").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.initialization = event.currentTarget.value;
		if (neuronArray[neuronCurrentIndex].use) {
			removeNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
			addNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
		}
	});
	d3.select("#kernel-initialization-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel initialization type.");
	});
	d3.select("#kernel-resize-function-select")
		.selectAll("option")
		.data(Object.keys(resizeFunctionObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#kernel-resize-function-select").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.resizeFunction = event.currentTarget.value;
	});
	d3.select("#kernel-resize-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel resize function.");
	});
	d3.select("#kernel-resize-multiplier-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.resizeMultiplier = parseFloat(event.currentTarget.value);
		d3.select("#kernel-resize-multiplier-text").html(`resize x: ${neuronArray[neuronCurrentIndex].kernel.resizeMultiplier}`);
	});
	d3.select("#kernel-resize-multiplier-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel resize multiplier.");
	});
	d3.select("#kernel-size-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.size = parseInt(event.currentTarget.value);
		d3.select("#kernel-size-text").html(`size: ${neuronArray[neuronCurrentIndex].kernel.size}`);
		if (neuronArray[neuronCurrentIndex].use) {
			removeNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
			addNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
		}
	});
	d3.select("#kernel-size-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel resize.");
	});
	d3.select("#kernel-stride-input-range").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.stride = parseInt(event.currentTarget.value);
		d3.select("#kernel-stride-text").html(`stride: ${neuronArray[neuronCurrentIndex].kernel.stride}`);
	});
	d3.select("#kernel-stride-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the kernel stride.");
	});
	d3.select("#learning-rate-exponent-input-range").on("input", (event) => {
		learningRateExponent = parseFloat(event.currentTarget.value);
		d3.select("#learning-rate-text").html(`lr: 10<sup>${learningRateExponent}</sup>`);
	});
	d3.select("#learning-rate-exponent-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the learning rate.");
	});
	d3.select("#loss-function-select")
		.selectAll("option")
		.data(Object.keys(lossFunctionObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#loss-function-select").on("change", (event) => {
		lossFunctionKey = event.currentTarget.value;
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#loss-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the loss function.");
	});
	d3.select("#neuron-colored-box-svg").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Indicates the color of the currently selected neuron.");
	});
	d3.select("#neuron-index-select")
		.selectAll("option")
		.data(neuronIndexArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#neuron-index-select").on("change", () => {
		neuronCurrentIndex = parseInt(d3.select("#neuron-index-select").property("value"));
		d3.select("#neuron-use-input-checkbox").property("checked", neuronArray[neuronCurrentIndex].use);
		d3.select("#activation-regulated-input-checkbox").property("checked", neuronArray[neuronCurrentIndex].activation.regulated);
		d3.select("#activation-regulates-input-checkbox").property("checked", neuronArray[neuronCurrentIndex].activation.regulates);
		d3.select("#conv-encoder-use-input-checkbox")
			.property("checked", neuronArray[neuronCurrentIndex].convEncoderUse)
			.dispatch("change");
		d3.select("#activation-amplitude-min-input-range").property("value", neuronArray[neuronCurrentIndex].activation.amplitudeMin);
		d3.select("#activation-distance-min-input-range").property("value", neuronArray[neuronCurrentIndex].activation.distanceMin);
		d3.select("#kernel-amplitude-input-range").property("value", neuronArray[neuronCurrentIndex].kernel.amplitude);
		d3.select("#kernel-resize-multiplier-input-range").property("value", neuronArray[neuronCurrentIndex].kernel.resizeMultiplier);
		d3.select("#kernel-size-input-range").property("value", neuronArray[neuronCurrentIndex].kernel.size);
		d3.select("#kernel-stride-input-range").property("value", neuronArray[neuronCurrentIndex].kernel.stride);
		d3.select("#activation-function-select").property("value", neuronArray[neuronCurrentIndex].activation.function_);
		d3.select("#kernel-initialization-select").property("value", neuronArray[neuronCurrentIndex].kernel.initialization);
		d3.select("#kernel-resize-function-select").property("value", neuronArray[neuronCurrentIndex].kernel.resizeFunction);
		d3.select("#stride-resize-function-select").property("value", neuronArray[neuronCurrentIndex].kernel.strideResizeFunction);
		d3.select("#neuron-colored-box-svg").style("fill", neuronColorArray[neuronCurrentIndex]);
		d3.select("#activation-amplitude-min-text").html(`min amp (T\u2090): ${neuronArray[neuronCurrentIndex].activation.amplitudeMin}`);
		d3.select("#activation-distance-min-text").html(`min dist (T\u2091): ${neuronArray[neuronCurrentIndex].activation.distanceMin}`);
		d3.select("#kernel-amplitude-text").html(`amplitude: ${neuronArray[neuronCurrentIndex].kernel.amplitude}`);
		d3.select("#kernel-resize-multiplier-text").html(`resize x: ${neuronArray[neuronCurrentIndex].kernel.resizeMultiplier}`);
		d3.select("#kernel-size-text").html(`size: ${neuronArray[neuronCurrentIndex].kernel.size}`);
		d3.select("#kernel-stride-text").html(`stride: ${neuronArray[neuronCurrentIndex].kernel.stride}`);
	});
	d3.select("#neuron-index-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the neuron index. It sets the options in the first four dark grey blocks.");
	});
	d3.select("#neuron-use-input-checkbox").on("change", (event) => {
		if (event.currentTarget.checked) {
			addNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
			neuronArray[neuronCurrentIndex].use = true;
		} else {
			removeNeuronKernelWeightAndVisualizations(neuronCurrentIndex);
			neuronArray[neuronCurrentIndex].use = false;
			if (neuronArray.every((v) => v.use === false)) {
				d3.select("#example-select").dispatch("change");
			}
		}
	});
	d3.select("#neuron-use-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether the current kernel is used.");
	});
	d3.select("#noise-initialize-input-checkbox").on("change", (event) => {
		input.noiseInitialize = event.currentTarget.checked;
	});
	d3.select("#noise-initialize-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether the noise changes dynamically each epoch by resampling.");
	});
	d3.select("#noise-sigma-input-range").on("input", (event) => {
		input.noiseSigma = parseFloat(event.currentTarget.value);
		d3.select("#noise-sigma-text").html(`\u03c3: ${input.noiseSigma}`);
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#noise-sigma-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the noise parametrer depending on the type.");
	});
	d3.select("#optimizer-select")
		.selectAll("option")
		.data(Object.keys(optimizerObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#optimizer-select").on("change", (event) => {
		optimizerKey = event.currentTarget.value;
	});
	d3.select("#optimizer-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the optimizer.");
	});
	d3.select("#quantization-states-num-input-range").on("input", (event) => {
		input.quantizationStatesNum = parseInt(event.currentTarget.value);
		d3.select("#quantization-states-num-text").html(`quant states: ${input.quantizationStatesNum}`);
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#quantization-states-num-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the number of quantization states of the amplitude.");
	});
	d3.select("#input-reconstruction-loss-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Reconstruction loss of the SAN representation.");
	});
	d3.select("#reference-action-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Action of the reference function.");
	});
	d3.select("#reference-function-select")
		.selectAll("option")
		.data(referenceFunctionArray)
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#reference-function-select").on("change", (event) => {
		referenceFunction = event.currentTarget.value;
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#reference-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the reference function.");
	});
	d3.select("#reference-reconstruction-loss-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Reconstruction loss of the reference function.");
	});
	d3.select("#resize-multiplier-input-range").on("input", (event) => {
		input.resizeMultiplier = parseFloat(event.currentTarget.value);
		d3.select("#resize-multiplier-text").html(`resize x: ${input.resizeMultiplier}`);
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#resize-multiplier-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the number of quantization states of the amplitude.");
	});
	d3.select("#size-input-range").on("click", (event) => {
		input.size = parseInt(event.currentTarget.value);
		d3.select("#size-text").html(input.size);
		d3.select("#loss-description-length-text").html(input.size);
		d3.select("#reference-loss-description-length-text").html(input.size);
		d3.select("#kernel-size-input-range").property("max", input.size - 3);
		d3.select("#kernel-stride-input-range").property("max", input.size);
		d3.select("#activation-distance-min-input-range").property("max", input.size);
		ndnlX.domain([0, 2 * input.size]);
		x.domain([0, input.size]);
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#size-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the size of the input.");
	});
	d3.select("#standardize-input-checkbox").on("change", (event) => {
		input.standardize = event.currentTarget.checked;
		processInputChannelsData(false);
		generateAndProcessReference();
	});
	d3.select("#standardize-input-checkbox").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls whether the input is standardized in the range [-1, 1].");
	});
	d3.select("#start-pause-button").on("click", (event) => {
		if (event.currentTarget.textContent === "start") {
			if (neuronArray.every((v) => v.use === false)) {
				return;
			}
			event.currentTarget.textContent = "pause";
			interval = setInterval(() => {
				train();
			}, 0);
		} else {
			event.currentTarget.textContent = "start";
			clearInterval(interval);
		}
	});
	d3.select("#start-pause-button").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Starts/pauses training.");
	});
	d3.select("#stop-button").on("click", () => {
		d3.select("#example-select").dispatch("change");
		const startPauseButton = document.getElementById("start-pause-button");
		startPauseButton.textContent = "start";
		clearInterval(interval);
	});
	d3.select("#stop-button").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Stops training and resets selected example.");
	});
	d3.select("#stride-resize-function-select")
		.selectAll("option")
		.data(Object.keys(resizeFunctionObject))
		.enter()
		.append("option")
		.text((d) => d);
	d3.select("#stride-resize-function-select").on("input", (event) => {
		neuronArray[neuronCurrentIndex].kernel.strideResizeFunction = event.currentTarget.value;
	});
	d3.select("#stride-resize-function-select").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the stride resize function.");
	});
	d3.select("#velocity-input-range").on("input", (event) => {
		input.velocity = parseInt(event.currentTarget.value);
		d3.select("#velocity-text").html(`velocity: ${input.velocity}`);
	});
	d3.select("#velocity-input-range").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Controls the velocity of the input during training.");
	});
	d3.select("#input-reconstruction-energy-text").on("mouseover", () => {
		d3.select("#help-div").property("innerHTML", "Energy of the reconstruction.");
	});
	d3.select("#stop-button").dispatch("click");
})();
