const sanConvEncoder = {
	"input": {
		"data": null,
		"dataNoise": null,
		"channelArray": [{
			"amplitudeBase": 0.5,
			"amplitudeMax": 0.5,
			"amplitudes": null,
			"data": null,
			"distanceMax": 20,
			"distanceMin": 10,
			"distances": null,
			"motifSize": 30,
			"motifType": "cos",
			"use": true
		}, {
			"amplitudeBase": 0.5,
			"amplitudeMax": 0,
			"amplitudes": null,
			"data": null,
			"distanceMax": 20,
			"distanceMin": 10,
			"distances": null,
			"motifSize": 10,
			"motifType": "cos",
			"use": false
		}, {
			"amplitudeBase": 0.5,
			"amplitudeMax": 0,
			"amplitudes": null,
			"data": null,
			"distanceMax": 20,
			"distanceMin": 10,
			"distances": null,
			"motifSize": 10,
			"motifType": "cos",
			"use": false
		}],
		"noiseInitialize": false,
		"noiseSigma": 0,
		"noiseTypeKey": "normal",
		"quantizationStatesNum": 100,
		"resizeFunctionKey": "bilinear",
		"resizeMultiplier": 1,
		"size": 100,
		"standardize": false,
		"velocity": 0
	},
	"learningRateExponent": -1,
	"lossFunctionKey": "mse",
	"neuronArray": [{
		"activation": {
			"amplitudeMin": 0,
			"data": null,
			"distanceMin": 8,
			"function_": "none",
			"regulated": false,
			"regulates": false
		},
		"convEncoderUse": true,
		"kernel": {
			"amplitude": 1,
			"initialization": "constant",
			"resizeFunction": "bilinear",
			"resizeMultiplier": 1,
			"size": 15,
			"stride": 1,
			"strideResizeFunction": "bilinear"
		},
		"use": true
	}, {
		"activation": {
			"amplitudeMin": 0,
			"data": null,
			"distanceMin": 0,
			"function_": "none",
			"regulated": false,
			"regulates": false
		},
		"convEncoderUse": false,
		"kernel": {
			"amplitude": 1,
			"initialization": "constant",
			"resizeFunction": "bilinear",
			"resizeMultiplier": 1,
			"size": 1,
			"stride": 1,
			"strideResizeFunction": "bilinear"
		},
		"use": false
	}, {
		"activation": {
			"amplitudeMin": 0,
			"data": null,
			"distanceMin": 0,
			"function_": "none",
			"regulated": false,
			"regulates": false
		},
		"convEncoderUse": false,
		"kernel": {
			"amplitude": 1,
			"initialization": "constant",
			"resizeFunction": "bilinear",
			"resizeMultiplier": 1,
			"size": 1,
			"stride": 1,
			"strideResizeFunction": "bilinear"
		},
		"use": false
	}],
	"optimizerKey": "sgd",
	"referenceFunction": "absolute topk"
};
