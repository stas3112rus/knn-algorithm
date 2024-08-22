const _ = require('lodash');

const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');


let {
  features,
  labels,
  testFeatures,
  testLabels
} = loadCSV('kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long', 'sqft_lot', ],
  labelColumns: ['price']
});

features = tf.tensor(features);
labels = tf.tensor(labels);

function knn(features, labels, predictionPoint, k) {
  const {
    mean, // среднее значение
    variance // дисперсия
  } = tf.moments(features, 0);

  //Стандартизация для предсказуемой точки
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5))

  return (features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPrediction)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
    .slice(0, k)
    .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k);
}

const errors = [];

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0]
  errors.push(Math.abs(err))

  console.log('Err:', Math.abs(err));
});

const averageError = _.sum(errors) / errors.length;
console.log("averageError:", averageError)