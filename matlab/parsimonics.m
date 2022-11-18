clc;
clear;

dataset_path = '<PATH_TO_DATASET>'
shuffled_imds = shuffle(imageDatastore(dataset_path,...
    'IncludeSubfolders',true,'LabelSource','foldernames'));

training_percentage = .75;
[imdsTrain,imdsValidation] = splitEachLabel(shuffled_imds,training_percentage,'randomized');

layers = [
    imageInputLayer([50 50 1])

    convolution2dLayer(5,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    dropoutLayer(.33)
    fullyConnectedLayer(29)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',32,...
    'InitialLearnRate',0.02, ...
    'LearnRateDropPeriod',1,...
    'LearnRateDropFactor',.669,...
    'LearnRateSchedule','piecewise',...
    'MaxEpochs',6, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)