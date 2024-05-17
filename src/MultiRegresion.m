clear; close all;

% Data loading
data = readtable('../data/properati_argentina_2021_tp1.csv');

%% Data Preprocessing Phase

% Convert property_type to columns by type
data.property_type = categorical(data.property_type);
typeDummies = dummyvar(data.property_type);
dummyTable = array2table(typeDummies, 'VariableNames', {'House', 'Apartment', 'PH'});
dummyTable.PH = [];  % Elimination to avoid multicollinearity
data = [data dummyTable];

% Conversion of price from dollars to euros
data.property_price = data.property_price * 0.93;

% Removal of unnecessary features
data.property_type = [];
data.pxm2 = []; 
data.place_l3 = [];
data.tipo_precio = [];

% Outlier detection
outliers = isoutlier(data.property_price, 'quartiles');
data(outliers, :) = [];
outliers = isoutlier(data.property_surface_total, 'quartiles');
data(outliers,:) = [];
outliers = isoutlier(data.property_surface_covered, 'quartiles');
data(outliers,:) = [];

% Define dependent and independent variables
y = log(data.property_price); % We use log make the distribution more simetric
data.property_price = []; 

x = data.Variables;

% Plot histograms for each feature

numFeatures = size(x, 2);
for i = 1:numFeatures
    subplot(ceil(numFeatures/3), 3, i);
    histogram(x(:, i), 20);
    title(data.Properties.VariableNames{i});
end

% Simple normalization of the data to mean 0 and variance 1
mu = mean(x);
sigma = std(x);
x = (x - mu) ./ sigma;

% Shuffle of the data
randomIndices = randperm(size(y, 1));
y = y(randomIndices);
x = x(randomIndices,:);

% Data split into validation and test sets
splitPoint = floor(0.80 * size(x, 1));
XValidation = x(1:splitPoint, :);
YValidation = y(1:splitPoint);
XTest = x(splitPoint+1:end, :);
YTest = y(splitPoint+1:end);

% Random sampling to see which polynomial fits better
% Instead of using OLS its been used Ridge regresion
% to prevent errors due to overfitting.
numRepetitions = 10;
errorSummaries = zeros(3, 1);
errorSummariessin = zeros(3, 1);
errorSummariessincos = zeros(3,1);
rp = 0.5; % Ridge parameter (0<rp<1)

for order = 1:3
    error = zeros(numRepetitions, 1);
    errorsin = zeros(numRepetitions, 1);
    errorsincos = zeros(numRepetitions, 1);
    for k = 1:numRepetitions
        % Random sampling of validation data
        ids = randperm(size(XValidation, 1));
        midPoint = floor(0.75 * length(ids));
        XTrain = XValidation(ids(1:midPoint), :);
        YTrain = YValidation(ids(1:midPoint));
        XTry = XValidation(ids(midPoint+1:end), :);
        YTry = YValidation(ids(midPoint+1:end));

        % Add polynomial terms
        XTrainPoly = XTrain;
        XTryPoly = XTry;
        for j = 2:order
            XTrainPoly = [XTrainPoly XTrain.^j];
            XTryPoly = [XTryPoly XTry.^j];
        end

        % x alone
        A = [XTrainPoly, ones(size(XTrainPoly, 1), 1)];
        sol = ridge(YTrain, A, rp);
        ATest = [XTryPoly, ones(size(XTryPoly, 1), 1)];
        predictions = ATest*sol;
        error(k) = mean(abs(exp(YTry) - exp(predictions)));  % MAE

        % x + sin(x)
        A = [XTrainPoly sin(XTrain), ones(size(XTrainPoly, 1), 1)];
        sol = ridge(YTrain, A, rp);
        ATest = [XTryPoly sin(XTry), ones(size(XTryPoly, 1), 1)];
        predictions = ATest*sol;
        errorsin(k) = mean(abs(exp(YTry) - exp(predictions)));  % MAE

        % x + sin(x) + cos(x)
        A = [XTrainPoly sin(XTrain) cos(XTrain), ones(size(XTrainPoly, 1), 1)];
        sol = ridge(YTrain, A, rp);
        ATest = [XTryPoly sin(XTry) cos(XTry), ones(size(XTryPoly, 1), 1)];
        predictions = ATest*sol;
        errorsincos(k) = mean(abs(exp(YTry) - exp(predictions)));  % MAE
    end
    errorSummaries(order) = mean(error);
    errorSummariessin(order) = mean(errorsin);
    errorSummariessincos(order) = mean(errorsincos);
end

figure(2);
errors = [errorSummaries; errorSummariessin; errorSummariessincos];
bar(errors);


labels = {'x', 'x²', 'x³', 'x+sin(x)', 'x²+sin(x)', 'x³+sin(x)','x+sin(x)+cos(x)', 'x²+sin(x)+cos(x)', 'x³+sin(x)+cos(x)'};

set(gca, 'XTickLabel', labels, 'XTick', 1:numel(labels), 'XTickLabelRotation', 45);

for i = 1:length(errors)
    text(i, errors(i), num2str(errors(i), '%.2f'),'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
end

ylim([0 50000]);
xlabel('Models');
ylabel('Average MAE');
title('Model Complexity vs. Prediction Error'), hold off;

[~, optimalOrder] = min(errors);

if optimalOrder<4 
    disp("The best fitting polynomial is of order "+ optimalOrder);
elseif optimalOrder > 6
    disp("The best fitting polynomial is of order "+ mod(optimalOrder,3) + " plus sine plus cosine");
else
    disp("The best fitting polynomial is of order "+ mod(optimalOrder,3) + " plus sine");
end

pause;

%% Features Selection whit Random Sampling
XTraining = XValidation;
YTraining = YValidation;
XAuxiliary = XTest;

XTrainingPoly = XTraining;
XTestPoly = XTest;
for i = 2:mod(optimalOrder, 3)
    XTrainingPoly = [XTrainingPoly XValidation.^i];
    XTestPoly = [XTestPoly XAuxiliary.^i];
end

% Generate all combinations of characteristics
features = 1:size(XTrainingPoly,2);
combinations = {};
for k = 1:length(features)
    combs = nchoosek(features, k);
    for j = 1:size(combs, 1)
        combinations{end+1} = combs(j, :); 
    end
end

for i=1:length(combinations)
    error = zeros(5, 1);
    for k = 1:5
        % Random sampling of validation data
        ids = randperm(size(XTrainingPoly, 1));
        midPoint = floor(0.75 * length(ids));
        XTrain = XTrainingPoly(ids(1:midPoint), combinations{i});
        YTrain = YTraining(ids(1:midPoint));
        XTry = XTrainingPoly(ids(midPoint+1:end), combinations{i});
        YTry = YTraining(ids(midPoint+1:end));
    
        A = [XTrain, ones(size(XTrain, 1), 1)];
        sol = ridge(YTrain, A, rp);
        ATest = [XTry, ones(size(XTry, 1), 1)];
        predictions = ATest*sol;
        error(k) = mean(abs(exp(YTry) - exp(predictions)));  % MAE
    end
    errorSummaries(i) = mean(error);
    disp(i);
end

[~, better] = min(errorSummaries);
modelFeatures = combinations{better};


disp("After the feature selection, this features left: ");
disp(modelFeatures);

%% Final Model Testing
% Final model with selected features

A = [XTrainingPoly(:, modelFeatures) ones(size(XTraining, 1), 1)];
coefsRidge = ridge(YTraining, A, rp);
ATest = [XTestPoly(:, modelFeatures) ones(size(XTest, 1), 1)];
YPredicted = ATest * coefsRidge;
testMAE = mean(abs(exp(YPredicted) - exp(YTest)));
disp(['Test MAE of model: ', num2str(testMAE)]);

[YTest, indices] = sort(YTest);
YPredicted = YPredicted(indices);

% Visualization of results
figure(3);
plot(exp(YPredicted), 'r.'); hold on;
plot(exp(YTest), 'b.'); hold off;
legend('Predictions', 'Real Data');
xlabel('Sample Number');
ylabel('Property Price');
title('Comparison of Real Data and Predictions');
hold off;



%% Prediction on a property

lat = input('Introduce the latitude of your property in Buenos Aires: ');
lon = input('Introduce the longitude of your property un Buenos Aires: ');
hab = input('Introduce the number of rooms in your property: ');
dorm = input('Introduce the number of bedrooms in your property: ');
surftot = input('Introduce the total surface of your property: ');
surfcov = input('Introduce the covered surface of your property: ');
casa = input('Introduce 1 if it is a house: ');
dep = input('Introduce 1 if it is an apartment: ');
ph = input('Introduce 1 if it is a condominium: ');

Xhouse = [lat lon hab dorm surftot surfcov casa dep];
Xhouse = (Xhouse - mu) ./ sigma;
Xaux = Xhouse;
for i = 2:order
    Xhouse = [Xhouse Xaux.^i];
end
A = [Xhouse(:, modelFeatures) ones(size(Xhouse, 1), 1)];
Yhouse = A * coefsRidge;
Yhouse = exp(Yhouse);

disp("Your property price is: " + Yhouse);
