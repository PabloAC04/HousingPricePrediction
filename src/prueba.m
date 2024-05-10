clear; close all;

% Carga de datos
data = readtable('../data/properati_argentina_2021_tp1.csv');

%% Fase de preprocesado de datos

% Convertir property_type a columna por tipo
data.property_type = categorical(data.property_type);
dummies = dummyvar(data.property_type);
dummyTable = array2table(dummies, 'VariableNames', {'Casa', 'Departamento', 'PH'});
dummyTable.PH = [];  % Eliminación para evitar multicolinealidad
data = [data dummyTable];

% Eliminacion de caracteristicas no necesarias
data.property_type = [];
data.pxm2 = []; 
data.place_l3 = [];
data.tipo_precio = [];

% Definir la variable dependiente y las independientes
y = data.property_price;
data.property_price = []; % Eliminar la columna del precio de la tabla

x = data.Variables;


% Identificación de valores atípicos
outliers = isoutlier(y, 'quartiles');
x(outliers, :) = [];
y(outliers) = [];

% Normalizacion simple de los datos media 0 y varianza 1

x = (x-mean(x)) ./ std(x);

% División de datos en validación y prueba
splitPoint = floor(0.80 * size(x, 1));
XVal = x(1:splitPoint,:);
YVal = y(1:splitPoint);
XTest = x(splitPoint+1:end,:);
YTest = y(splitPoint+1:end);

%% Validacion
% Random sampling para ver qué modelo se ajusta mejor
numReps = 3;
Ers = zeros(6,1);

for order=1:10
    error = zeros(numReps,1);
    for k=1:numReps
        % Muestreo aleatorio de los datos de validación
        ids = randperm(size(XVal, 1));
        midPoint = floor(0.75 * length(ids));
        xtrn = XVal(ids(1:midPoint), :);
        ytrn = YVal(ids(1:midPoint));
        xtst = XVal(ids(midPoint+1:end), :);
        ytst = YVal(ids(midPoint+1:end));
        
        % Añadir términos polinomiales
        for j = 2:order
            xtrn = [xtrn XVal(ids(1:midPoint),:).^j];
            xtst = [xtst XVal(ids(midPoint+1:end),:).^j];
        end
        
        % Modelo de regresión
        A = [xtrn, ones(size(xtrn, 1), 1)];
        sol = pinv(A) * ytrn;
        ATest = [xtst, ones(size(xtst, 1), 1)];
        ypred = ATest * sol;
        error(k) = mean(abs(ytst - ypred));  % RMSE
    end
    Ers(order) = mean(error);
end

bar(Ers);
xlabel('Order of Polynomial');
ylabel('Average RMSE');
title('Model Complexity vs. Prediction Error');

pause;

[~,orden] = min(Ers);

disp("El polinomio que mejor se ajusta es el de orden "+orden);

%% Seleccion de caracteristica por pasos
XTrain = XVal;
YTrain = YVal;
Xaux = XTest;

selected_features = false(1, size(XTrain, 2));
model_features = [];
best_rmse = inf;

for i=2:orden
    XTrain = [XTrain XVal.^i];
    XTest = [XTest XTest.^i];
end

for i = 1:size(XTrain, 2)
    feature_rmse = inf(1, size(XTrain, 2));

    for j = find(~selected_features)
        features = [model_features j];

        % Añadir una columna de unos para el término intercepto
        A = [XTrain(:, features) ones(size(XTrain, 1), 1)];
        sol = pinv(A) * YTrain;
        YPredTrain = A * sol;
        feature_rmse(j) = mean(abs(YPredTrain - YTrain));
    end

    [min_rmse, best_feature] = min(feature_rmse);
    if min_rmse < best_rmse-200
        best_rmse = min_rmse;
        model_features = [model_features best_feature];
        selected_features(best_feature) = true;
        disp(['Adding feature ', num2str(best_feature), ' with RMSE: ', num2str(best_rmse)]);
    else
        break;
    end
end


% Modelo final con las características seleccionadas
A = [XTrain(:,features) ones(size(XTrain, 1), 1)];
sol = pinv(A) * YTrain;
YPredTrain = A * sol;
final_rmse = mean(abs(YPredTrain - YTrain));
disp(['Final RMSE with selected features: ', num2str(final_rmse)]);

% Predicciones sobre el conjunto de prueba
ATest = [XTest(:,features) ones(size(XTest, 1), 1)];
YPred = ATest * sol;
test_rmse = mean(abs(YPred - YTest));
disp(['Test RMSE with selected features: ', num2str(test_rmse)]);

% Visualización de los resultados
figure(1);
plot(YTest, 'b.'); hold on;
plot(YPred, 'r.'); 
legend('Datos reales', 'Predicciones');
xlabel('Número de muestra');
ylabel('Precio de la Propiedad');
title('Comparación de datos reales y predicciones');
hold off;

