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

% Conversion precio de dolares a euros teniendo en cuenta que
% estos datos son de 2023

data.property_price = data.property_price * 0.93;

% Eliminacion de caracteristicas no necesarias
data.property_type = [];
data.pxm2 = []; 
data.place_l3 = [];
data.tipo_precio = [];

% Identificación de valores atípicos
outliers = isoutlier(data.property_price, 'quartiles');
data(outliers, :) = [];

outliers = isoutlier(data.property_surface_total, 'quartiles');
data(outliers,:) = [];

outliers = isoutlier(data.property_surface_covered, 'quartiles');
data(outliers,:) = [];

% Transformaciones de las variables


% Definir la variable dependiente y las independientes
y = data.property_price;
data.property_price = []; % Eliminar la columna del precio de la tabla

x = data.Variables;

% Graficar histogramas para cada característica
figure(1);
num_features = size(x, 2);
for i = 1:num_features
    subplot(ceil(num_features/3), 3, i);
    histogram(x(:, i), 20);
    title(data.Properties.VariableNames{i});
end

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
numReps = 10;
Ers = zeros(6,1);

for order=1:12
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

figure(2);
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
    XTest = [XTest Xaux.^i];
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
    if min_rmse < best_rmse-50
        best_rmse = min_rmse;
        model_features = [model_features best_feature];
        selected_features(best_feature) = true;
        disp(['Adding feature ', num2str(best_feature), ' with ERRABS: ', num2str(best_rmse)]);
    else
        break;
    end
end

%% Prueba del modelo
% Modelo final con las características seleccionadas

A = [XTrain(:,features) ones(size(XTrain, 1), 1)];
sol = pinv(A) * YTrain;
ATest = [XTest(:,features) ones(size(XTest, 1), 1)];
YPred = ATest * sol;
errabs_test = mean(abs(YPred - YTest));
errabs_muestra = abs(YPred-YTest);
disp(['ErrAbs modelo y test: ', num2str(errabs_test)]);

[YTest, ind] = sort(YTest);
YPred = YPred(ind);

% Visualización de los resultados
figure(3);
plot(YPred, 'r.'); hold on;
plot(YTest, 'b.'); hold off;
legend('Predicciones', 'Datos reales');
xlabel('Número de muestra');
ylabel('Precio de la Propiedad');
title('Comparación de datos reales y predicciones');
hold off;

figure(4);
plot(errabs_muestra, '*y');
legend('Error absoluto');
xlabel('Numero de muestra');
ylabel('Error');
title('Error absoluto');

%% Prediccion sobre una casa

lat = input('Introduce la latitud, (deberia ir entre -34.689943 y -34.5359645): ');
lon = input('Introduce la longitud, (deberia ir entre -58.343238830600001 y -58.529930788599998): ');
hab = input('Introduce el numero de habitaciones de la propiedad: ');
dorm = input('Introduce el numero de dormitorios de la propiedad: ');
surftot = input('Introduce la superficie total de la propiedad: ');
surfcov = input('Introduce la superficie cubierta de la propiedad: ');
casa = input('Introduce 1 si es una casa 0 si no: ');
dep = input('Introduce 1 si es un departamento 0 si no: ');
ph = input('Introduce 1 si es una propiedad horizontal 0 si no: ');

Xcasa = [lat lon hab dorm surftot surfcov casa dep];
Xcasa= (Xcasa-mean(x)) ./ std(x);
Xaux = Xcasa;
for i=2:orden
    Xcasa = [Xcasa Xaux.^i];
end
A = [Xcasa(:,features) ones(size(Xcasa,1), 1)];
Ycasa = A*sol;

disp("Tu propiedad deberia tener un precio rondando los: "+Ycasa);
