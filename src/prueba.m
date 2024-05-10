clc; clear; close all;

% Carga de datos
data = readtable('../data/properati_argentina_2021_tp1.csv');

% Convertir property_type a variables dummy y concatenar
data.property_type = categorical(data.property_type);
dummies = dummyvar(data.property_type);
dummyTable = array2table(dummies, 'VariableNames', {'Casa', 'Departamento', 'PH'});
dummyTable.Casa = [];  % Eliminación para evitar multicolinealidad

data = [data dummyTable];

% Eliminar columnas innecesarias
data.property_type = [];
data.pxm2 = []; % Asumiendo que quieres eliminar esta columna
data.place_l3 = [];
data.tipo_precio = [];

% Transformaciones potenciales
% Escalar latitud y longitud
data.latitud = data.latitud * 1000;
data.longitud = data.longitud * 1000;

% Interacción entre algunas variables
data.Interaccion1 = data.property_rooms .* data.property_bedrooms;
data.Interaccion2 = data.property_surface_total .* data.property_surface_covered;
%data.Departamento = [];
%data.PH = [];

% Definir la variable dependiente y las independientes
y = data.property_price;
data.property_price = []; % Eliminar la columna del precio de la tabla

x = data.Variables;

idout1 = find(y>mean(y)*3);
idout2 = find(y<mean(y)/3);

y([idout1; idout2]) = [];
x([idout1; idout2],:) = [];

% Asegurar que todas las otras columnas son numéricas
T = size(x, 1);

% Barajamos los datos
id = randperm(T);
x = x(id, :);
y = y(id);

XTrain = x(1:floor(T*0.75)-1,:);
YTrain = y(1:floor(T*0.75)-1,:);

XTest = x(floor(T*0.75):end,:);
YTest = y(floor(T*0.75):end,:);

% Entrenar un modelo de regresión lineal
mdl = fitlm(XTrain, YTrain);

% Evaluar el modelo
YPred = predict(mdl, XTest);
rmse = sqrt(mean((YPred - YTest).^2));

disp(['RMSE del modelo: ', num2str(rmse)]);

% Inspeccionar los coeficientes
disp(mdl.Coefficients);

figure(1);
plot(1:sizee, YTest, 'r*'); hold on;
plot(1:19661, YPred, 'b*'); hold off;
