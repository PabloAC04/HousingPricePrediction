clc; clear; close all;

% Leemos los datos de la tabla.
data = readtable('../data/properati_argentina_2021_tp1.csv');

% Separamos el tipo de propiedad en tres columnas distintas.

data.property_type = categorical(data.property_type);
dummies = dummyvar(data.property_type);
dummyTable = array2table(dummies, 'VariableNames', {'Casa', 'Departamento', 'PH'});
data = [data dummyTable];

% Nos deshacemos de los datos que no nos interesan.
data.property_type = [];
data.pxm2 = [];
data.place_l3 = [];
data.tipo_precio = [];

% Separamos nuestros datos a predecir de los datos caracteristicos.
y = data.property_price;
data.property_price = [];

x = data.Variables;