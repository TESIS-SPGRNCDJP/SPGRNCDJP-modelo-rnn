import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd

file_path = "./PRY20241075.xlsx"
df = pd.read_excel(file_path)

gastos_entretenimiento = df['Entretenimiento'].tolist()
print(gastos_entretenimiento)

gastos = []
gasto_futuro = []

for i in range(0, len(gastos_entretenimiento) - 5, 6):
    # Obtener los gastos de los primeros 5 meses del bloque
    bloque = gastos_entretenimiento[i:i+5]
    gastos.append(bloque)

    # Obtener el gasto del sexto mes (si existe)
    if i + 5 < len(gastos_entretenimiento):
        resultado = [gastos_entretenimiento[i+5]]
        gasto_futuro.append(resultado)

gastos = np.array(gastos)
print("gastos:",gastos)

gasto_futuro = np.array(gasto_futuro)
print("gastos futiros: ",gasto_futuro)


# # Datos de los últimos 5 gastos (ejemplo)
# gastos = np.array([[100, 200, 300, 400, 500],  # Últimos 5 gastos de la persona 1
#                    [50, 75, 100, 150, 200],    # Últimos 5 gastos de la persona 2
#                    [300, 400, 500, 600, 700],  # Últimos 5 gastos de la persona 3
#                    [1000, 1500, 2000, 2500, 3000]])  # Últimos 5 gastos de la persona 4

# # Salida correspondiente (gasto 6 real para entrenamiento)
# gasto_futuro = np.array([600, 250, 800, 3500])

# Escalar los datos
scaler_gastos = MinMaxScaler()  # Escalador para los gastos de entrada
scaler_futuro = MinMaxScaler()  # Escalador para los gastos futuros (salida)

gastos_scaled = scaler_gastos.fit_transform(gastos)
gasto_futuro_scaled = scaler_futuro.fit_transform(gasto_futuro.reshape(-1, 1))

# Convertir los datos a tensores de PyTorch
gastos_tensor = torch.FloatTensor(gastos_scaled)
gasto_futuro_tensor = torch.FloatTensor(gasto_futuro_scaled)

# Definir el modelo de la red neuronal
class GastoPredictor(nn.Module):
    def __init__(self):
        super(GastoPredictor, self).__init__()
        self.fc1 = nn.Linear(5, 10)  # 5 entradas -> 10 neuronas ocultas
        self.fc2 = nn.Linear(10, 1)  # 10 neuronas ocultas -> 1 salida (el gasto futuro)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Crear una instancia del modelo
model = GastoPredictor()

# Definir el optimizador y la función de pérdida
criterion = nn.MSELoss()  # Error cuadrático medio
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento del modelo
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(gastos_tensor)
    loss = criterion(outputs, gasto_futuro_tensor)

    # Backward pass y optimización
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# Predicción para un nuevo conjunto de gastos
nuevos_gastos = np.array([[50, 75, 100, 150, 200]])  # Ejemplo de nuevos 5 gastos
nuevos_gastos_scaled = scaler_gastos.transform(nuevos_gastos)  # Escalar los datos de entrada
nuevos_gastos_tensor = torch.FloatTensor(nuevos_gastos_scaled)

model.eval()
with torch.no_grad():
    prediccion = model(nuevos_gastos_tensor)
    prediccion_gasto = scaler_futuro.inverse_transform(prediccion.numpy())  # Desescalar la predicción

print(f"Predicción del sexto gasto: {prediccion_gasto[0][0]}")

def guardar_modelo(modelo, ruta_archivo):
    torch.save(modelo.state_dict(), ruta_archivo)

# Guardar el modelo
guardar_modelo(model, 'modelo_rnn.pth')
joblib.dump(scaler_gastos, 'scaler_gastos.pkl')
joblib.dump(scaler_futuro, 'scaler_futuro.pkl')