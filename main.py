import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pennylane as qml
from sklearn.metrics import accuracy_score


data=load_breast_cancer()
x,y=data.data,data.target
print(data.feature_names,data.target_names)

scaler = StandardScaler()
x_scaled=scaler.fit_transform(x)

n_features=30
n_qbits=5
dim_tar=2**n_qbits
x_pad=np.pad(x_scaled,((0,0),(0,dim_tar-n_features)),'constant')
x_train,x_test,y_train,y_test = train_test_split(x_pad,y,test_size=0.2)

dev=qml.device('lightning.qubit',wires=n_qbits)
zero_state_projector = qml.Projector([0] * n_qbits, wires=range(n_qbits))

@qml.qnode(dev)
def quantum_kernel(X1, x2):
    qml.AmplitudeEmbedding(X1,wires=range(n_qbits),normalize=True)

    qml.adjoint(qml.AmplitudeEmbedding(x2,wires=range(n_qbits),normalize=True))

    return  qml.expval(zero_state_projector)

def k_closet_neighobur(X_train, y_train, x_test_point, k):
    distances = []
    for x_train_point in X_train:
        kernel_value = quantum_kernel(x_train_point, x_test_point)
        distance = 1 - kernel_value
        distances.append(distance)


    sorted_distances=np.argsort(distances)
    k_nearest_labels=y_train[sorted_distances]
    count=np.bincount(k_nearest_labels)
    pred=np.argmax(count)
    return pred

def perdict(X, X_train, y_train, k):
    return k_closet_neighobur(X_train, y_train, X, k)


k=3
predictions = [k_closet_neighobur(x_train,y_train, x_test_point, k) for x_test_point in x_test]
accuracy = accuracy_score(y_test, predictions)
pred_lebel=perdict(x_test[0],x_train, y_train, k)
true_label=y_test[0]
print(f"Predicted label: {y[pred_lebel]} Vs true label: {y[true_label]}")

print(f"\nAccuracy for Qk-NN with Amplitude Encoding (5 qubits): {accuracy}")
