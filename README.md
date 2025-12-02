Este projeto implementa um sistema básico de reconhecimento facial usando **OpenCV** e o algoritmo **Local Binary Pattern Histograms (LBPH)**. Ele permite a captura de amostras faciais, o treinamento do modelo e o reconhecimento em tempo real de indivíduos, exibindo o nome, a confiança da previsão, e a idade (se a data de nascimento for fornecida).

#### Bibliotecas

* `opencv-python`
* `opencv-contrib-python` (necessário para o módulo `cv2.face`, onde está o LBPH)
* `numpy`

```bash
pip install opencv-python opencv-contrib-python numpy
```
