# Anomaly Detection in Wireless Networks 

AS-CNN is an approach proposed for enhancing anomaly detection in wireless networks, particularly focusing on addressing the vulnerabilities present in traditional methods. This project integrates Adaptive Synthetic Sampling (ADASYN) and a novel Convolutional Neural Network (CNN) architecture, termed Split-Convolution CNN (SPC-CNN), to achieve improved accuracy, detection rates, and reduced false alarm rates compared to conventional IDS models.

## Key Features

- **ADASYN Integration:** Balances the sample distribution by generating synthetic samples for minority classes, thus mitigating the bias towards frequent classes commonly observed in imbalanced datasets.

- **SPC-CNN Architecture:** Utilizes Split-Convolution Modules to enhance feature diversity and eliminate interchannel redundancy during model training. This architecture enables the extraction of multi-scale features from oversampled data, improving the model's representation capability.

- **Performance Evaluation:** The AS-CNN model is evaluated using the widely-used NSL-KDD dataset, encompassing various attack types. Evaluation metrics include Accuracy (ACC), Detection Rate (DR), and False Alarm Rate (FAR), providing insights into the model's effectiveness in detecting anomalies.

## Datasets

 - KDDTrain
 - KDDTest+
 - KDDTest-21

## Results

- AS-CNN demonstrates superior performance compared to traditional CNN models, exhibiting higher accuracy, increased detection rates, and significantly reduced false alarm rates across different subsets of the NSL-KDD dataset.

- The model's robustness is particularly highlighted in its ability to detect minority attack classes such as Remote-to-Local (R2L) and User-to-Root (U2R) attacks, thereby enhancing cyber threat detection capabilities in wireless networks.



|                    DR                    |            ACC             |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image](https://github.com/rohzzn/nids/assets/47408756/2cc15b24-60ff-4fbd-8b7a-88023a4c2380) | ![image](https://github.com/rohzzn/nids/assets/47408756/6b999368-2056-43ae-9b28-979f37af2342) |

## Performance Comparison

| **Dataset**  | **Model**     | **ACC**   | **DR**    | **FAR**   |
| ------------ | ------------- | --------- | --------- | --------- |
| KDDTest+     | CNN           | 79.48     | 68.66     | 27.90     |
|              | **SPC - CNN** | **83.83** | **74.61** | **22.41** |
| KDDTest - 21 | CNN           | 60.71     | 58.47     | 71.88     |
|              | **SPC - CNN** | **69.41** | **66.44** | **60.17** |


## Attack Detection Performance

- **AS-CNN** shows superior detection rates for minority attack classes.
- Other models show significantly lower detection rates for R2L and U2R attacks.
- AS-CNN’s robustness makes it reliable for cyber threat detection.

1. DoS: Denial of Service
2. Probe: Network Probe
3. R2L: Remote-to-Local
4. U2R: User-to-Root

## Installation

To run the AS-CNN model and reproduce the results:

1. Clone the repository:

```bash
git clone https://github.com/rohzzn/nids.git
```

2. Navigate to the project directory:

```bash
cd nids
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Execute the main script to train and evaluate the AS-CNN model

```bash
python main.py
```

## Contributors

- 
- 


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
