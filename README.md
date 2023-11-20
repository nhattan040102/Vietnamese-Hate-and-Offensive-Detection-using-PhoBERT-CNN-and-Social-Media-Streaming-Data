# Vietnamese-Hate-and-Offensive-Detection-using-PhoBERT-CNN-and-Social-Media-Streaming-Data
This is a repository re-implementing the code of the paper ```Vietnamese-Hate-and-Offensive-Detection-using-PhoBERT-CNN-and-Social-Media-Streaming-Data```  for CS431 final project

Link paper: https://arxiv.org/pdf/2206.00524.pdf

Author's gmail: 18520908@gm.uit.edu.vn

Author's github: https://github.com/kh4nh12

# Reference
- ```PhoBERT```: Pre-trained language models for Vietnamese - https://github.com/VinAIResearch/PhoBERT
- ```Convolutional Neural Networks``` for Sentence Classification - https://github.com/yoonkim/CNN_sentence
- ```Apache spark```: a unified engine for big data processing - https://spark.apache.org/docs/3.1.1
- ```Apache kafka```: a distributed event-store and streaming platform: - https://kafka.apache.org/

# Usage
- Install necessary packages from requirements.txt file
```bash
    pip install -r HateSpeechDetectionApp/requirements.txt
```

- Set up kafka cluster locally: please refer to this document https://kafka.apache.org/quickstart#quickstart_startserver

- Create topic for data storage (run this code in the kafka path you have just set up): 
```bash
    bin/kafka-topics.sh --create --topic rawData  --bootstrap-server localhost:9092

    bin/kafka-topics.sh --create --topic anomalies  --bootstrap-server localhost:9092
```

- Run ```producer.py``` in ```HateSpeechDetectionApp``` folder to begin producing data to kafka ```rawData``` topic
```bash
    python HateSpeechDetectionApp/producer.py
```

- Run ```spark-flask.py``` in ```HateSpeechDetectionApp``` to set up the App, the data from the web will be retrieved from ```cleanData``` topic
```bash
    python HateSpeechDetectionApp/spark-flask.py
```

- Finally run ```streaming.py``` in ```HateSpeechDetectionApp``` so that Spark can process the streaming data in realtime and load it to the ```cleanData``` topic
```bash
    spark-submit streaming.py
```



# Evaluation on test dataset
| Metric | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 | 0.9284 | 0.9261 | 0.9273 | 5562 |
| 1 | 0.4189 | 0.3932 | 0.4057 | 473 |
| 2 | 0.5218 | 0.5566 | 0.5386 | 645 |
| Accuracy | | | 0.8527 | 6680 |
| Macro Avg | 0.6231 | 0.6253 | 0.6239 | 6680 |
| Weighted Avg | 0.8531 | 0.8527 | 0.8528 | 6680 |
