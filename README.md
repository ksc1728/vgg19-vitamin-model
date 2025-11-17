<h1>Vitamin Detection from Food Images using Deep Learning</h1>

<p>This project is a deep learning web application that predicts the dominant vitamin content (A, B, C, D, or E) in food images using a MobileNetV2-based CNN model. The model is trained on a curated dataset and deployed with a Flask web interface for easy image upload and prediction.</p>

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/65efd647-3d8a-4ab6-a01d-6b837681d231" />

<img width="1200" height="800" alt="image" src="https://github.com/user-attachments/assets/c6492052-3ecf-4833-8e7d-26ca03765cb4" />


<h2>Features</h2>

<ul>
  <li>Predicts whether a food image is rich in Vitamin A, B, C, D, or E</li>
  <li>Uses transfer learning with MobileNetV2 pretrained on ImageNet</li>
  <li>Simple and responsive web interface built with Flask and HTML</li>
  <li>Achieved 77.76% test accuracy</li>
  <li>Includes vitamin-specific health benefits in the results</li>
</ul>


<h2>Model Architecture</h2>

<dl>
  <dt>Base Model</dt>
  <dd>-MobileNetV2</dd>

  <dt>Custom Layers</dt>
  <dd>-Global Average Pooling</dd>
  <dd>-Dense layers with Dropout</dd>
  <dd>-Final Dense layer with softmax activation</dd>

  <dt>Optimizer</dt>
  <dd>-Adam</dd>

  <dt>Callbacks</dt>
  <dd>-EarlyStopping, ModelCheckpoint</dd>
</dl>

<h2>Dataset</h2>

<dl>
  <dt>Source</dt>
  <dd>USDA Barley Dataset on Kaggle - https://www.kaggle.com/api/v1/datasets/download/adityaaggarwal09/usda-barley?dataset_version_number=1</dd>

  <dt>Preprocessing</dt>
  <dd>Images resized to 224x224, normalized, and augmented using ImageDataGenerator</dd>
</dl>

<h2>Web Interface</h2>

<p>The frontend is built using HTML. After uploading an image, the app displays:</p>

<p>- The uploaded image<br>
- The predicted vitamin category<br>
- A short description of the vitamin's health benefits</p>

<h2>Installation</h2>

<p>Clone the repository: 
   git clone https://github.com/ksc1728/vgg19-vitamin-model.git</p>
<p>Install dependencies: 
   pip install -r requirements.txt</p>
<p>Run the app: 
   python app.py</p>
   
