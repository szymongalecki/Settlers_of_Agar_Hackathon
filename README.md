## Novo Nordisk Hackathon - Settlers of Agar


### Image preprocessing ideas
- Convert to grayscale and increase contrast to make bacteria colonies more visible
- Remove the timestamp visible on the petri dish so that the model doesn't learn the noise
- It may not be neural net can recognise timestamp as noise
- Augment data with added noise, horizontal/vertical flipping, blur to produce more data samples. Agar plates without colonies make up roughly 10% of the dataset which creates unbalanced dataset for binary classification.
### Model
[EfficientNet](https://paperswithcode.com/method/efficientnet)
[U-Net](https://paperswithcode.com/method/u-net)

### Recommendations
- Plot recall to observe threshold during training
- Class balancer in `pytorch` to get the same number of data samples for both classes

### What could we have done better
- Start of with a small dataset
- Get baseline model as fast as possible
- Do not underestimate the influence of data samples on training -> rescale pictures to smaller size and SAVE LOCALLY
- Look at the distribution of data straight away
- For binary classification explore the set threshold value by plotting recall 
- Research might be helpful
- YOLOv8 model does both classification and object detection
- Preprocessing:
	- Training using synthetic data
	- Background removal
- Too much focus on data preprocessing
- Be smarter next time