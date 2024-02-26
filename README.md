## Novo Nordisk Hackathon - Settlers of Agar
![preprocessing|500](preprocessing.png)

### Tasks
1. [Classification](https://www.kaggle.com/competitions/agarvision-task-1-classification/)
2. [Counting](https://www.kaggle.com/competitions/agarvision-Task2-Counting)

### Image preprocessing ideas
- Convert image to grayscale and increase contrast to make bacteria colonies more visible.
- Remove the timestamp visible on the petri dish so that the model doesn't learn the noise. (It turned out that it is hard to automate that for different lighting setups)
- Removing timestamp might be unnecessary as neural network might just recognise it as a noise on its own.
- Augment data with added noise: horizontal/vertical flipping, blur, to produce more data samples.
- Agar plates without colonies make up roughly 10% of the dataset which creates unbalanced dataset for binary classification.
### Model
[EfficientNet](https://paperswithcode.com/method/efficientnet)  
[U-Net](https://paperswithcode.com/method/u-net)

### Recommendations
- Plot recall to observe threshold during training. Arbitrarily setting it to 0.5 might not be the best idea.
- Use class balancer in PyTorch to get the same number of data samples for both classes.

### Mistakes
- Not starting with a small subset of data.
- Not getting the baseline model as fast as possible.
- Underestimating the influence of the size of each data sample on training.
- Not rescaling pictures to smaller size early on.
- Not saving the rescaled pictures locally.
- Not looking at the distribution of data before the training.
- Not setting the threshold for binary classification based on the recall plot.
- Finding the research paper that doesn't focus on the architecture.
- Not using [Papers with Code](https://paperswithcode.com/sota).
- Focusing too mouch on the data preprocessing.

