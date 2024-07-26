----
# Deep Learning Questions and Answers
----
<br><br>


## 01. Introduction
----

**1. Do you need these for deep learning?**
   * Lots of math: False
   * Lots of data: False
   * Lots of expensive computers: False
   * A PhD: False

**2. Name five areas where deep learning is now the best in the world.**
   - NLP, CV, Medicine, Playing games, Recommendation systems
>Any five of the following:
>* **Natural Language Processing (NLP):** Question Answering, Document Summarization and Classification, etc.
>* **Computer Vision (CV):** Satellite and drone imagery interpretation, face detection and recognition, image captioning, etc.
>* **Medicine:** Finding anomalies in medical images (ex: CT, X-ray, MRI), detecting features in tissue slides (pathology), diagnosing diabetic retinopathy, etc.
>* **Biology:** Folding proteins, classifying, genomics tasks, cell classification, etc.
>* **Image generation/enhancement:** colorizing images, improving image resolution (super-resolution), removing noise from images (denoising), converting images to art in style of famous artists (style transfer), etc.
>* **Recommendation systems:** web search, product recommendations, etc.
>* **Playing games:** Super-human performance in Chess, Go, Atari games, etc
>* **Robotics:** handling objects that are challenging to locate (e.g. transparent, shiny, lack of texture) or hard to pick up
>* **Other applications:** financial and logistical forecasting; text to speech; much much more.



**3. What was the name of the first device that was based on the principle of the artificial neuron?**
>The Mark I perceptron built by Frank Rosenblatt

**4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?**
- A set of processing units
- A state of activation
- An output function for each unit
- A pattern of connectivity among units
- A propagation rule for propagating patterns of activities through the network of connectivities
- An activation rule for combining the inputs impinging on a unit with the current state of that unit to produce an output for the unit
- A learning rule whereby patterns of connectivity are modified by experience
- An environment within which the system must operate


**5. What were the two theoretical misunderstandings that held back the field of neural networks?**
   - The belief that single-layer neural networks couldn't solve complex problems (XOR problem)
   - The misconception that adding an extra layer of neurons makes networks too big and too slow to be useful

>In 1969, Marvin Minsky and Seymour Papert demonstrated in their book, “Perceptrons”, that a single layer of artificial neurons cannot learn simple, critical mathematical functions like XOR logic gate. While they subsequently demonstrated in the same book that additional layers can solve this problem, only the first insight was recognized, leading to the start of the first AI winter.
>
>In the 1980’s, models with two layers were being explored. Theoretically, it is possible to approximate any mathematical function using two layers of artificial neurons. However, in practices, these networks were too big and too slow. While it was demonstrated that adding additional layers improved performance, this insight was not acknowledged, and the second AI winter began. In this past decade, with increased data availability, and improvements in computer hardware (both in CPU performance but more importantly in GPU performance), neural networks are finally living up to its potential.

**6. What is a GPU?**
- A GPU (Graphics Processing Unit) is a specialized processor designed to accelerate graphics rendering. In deep learning, GPUs are used for their ability to perform many simple computations in parallel.
>GPU stands for Graphics Processing Unit (also known as a graphics card). Standard computers have various components like CPUs, RAM, etc. CPUs, or central processing units, are the core units of all standard computers, and they execute the instructions that make up computer programs. GPUs, on the other hand, are specialized units meant for displaying graphics, especially the 3D graphics in modern computer games. The hardware optimizations used in GPUs allow it to handle thousands of tasks at the same time. Incidentally, these optimizations allow us to run and train neural networks hundreds of times faster than a regular CPU.

**7. Open a notebook and execute a cell containing: `1+1`. What happens?**
>Executing `1+1` in a Jupyter notebook cell will output `2`.

**8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.**<br>
Approach to predicting cell outputs:
   - Read the code in each cell carefully
   - Consider what each line does
   - Think about any variables or data structures being manipulated
   - Predict the output based on your understanding
   - Execute the cell to check your prediction

**9. Complete the Jupyter Notebook online appendix.**
>No specific answer provided - this is a task for the reader.

**10. Why is it hard to use a traditional computer program to recognize images in a photo?**<br>
  - Images are complex, with many pixels and color variations
  - Objects in images can vary in size, orientation, lighting, etc.
  - Traditional programs struggle with abstracting features and patterns
>For us humans, it is easy to identify images in a photos, such as identifying cats vs dogs in a photo. This is because, subconsciously our brains have learned which features define a cat or a dog for example. But it is hard to define set rules for a traditional computer program to recognize a cat or a dog. Can you think of a universal rule to determine if a photo contains a cat or dog? How would you encode that as a computer program? This is very difficult because cats, dogs, or other objects, have a wide variety of shapes, textures, colors, and other features, and it is close to impossible to manually encode this in a traditional computer program.

**11. What did Samuel mean by "weight assignment"?**
>“weight assignment” refers to the current values of the model parameters. Arthur Samuel further mentions an “ automatic means of testing the effectiveness of any current weight assignment ” and a “ mechanism for altering the weight assignment so as to maximize the performance ”. This refers to the evaluation and training of the model in order to obtain a set of parameter values that maximizes model performance.

**12. What term do we normally use in deep learning for what Samuel called "weights"?**
- In deep learning, we typically use the term _"parameters"_ for what Samuel called "weights."

>We instead use the term parameters. In deep learning, the term “weights” has a separate meaning. (The neural network has various parameters that we fit our data to. As shown in upcoming chapters, the two types of neural network parameters are weights and biases)

**13. Draw a picture that summarizes Samuel's view of a machine learning model.**

```
                    ______________
[inputs]  ------>   |            |
                    |   Model    |    ------>  results ----->  performance
[weights] ------>   |____________|                                  |
     ^                                                              |
     |                                                              |
     |                                                              |
     |______________________________________________________________|
                       Update

```

**14. Why is it hard to understand why a deep learning model makes a particular prediction?**
- Models have many layers and millions of parameters and as such have a **"deep nature."**
- The learned features are often abstract and not easily interpretable
- The decision process is distributed across the network

>This is a highly-researched topic known as interpretability of deep learning models. Deep learning models are hard to understand in part due to their **“deep” nature.** Think of a linear regression model. Simply, we have some input variables/data that are multiplied by some weights, giving us an output. We can understand which variables are more important and which are less important based on their weights. A similar logic might apply for a small neural network with 1-3 layers. However, deep neural networks have hundreds, if not thousands, of layers. It is hard to determine which factors are important in determining the final output. The neurons in the network interact with each other, with the outputs of some neurons feeding into other neurons. Altogether, due to the complex nature of deep learning models, it is very difficult to understand why a neural network makes a given prediction.
>
>However, in some cases, recent research has made it easier to better understand a neural network’s prediction. For example, as shown in this chapter, we can analyze the sets of weights and determine what kind of features activate the neurons. When applying CNNs to images, we can also see which parts of the images highly activate the model. We will see how we can make our models interpretable later in the book.

**15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?**
- The Universal Approximation Theorem shows that a neural network can solve any mathematical problem to any level of accuracy.
  
>The universal approximation theorem states that neural networks can theoretically represent any mathematical function. However, it is important to realize that practically, due to the limits of available data and computer hardware, it is impossible to practically train a model to do so. But we can get very close!

**16. What do you need in order to train a model?**
- A dataset (input data and corresponding labels/outputs)
- A model architecture
- A loss function
- An optimization algorithm
  
>You will need an architecture for the given problem. You will need data to input to your model. For most use-cases of deep learning, you will need labels for your data to compare your model predictions to. You will need a loss function that will quantitatively measure the performance of your model. And you need a way to update the parameters of the model in order to improve its performance (this is known as an optimizer).

**17. How could a feedback loop impact the rollout of a predictive policing model?**
- A feedback loop in predictive policing could lead to biased predictions reinforcing themselves, potentially amplifying existing biases.
  
>In a predictive policing model, we might end up with a positive feedback loop, leading to a highly biased model with little predictive power. For example, we may want a model that would predict crimes, but we use information on arrests as a proxy . However, this data itself is slightly biased due to the biases in existing policing processes. Training with this data leads to a biased model. Law enforcement might use the model to determine where to focus police activity, increasing arrests in those areas. These additional arrests would be used in training future iterations of models, leading to an even more biased model. This cycle continues as a positive feedback loop

**18. Do we always have to use 224×224-pixel images with the cat recognition model?**
- No, we don't always have to use 224x224-pixel images. Models can be adapted to different image sizes.
>No we do not. 224x224 is commonly used for historical reasons. You can increase the size and get better performance, but at the price of speed and memory consumption.

**19. What is the difference between classification and regression?**
- Classification predicts discrete categories or labels, while regression predicts continuous values.
>Classification is focused on predicting a class or category (ex: type of pet). Regression is focused on predicting a numeric quantity (ex: age of pet).

**20. What is a validation set? What is a test set? Why do we need them?**
- Validation set: Used to tune hyperparameters and evaluate during training.
- Test set: Used to evaluate final model performance.
- They're needed to assess how well the model generalizes to unseen data.

>The validation set is the portion of the dataset that is not used for training the model, but for evaluating the model during training, in order to prevent _overfitting_. This ensures that the model performance is not due to “cheating” or memorization of the dataset, but rather because it learns the appropriate features to use for prediction. However, it is possible that we overfit the validation data as well. This is because the human modeler is also part of the training process, adjusting _hyperparameters_ (see **question 32** for definition) and training procedures according to the validation performance. Therefore, another unseen portion of the dataset, the test set, is used for final evaluation of the model. This splitting of the dataset is necessary to ensure that the model _generalizes_ to _unseen_ data.

**21. What will fastai do if you don't provide a validation set?**
- If no validation set is provided, fastai typically creates one by randomly splitting off a portion (20%) of the training data.
>fastai will automatically create a validation dataset. It will randomly take 20% of the data and assign it as the validation set ( valid_pct = 0.2 ).

**22. Can we always use a random sample for a validation set? Why or why not?**
- We can't always use a random sample for a validation set, especially with time series data or when there's a specific structure to the data.

>A good validation or test set should be representative of new data you will see in the future. Sometimes this isn’t true if a random sample is used. For example, for time series data, selecting sets randomly does not make sense. Instead, defining different time periods for the train, validation, and test set is a better approach.

**23. What is overfitting? Provide an example.**
- Overfitting: When a model performs well on training data but poorly on new, unseen data.
- <u>Example:</u> memorizing noise in training data rather than learning general patterns.

>Overfitting is the most challenging issue when it comes to training machine learning models. Overfitting refers to when the model fits too closely to a limited set of data but _does not generalize well to unseen data._ This is especially important when it comes to neural networks, because neural networks can potentially “memorize” the dataset that the model was trained on, and will perform abysmally on unseen data because it didn’t “memorize” the ground truth values for that data. This is why a proper validation framework is needed by splitting the data into training, validation, and test sets.

**24. What is a metric? How does it differ from "loss"?**
- A metric is a measure used to evaluate model performance. Unlike loss, metrics are often more interpretable and directly related to the task.

>A _metric_ is a function that measures quality of the model’s predictions using the validation set. This is similar to the _­loss_, which is also a measure of performance of the model. However, loss is meant for the optimization algorithm (like SGD) to efficiently update the model parameters, while metrics are human-interpretable measures of performance. Sometimes, a metric may also be a good choice for the loss.

**25. How can pretrained models help?**
- Pretrained models provide _a good starting point,_ having already learned useful features from a large dataset.

>Pretrained models have been trained on other problems that may be quite similar to the current task. For example, pretrained image recognition models were often trained on the ImageNet dataset, which has 1000 classes focused on a lot of different types of visual objects. Pretrained models are useful because they have already learned how to handle a lot of simple features like edge and color detection. However, since the model was trained for a different task than already used, this model cannot be used as is.

**26. What is the "head" of a model?**
- The "head" of a model typically refers to the final layers that are task-specific, often added on top of a pretrained base.

>When using a pretrained model, the later layers of the model, which were useful for the task that the model was originally trained on, are replaced with one or more new layers with randomized weights, of an appropriate size for the dataset you are working with. These new layers are called the _“head”_ of the model.

**27. What kinds of features do the early layers of a CNN find? How about the later layers?**
- Early CNN layers find simple features like edges and textures. Later layers find more complex, abstract features specific to the task.
>Earlier layers learn simple features like diagonal, horizontal, and vertical edges. Later layers learn more advanced features like car wheels, flower petals, and even outlines of animals.

**28. Are image models only useful for photos?**
- No, image models can be useful for various types of 2D data, not just photos (e.g., medical imaging, satellite imagery, audio spectrograms).
>Nope! Image models can be useful for other types of images like sketches, medical data, etc.

However, a lot of information can be represented as _images_ . For example, a sound can be converted into a spectrogram, which is a visual interpretation of the audio. Time series (ex: financial data) can be converted to image by plotting on a graph. Even better, there are various transformations that generate images from time series, and have achieved good results for time series classification. There are many other examples, and by being creative, it may be possible to formulate your problem as an image classification problem, and use pretrained image models to obtain state-of-the-art results!

**29. What is an "architecture"?**
- An "architecture" refers to the specific structure and arrangement of layers in a neural network.
>The architecture is the template or structure of the model we are trying to fit. It defines the mathematical model we are trying to fit.

**30. What is segmentation?**
- Segmentation is the task of dividing an image into multiple segments or objects, often assigning a class label to each pixel.
>At its core, segmentation is a _pixelwise classification problem._ We attempt to predict a label for every single pixel in the image. This provides a mask for which parts of the image correspond to the given label.

**31. What is `y_range` used for? When do we need it?**
- `y_range` specifies the range of the target variable in regression problems, constraining the output of the model to a specific range.
>`y_range` is being used to limit the values predicted when our problem is focused on predicting a numeric value in a given range (ex: predicting movie ratings, range of 0.5-5).

**32. What are "hyperparameters"?**
- Hyperparameters are configuration settings for the learning process that are not learned from the data (e.g., learning rate, batch size, number of epochs).
>Training models requires various other parameters that define _how_ the model is trained. For example, we need to define how long we train for, or what learning rate (how fast the model parameters are allowed to change) is used. These sorts of parameters are hyperparameters.

**33. What's the best way to avoid failures when using AI in an organization?**<br>
Best practices to avoid AI failures in organizations:
- Thorough testing and validation
- Test out a simple baseline model
- Continuous monitoring and updating of models
- Understanding and mitigating potential biases
- Having a diverse team with both technical and domain expertise
- Establishing clear ethical guidelines and governance structures
- Maintaining transparency about the capabilities and limitations of AI systems
>Key things to consider when using AI in an organization:
>
>1. Make sure a training, validation, and testing set is defined properly in order to evaluate the model in an appropriate manner.
>2. Try out a simple baseline, which future models should hopefully beat. Or even this simple baseline may be enough in some cases.

### Further Research Questions

1. **Why is a GPU useful for deep learning? How is a CPU different, and why is it less effective for deep learning?**
   
   **GPU (Graphics Processing Unit):**
   - **Parallel Processing:** GPUs have thousands of small, efficient cores designed to handle multiple tasks simultaneously (parallel processing). This parallelism is well-suited and highly efficient for the matrix & vector operations common in deep learning.
   - **High Throughput:** GPUs can process large amounts of data simultaneously, which speeds up the training of deep learning models.
   - **High memory bandwidth:** GPUs can transfer large amounts of data quickly, which is crucial for handling the large datasets used in deep learning.
   - **Specialized Hardware:** GPUs are specifically designed to handle the kinds of calculations used in graphics rendering and deep learning, such as linear algebra operations.
   - **Specialized for floating-point operations:** Deep learning involves numerous floating-point calculations, which GPUs are optimized to perform.

   **CPU (Central Processing Unit):**
   - **General Purpose:** CPUs are designed for general-purpose computing and have fewer cores optimized for sequential processing.
   - **Lower Parallelism:** While CPUs can handle multiple tasks, they are not as effective at parallelizing the specific tasks required for deep learning.
   - **Lower memory bandwidth:** CPUs generally have lower memory bandwidth compared to GPUs.
   - **Latency Optimized:** CPUs are optimized for tasks requiring low latency, which is different from the high-throughput, parallel nature of deep learning tasks.
   - **Versatility:** CPUs are designed for a wide range of tasks, making them less specialized for the specific needs of deep learning.

   **Effectiveness:**
   - **Speed:** GPUs can significantly reduce the time required to train deep learning models compared to CPUs.
   - **Efficiency:** Deep learning often involves repeating similar calculations over large datasets, which is more suited to GPU architecture.
   - **Power Consumption:** For the same amount of deep learning computation, CPUs generally consume more power than GPUs.
   - **Scalability:** For large-scale deep learning tasks, GPUs provide better scalability and performance, making them more effective for deep learning compared to CPUs.
  
>However, it's worth noting that CPUs still play a crucial role in the overall computing system and are often used in conjunction with GPUs for deep learning tasks.

2. **Try to think of three areas where feedback loops might impact the use of machine learning. See if you can find documented examples of that happening in practice.**

   **1. Predictive Policing:**
   - **Feedback Loop:** Predictive policing models might suggest increased police presence in certain areas based on historical crime data. More police presence can lead to more recorded crimes, reinforcing the model's bias. 
   - **Documented Example:** In Oakland, California, predictive policing software led to biased policing in minority neighborhoods, as reported by various civil rights groups.

   - **Feedback Loop 2:** Increased policing in predicted high-crime areas leads to more arrests, which then reinforces the model's prediction of high crime in those areas.
   - **Documented Example 2:** A 2016 study by the Human Rights Data Analysis Group found that PredPol, a predictive policing software, repeatedly sent officers to the same neighborhoods regardless of the true crime rate, creating a feedback loop of over-policing.

   **2. Social Media Algorithms:**
   - **Feedback Loop:** Social media algorithms that promote content based on user engagement can create echo chambers. Users are shown content that reinforces their existing beliefs, leading to increased polarization.
   - **Documented Example:** A 2015 study published in Science showed that Facebook's news feed algorithm contributed to political polarization by exposing users primarily to views that aligned with their own. Facebook's algorithm has been criticized for promoting divisive content that increases user engagement but also leads to echo chambers and misinformation.
      
   **3. Loan Approval Systems:**
   - **Feedback Loop:** Automated loan approval systems might use historical data that includes biases. If certain demographics are historically less likely to receive loans, the system may perpetuate this bias by continuing to deny loans to these groups.
   - **Documented Example:** There have been cases where algorithmic biases in credit scoring systems have led to discriminatory practices against minorities, as reported by various financial oversight bodies.
  
   **4. Credit Scoring:**
   - **Feedback Loop:** Individuals with low credit scores receive less favorable loan terms, making it harder to improve their financial situation and credit score, thus perpetuating the cycle.
   - **Documented Example:** A 2017 study by the National Consumer Law Center found that error-ridden credit reports were disproportionately hurting low-income consumers, making it difficult for them to access credit and improve their scores.

>These examples highlight the importance of carefully designing and monitoring machine learning systems to prevent harmful feedback loops, especially in high-stakes applications that can significantly impact people's lives.


## 02. Production
----
