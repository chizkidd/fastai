----
# Deep Learning Questions and Answers
<br><br>
----

## 01. Introduction

**1. Do you need these for deep learning?**
   * Lots of math: False
   * Lots of data: False
   * Lots of expensive computers: False
   * A PhD: False

**2. Name five areas where deep learning is now the best in the world.**
   - Image & Speech recognition
   - Computer Vision (CV) & Robotics
   - Natural language processing (NLP)
   - Game playing (e.g., Go, chess)
   - Medical diagnosis (in certain specific tasks)

**3. What was the name of the first device that was based on the principle of the artificial neuron?**
- The first device based on the principle of the artificial neuron was the Perceptron.

**4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?**
   - A set of processing units
   - A state of activation
   - An output function for each unit
   - A pattern of connectivity among units
   - A propagation rule for spreading patterns of activities
   - An activation rule for combining inputs
   - A learning rule for modifying patterns of connectivity
   - An environment within which the system operates

**5. What were the two theoretical misunderstandings that held back the field of neural networks?**
   - The belief that single-layer neural networks couldn't solve complex problems (XOR problem)
   - The misconception that backpropagation couldn't work effectively for deep networks

**6. What is a GPU?**
- A GPU (Graphics Processing Unit) is a specialized processor designed to accelerate graphics rendering. In deep learning, GPUs are used for their ability to perform many simple computations in parallel.

**7. Open a notebook and execute a cell containing: `1+1`. What happens?**
- Executing `1+1` in a Jupyter notebook cell will output `2`.

**8. Follow through each cell of the stripped version of the notebook for this chapter. Before executing each cell, guess what will happen.**<br>
Approach to predicting cell outputs:
   - Read the code in each cell carefully
   - Consider what each line does
   - Think about any variables or data structures being manipulated
   - Predict the output based on your understanding
   - Execute the cell to check your prediction

**9. Complete the Jupyter Notebook online appendix.**
- (No specific answer provided - this is a task for the reader)

**10. Why is it hard to use a traditional computer program to recognize images in a photo?**<br>
  - Images are complex, with many pixels and color variations
  - Objects in images can vary in size, orientation, lighting, etc.
  - Traditional programs struggle with abstracting features and patterns

**11. What did Samuel mean by "weight assignment"?**
- "Weight assignment" likely refers to adjusting the importance of different features or inputs in a machine learning model.

**12. What term do we normally use in deep learning for what Samuel called "weights"?**
- In deep learning, we typically use the term "parameters" for what Samuel called "weights."

**13. Draw a picture that summarizes Samuel's view of a machine learning model.**

```
[Input] -> [Model with adjustable weights] -> [Output]
   ^                                              |
   |______________________________________________|
             (Learning feedback loop)
```

**14. Why is it hard to understand why a deep learning model makes a particular prediction?**
- Models have many layers and millions of parameters
- The learned features are often abstract and not easily interpretable
- The decision process is distributed across the network

**15. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?**
- The Universal Approximation Theorem shows that a neural network can solve any mathematical problem to any level of accuracy.

**16. What do you need in order to train a model?**
- A dataset (input data and corresponding labels/outputs)
- A model architecture
- A loss function
- An optimization algorithm

**17. How could a feedback loop impact the rollout of a predictive policing model?**
- A feedback loop in predictive policing could lead to biased predictions reinforcing themselves, potentially amplifying existing biases.

**18. Do we always have to use 224×224-pixel images with the cat recognition model?**
- No, we don't always have to use 224x224-pixel images. Models can be adapted to different image sizes.

**19. What is the difference between classification and regression?**
- Classification predicts discrete categories or labels, while regression predicts continuous values.

**20. What is a validation set? What is a test set? Why do we need them?**
- Validation set: Used to tune hyperparameters and evaluate during training.
- Test set: Used to evaluate final model performance.
- They're needed to assess how well the model generalizes to unseen data.

**21. What will fastai do if you don't provide a validation set?**
- If no validation set is provided, fastai typically creates one by randomly splitting off a portion of the training data.

**22. Can we always use a random sample for a validation set? Why or why not?**
- We can't always use a random sample for a validation set, especially with time series data or when there's a specific structure to the data.

**23. What is overfitting? Provide an example.**
- Overfitting: When a model performs well on training data but poorly on new, unseen data. Example: memorizing noise in training data rather than learning general patterns.

**24. What is a metric? How does it differ from "loss"?**
- A metric is a measure used to evaluate model performance. Unlike loss, metrics are often more interpretable and directly related to the task.

**25. How can pretrained models help?**
- Pretrained models provide a good starting point, having already learned useful features from a large dataset.

**26. What is the "head" of a model?**
- The "head" of a model typically refers to the final layers that are task-specific, often added on top of a pretrained base.

**27. What kinds of features do the early layers of a CNN find? How about the later layers?**
- Early CNN layers find simple features like edges and textures. Later layers find more complex, abstract features specific to the task.

**28. Are image models only useful for photos?**
- No, image models can be useful for various types of 2D data, not just photos (e.g., medical imaging, satellite imagery, audio spectrograms).

**29. What is an "architecture"?**
- An "architecture" refers to the specific structure and arrangement of layers in a neural network.

**30. What is segmentation?**
- Segmentation is the task of dividing an image into multiple segments or objects, often assigning a class label to each pixel.

**31. What is `y_range` used for? When do we need it?**
- `y_range` specifies the range of the target variable in regression problems, constraining the output of the model to a specific range.

**32. What are "hyperparameters"?**
- Hyperparameters are configuration settings for the learning process that are not learned from the data (e.g., learning rate, batch size, number of epochs).

**33. What's the best way to avoid failures when using AI in an organization?**<br>
Best practices to avoid AI failures in organizations:
- Thorough testing and validation
- Continuous monitoring and updating of models
- Understanding and mitigating potential biases
- Having a diverse team with both technical and domain expertise
- Establishing clear ethical guidelines and governance structures
- Maintaining transparency about the capabilities and limitations of AI systems


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
