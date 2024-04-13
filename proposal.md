### Title: Summarizes the main idea of your project.
Sample-Efficient play of Atari Games with Multi-Modal input

### Who: Names and logins of all your group members.
- Noah Rousell: `nrousell`
- Ryan Eng: 
- Julian Beaudry: 
- Adithya Sriram: 

### Introduction: What problem are you trying to solve and why?
We are implementing the paper [Mastering Atari Games with Limited Data](https://arxiv.org/pdf/2111.00210.pdf) and adding audio implement as well. The original model, `EfficientZero` was able to learn many Atari 2600 games at a superhuman level with only 2-hours of game-time.

We will implement a similar sample-efficient architecture that also has the capacity to intake audio as input. To focus our efforts (and due to limited compute), we plan to train our model on the Atari 2600 game *Amidar*, where audio is useful for the player.

### Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?
- [2015 DQN Paper](https://www-nature-com.revproxy.brown.edu/articles/nature14236)
  - This paper introduced Deep Q-Learning, which uses neural networks to perform Q-learning directly on high-dimensional sensory inputs. This removes the need for hand-engineered features on many Reinforcement Learning tasks.
- [EfficientZero Paper](https://arxiv.org/pdf/2111.00210.pdf)
  - Built on top of [MuZero](https://arxiv.org/pdf/1911.08265.pdf)
  - This 2021 paper introduced the first model architecture that has a mean performance better than humans on the limited Atari 100k training dataset. 
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
  - [PR that added audio support](https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/233)
- [CrossModal Attentive Skill Learner](https://arxiv.org/pdf/1711.10314.pdf)
    - This paper develops an architecture that attends to both video and audio inputs to train a model to play the Atari games *Amidar* and *H.E.R.O.*

### Data: What data are you using (if any)?
We plan on using the Atari 100k benchmark dataset, which consists of 100 thousand observations from an Atari game with a frame-skip of 4 (around 2 hours of game-time). 

### Methodology: What is the architecture of your model?
How are you training the model?
If you are implementing an existing paper, detail what you think will be the hardest part about implementing the model here.
If you are doing something new, justify your design. Also note some backup ideas you may have to experiment with if you run into issues.
Metrics: What constitutes “success?”
What experiments do you plan to run?
For most of our assignments, we have looked at the accuracy of the model. Does the notion of “accuracy” apply for your project, or is some other metric more appropriate?
If you are implementing an existing project, detail what the authors of that paper were hoping to find and how they quantified the results of their model.
If you are doing something new, explain how you will assess your model’s performance.
What are your base, target, and stretch goals?

### Ethics: Choose 2 of the following bullet points to discuss
- What broader societal issues are relevant to your chosen problem space?
- Why is Deep Learning a good approach to this problem?
- What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
- Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
- How are you planning to quantify or measure error or success? What implications does your quantification have?
- 
### Add your own: if there is an issue about your algorithm you would like to discuss or explain further, feel free to do so.

### Division of labor: Briefly outline who will be responsible for which part(s) of the project.