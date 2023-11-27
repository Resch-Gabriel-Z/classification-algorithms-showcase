# Classification Algorithms Showcase

Welcome to the Classification Algorithms Showcase project! This repository serves as a collection of various classification algorithms implemented in Python. Whether you're a machine learning enthusiast or a practitioner, this project aims to provide a hands-on experience with popular classification algorithms.

## Table of Contents
- [Classification Algorithms Showcase](#classification-algorithms-showcase)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Implemented Algorithms](#implemented-algorithms)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [Decision Tree](#decision-tree)
    - [Random Forest](#random-forest)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
  - [Getting Started](#getting-started)
  - [View Interactive Website](#view-interactive-website)

## Introduction
The goal of this project is to showcase the implementation of various classification algorithms. I previously attempted this project with a more messy and inexperienced approach. Now armed with new knowledge and skills, I am revisiting this project to provide a cleaner and more organized implementation of different classification algorithms.


Feel free to explore the implemented algorithms, use them in your projects, as you wish.
## Implemented Algorithms

### Support Vector Machine (SVM)

SVMs are a type of supervised learning algorithm that aims to find the optimal hyperplane, with maximum margin between the classes.

visualized, imagine you want to divide a bunch of red and blue dots with a line, but not just in any way, you want that the space between the line and the first point on either side to be as far away as possible


### Decision Tree

A Decision Tree is an algorithm that tries to build a tree, such that each leaf node is just one type of class, by dividing the dataset recursively into 2 parts, depending on a certain feature and its value.

Think of a Decision Tree as a game of '20 Questions' with a dataset full of animals. You want to classify each animal into a specific category (like species) by asking a series of yes-or-no questions. Imagine you start with a broad question, like 'Is it taller than 6ft?' This question acts as the first split in the tree. The Decision Tree then creatively organizes the animals: those taller than 6ft in one branch, and the rest in another. Now, you have two groups. For each group, you continue this process, asking new questions and dividing until each leaf node represents a specific type of animal. It's like navigating through a branching set of criteria to pinpoint exactly where each animal belongs, step by step.
### Random Forest


### K-Nearest Neighbors (KNN)

KNN is an algorithm that assumes that similar data can be represented in space such that they are close to each other.
With that in mind, KNN simply claims the data point is of the same class then its neighbors (it's k-closest to be exact).

As an analogy, imagine you would pick a movie, but you don't know its genre, but the 5 surrounding movies are all fantasy movies, so you conclude that it is most likely a fantasy movie. 

## Getting Started

Follow these steps to set up the project on your local machine.

1. Clone the repository:

    ```bash
    git clone https://github.com/Resch-Gabriel-Z/classification-algorithms-showcase.git
    cd classification-algorithms-showcase
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## View Interactive Website
The Website has the benefit of visualizing the algorithms in a better way. to run it simply navigate to the src folder and run:

```bash
python viewDemo.py
```
