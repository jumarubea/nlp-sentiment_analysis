# NLP Sentiment Analysis Tutorial

Welcome to the **NLP Sentiment Analysis Tutorial** repository! This project is designed to help you understand and implement sentiment analysis using Python and popular NLP libraries. It's beginner-friendly and provides a structured approach to learning sentiment analysis step-by-step.



## Project Structure

```
nlp-sentiment_analysis/
├── main.ipynb        # Main notebook with code and explanations
├── utils.py          # Utility functions for text processing and helper function
├── requirements.txt  # List of dependencies
└── README.md         # Project documentation
```


## Features

- **Step-by-Step Guide**: Walkthrough of key sentiment analysis concepts.
- **Preprocessing Utilities**: Helper functions in `utils.py`.
- **Hands-On Notebook**: `main.ipynb` contains a detailed tutorial and examples.



## Getting Started

### Prerequisites

Make sure you have Python 3.7 or higher installed. You'll also need Jupyter Notebook or Jupyter Lab to run the `main.ipynb`.

### Installation Steps

1. **Clone the Repository**

   Clone the repository to your local machine using:

   ```bash
   git clone https://github.com/jumarubea/nlp-sentiment_analysis.git
   cd nlp-sentiment_analysis
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**

   Create a virtual environment to isolate dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate        # On MacOS/Linux
   venv\Scripts\activate           # On Windows
   ```

3. **Install Dependencies**

   Install the required Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**

   Start the Jupyter Notebook server and open `main.ipynb`:

   ```bash
   jupyter notebook
   ```

   Navigate to the folder and click on `main.ipynb` to get started.

## Additional Tools

We are going to use `twitter_samples` datasets and english `stopwords` by installing them directly to from `nltk`

```Python
nltk.download('twitter_samples')

nltk.download('stopwords')
```


## Workflow

1. **Data Loading**: Import text data for analysis (examples provided in the notebook).
2. **Text Preprocessing**: Utilize utilities from `utils.py` for tasks like tokenization, stopword removal, and lemmatization.
3. **Sentiment Analysis**: Apply machine learning or rule-based methods to classify sentiments.
4. **Visualization**: Plot sentiment distributions and other insights using tools like Matplotlib and Seaborn.




## Dependencies

This project uses the following Python libraries:

- `nltk` - For natural language processing.
- `numpy` - For numerical computations.
- `pandas` - For data manipulation and analysis.
- `pprint` - For pretty print of json files

To install these, run:

```bash
pip install -r requirements.txt
```


## Contribution

Contributions are welcome! If you’d like to improve this tutorial or fix issues, feel free to fork the repository and submit a pull request.



## Acknowledgments

Special thanks to the open-source community for the libraries used in this project.
