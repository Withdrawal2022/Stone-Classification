# Hierarchical Stone Classification

This project implements a hierarchical stone classification system using EfficientNet-V2-S and multi-task learning, as detailed in the accompanying report.

---

## Getting Started

To run this project, follow these steps:

### 1. Project Structure

Ensure your project directory is structured as follows:

your_project_root/
├── datasets/
│   └── (contains all stone images)
├── cache_preprocess_384.py
├── stone_final_run.ipynb
└── README.md

Place `cache_preprocess_384.py` and `stone_final_run.ipynb` in the same directory as your `datasets` folder. The `datasets` folder should contain all your stone images.

### 2. Preprocessing Data

Before training, you need to preprocess the images and create cached tensors. This will generate a `cache_384` folder within your `datasets` directory.

Open your terminal or command prompt, navigate to `your_project_root/`, and run the following command:

```bash
python cache_preprocess_384.py
```

### 3. Train the Model and Get Results
Once preprocessing is complete, you can train the model and obtain results by following the steps outlined in the stone_final_run.ipynb Jupyter Notebook.

Open the notebook and execute the cells sequentially to train your model and evaluate its performance.

If you have any questions or encounter issues, please refer to the project report for more detailed information.
