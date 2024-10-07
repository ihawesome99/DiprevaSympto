from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

app = Flask(__name__)

# Load the model from C4.5_DecisionTree.pkl
model_filename = 'data/cart_model.joblib'
model = joblib.load(model_filename)

# Define the selected features
gejala_list = [
    'Kulit Gatal-gatal', 'Batuk', 'Demam', 'Kencing Sakit',
    'Kencing Tidak Lancar', 'Nyeri Daerah Panggul', 'Muncul Bintik Merah',
    'Mual', 'Pilek', 'Perut Sakit', 'Badan Lemas', 'Meriang', 'Nyeri Ulu Hati',
    'Pusing', 'Tenggorokan Gatal', 'Kulit, bengkak', 'Tenggorokan Sakit', 'Muntah', 
    'Nyeri Telan', 'Sendawa', 'Diare', 'Perut Kembung', 'Badan Sakit', 'Lidah Pahit', 'Sesak'
]

diagnosis_info = {
    "ISPA": {
        "description": "Infeksi Saluran Pernapasan Akut adalah infeksi yang mempengaruhi saluran pernapasan bagian atas, termasuk hidung, tenggorokan, dan paru-paru.",
        "treatment": "Istirahat, minum banyak cairan, dan obat pereda nyeri."
    },
    "Dermatitis": {
        "description": "Dermatitis adalah peradangan pada kulit yang menyebabkan gatal, kemerahan, dan ruam.",
        "treatment": "Krim kortikosteroid, antihistamin, dan menghindari pemicu."
    },
    "Gastritis": {
        "description": "Gastritis adalah peradangan pada lapisan lambung yang dapat menyebabkan nyeri perut, mual, dan muntah.",
        "treatment": "Antasida, inhibitor pompa proton, dan menghindari makanan pedas."
    },
    "Typhoid Fever": {
        "description": "Demam Tifoid adalah infeksi bakteri yang disebabkan oleh Salmonella typhi yang menyebabkan demam tinggi, sakit perut, dan ruam.",
        "treatment": "Antibiotik, istirahat, dan hidrasi yang cukup."
    },
    "ISK": {
        "description": "Infeksi Saluran Kemih adalah infeksi yang mempengaruhi bagian manapun dari sistem kemih, termasuk ginjal, ureter, kandung kemih, dan uretra.",
        "treatment": "Antibiotik dan minum banyak cairan."
    }
}

@app.route('/services')
def services():
    return render_template('services.html', gejala_list=gejala_list)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = [int(request.form.get(gejala, 0)) for gejala in gejala_list]
    
    # Predict the diagnosis using the loaded model
    prediction = model.predict([user_input])[0]
    
    # Get the probabilities of each class
    probabilities = model.predict_proba([user_input])[0]
    prob_dict = {model.classes_[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
    
    # Get diagnosis details
    diagnosis_details = diagnosis_info.get(prediction, {"description": "Informasi tidak tersedia", "treatment": "Pengobatan tidak tersedia"})
    
    # Visualize the decision tree using plot_tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=gejala_list, class_names=model.classes_, filled=True, rounded=True, precision=2)
    tree_image_path = "static/decision_tree.png"
    plt.tight_layout()
    plt.savefig(tree_image_path, dpi=800, bbox_inches='tight')  # Increase dpi for better clarity
    plt.close()
    
    response_data = {
        'prediction': prediction,
        'prob_dict': prob_dict,
        'diagnosis_details': diagnosis_details,
        'tree_image': tree_image_path
    }

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(response_data)
    else:
        return render_template('services.html', gejala_list=gejala_list, prediction=prediction, prob_dict=prob_dict, tree_image=tree_image_path, diagnosis_details=diagnosis_details)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blog-single')
def blog_single():
    return render_template('blog-single.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/doctors')
def doctors():
    return render_template('doctors.html')

@app.route('/')
def home():
    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)
