import os
import tkinter as tk
import webbrowser
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import threading
import time
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("Agg")  # Use Agg backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

report_data = {}

# Function to generate a Pie chart
def plot_pie_chart(features, title):
    figure, ax = plt.subplots()
    data = features['Importance']
    labels = features['Feature']

    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    return figure

def update_progress_bar():
    global analysis_progress
    if analysis_progress < 100:
        progress_var.set(analysis_progress)
        window.after(500, update_progress_bar)

def generate_pdf_report(report_data, file_path):
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    story = []

    # Add a title to the report
    styles = getSampleStyleSheet()
    title = "Brain Tumor Classifier Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Add a section for the analysis results
    story.append(Paragraph("Analysis Results:", styles["Heading1"]))
    story.append(Spacer(1, 12))

    for key, value in report_data.items():
        story.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))

    # Add a table for the feature importances
    feature_importance_data = report_data.get("Feature Importances", None)
    if feature_importance_data is not None and not feature_importance_data.empty:
        feature_importance_data = [["Feature", "Importance"]] + feature_importance_data.values.tolist()
        table = Table(feature_importance_data, colWidths=[200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),
            ('TEXTCOLOR', (0, 0), (-1, 0), (0, 0, 0)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), (0.85, 0.85, 0.85)),
            ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0)),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),  # Increase left padding
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),  # Increase right padding
            ('TOPPADDING', (0, 0), (-1, -1), 12),  # Increase top padding
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),  # Increase bottom padding
        ]))

        story.append(table)

    doc.build(story)

def open_report():
    report_file_path = "brain_tumor_classifier_report.pdf"
    if os.path.exists(report_file_path):
        webbrowser.open(report_file_path)
    else:
        messagebox.showerror("Error", "Report file not found.")

def analyze_and_visualize(file_path):
    global analysis_progress
    analysis_progress = 0
    progress_var.set(analysis_progress)

    data = pd.read_csv(file_path)

    label_encoder = LabelEncoder()
    data['Tumor_Present'] = label_encoder.fit_transform(data['Tumor_Present'])

    X = data.drop(columns=['Tumor_Present'])
    y = data['Tumor_Present']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegressionCV(cv=5, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier()
    }

    best_accuracies = {}
    top_features = {}
    accuracy_data = []
    feature_importance_data = {}

    for name, classifier in classifiers.items():
        print(f"Training {name}...")

        feature_selector = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_selector.fit(X_train, y_train)
        feature_importances = feature_selector.feature_importances_

        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        threshold = 0.01
        selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
        top_features[name] = selected_features

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        classifier.fit(X_train_selected, y_train)
        y_pred = classifier.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        best_accuracies[name] = accuracy
        accuracy_data.append((name, accuracy))
        feature_importance_data[name] = feature_importance_df.to_dict(orient='list')

        print(f"The most important features for {name} are: {', '.join(selected_features)}")
        print(f"Accuracy with selected features for {name}: {accuracy * 100:.2f}%")
        print()

        time.sleep(2)
        analysis_progress += 18
        progress_var.set(analysis_progress)
        update_progress_bar_thread = threading.Thread(target=update_progress_bar)
        update_progress_bar_thread.start()

    best_classifier = max(best_accuracies, key=best_accuracies.get)
    soft_voting_classifier = VotingClassifier(
        estimators=[(name, classifier) for name, classifier in classifiers.items()], voting='soft')
    soft_voting_classifier.fit(X_train, y_train)
    y_pred_soft = soft_voting_classifier.predict(X_test)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    print(f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%")

    hard_voting_classifier = VotingClassifier(
        estimators=[(name, classifier) for name, classifier in classifiers.items()], voting='hard')
    hard_voting_classifier.fit(X_train, y_train)
    y_pred_hard = hard_voting_classifier.predict(X_test)
    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    print(f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%")
    print(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%")

    time.sleep(2)
    analysis_progress += 18
    progress_var.set(analysis_progress)
    update_progress_bar_thread = threading.Thread(target=update_progress_bar)
    update_progress_bar_thread.start()

    # Clear the canvas before displaying new figures
    for widget in canvas.winfo_children():
        widget.destroy()

    # Create a bar chart for classifier accuracies
    accuracies = [accuracy[1] for accuracy in accuracy_data]
    classifiers = [accuracy[0] for accuracy in accuracy_data]

    fig = Figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    ax.bar(classifiers, accuracies, color='skyblue')
    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classifier Accuracies')

    # Create a pie chart for feature importances
    pie_chart_figure = plot_pie_chart(pd.DataFrame(feature_importance_data[best_classifier]), f"Feature Importances for {best_classifier}")

    canvas1 = FigureCanvasTkAgg(fig, master=canvas)
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    canvas2 = FigureCanvasTkAgg(pie_chart_figure, master=canvas)
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Report the results in a text box
    result_text.config(state="normal")
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%\n")
    result_text.insert(tk.END, f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%\n")
    result_text.insert(tk.END, f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%\n")
    result_text.insert(tk.END, "Top Features for the Best Classifier:\n")
    for i, param in enumerate(top_features[best_classifier], start=1):
        result_text.insert(tk.END, f"{i}. {param}\n")
    result_text.config(state="disabled")

    # Generate a report file
    with open("report.txt", "w") as report_file:
        report_file.write(f"The best classifier is {best_classifier} with accuracy {best_accuracies[best_classifier] * 100:.2f}%\n")
        report_file.write(f"Soft Voting Classifier Accuracy: {accuracy_soft * 100:.2f}%\n")
        report_file.write(f"Hard Voting Classifier Accuracy: {accuracy_hard * 100:.2f}%\n")
        report_file.write("Top Features for the Best Classifier:\n")
        for i, param in enumerate(top_features[best_classifier], start=1):
            report_file.write(f"{i}. {param}\n")

    report_data["Best Classifier"] = best_classifier
    report_data["Feature Importances"] = pd.DataFrame(feature_importance_data[best_classifier])
    report_data["Soft Voting Classifier Accuracy"] = accuracy_soft
    report_data["Hard Voting Classifier Accuracy"] = accuracy_hard

    # Generate the PDF report
    generate_pdf_report(report_data, "brain_tumor_classifier_report.pdf")

    print("Results and report generated.")

def browse_file():
    file_path = filedialog.askopenfilename(title="Select a dataset file", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        messagebox.showerror("Error", "Please select a valid dataset file.")
    else:
        analysis_thread = threading.Thread(target=analyze_and_visualize, args=(file_path,))
        analysis_thread.start()

window = tk.Tk()
window.title("Brain Tumor Classifier")
window.geometry("1000x800")
window.configure(bg="lightgray")

label = tk.Label(window, text="Brain Tumor Classifier", font=("Helvetica", 20), bg="lightgray")
label.pack(pady=20)

file_frame = tk.Frame(window, bg="lightgray")
file_frame.pack()

file_label = tk.Label(file_frame, text="Select a dataset file (CSV):", bg="lightgray")
file_label.pack(side="left", padx=10)

view_report_button = tk.Button(window, text="View Report", command=open_report, bg="green", fg="white")
view_report_button.pack(pady=10)

select_button = tk.Button(file_frame, text="Browse", command=browse_file, bg="blue", fg="white")
select_button.pack(side="left")

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(window, length=800, variable=progress_var)
progress_bar.pack(pady=10)

canvas = tk.Frame(window, bg="lightgray")
canvas.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

result_text = tk.Text(window, height=10, width=50, wrap=tk.WORD)
result_text.pack(pady=10)

# Create a Scrollbar widget for scrolling the Text widget
scrollbar = tk.Scrollbar(window, command=result_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the Text widget to use the Scrollbar
result_text.config(yscrollcommand=scrollbar.set)

window.mainloop()
