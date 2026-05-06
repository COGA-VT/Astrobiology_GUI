import streamlit as st


st.title('About')
st.subheader("Teachable Artificial Intelligence for Astrobiology Investigations (AI$^{2}$)")
st.markdown(
    "This application is designed to allow astrobiologists to experiment with machine learning approaches including unsupervised and "
    "supervised methods using their own data."
    " This application will be especially useful for those who may be interested in applying machine learning but either don't know "
    "where to start or do not have the time to learn programming languages.")

st.markdown('Developed by Floyd Nichols and Grant Major')
st.markdown('email: floydnichols@vt.edu\n\n'
            'email: grantmajor@vt.edu')

st.divider()
st.subheader("Machine Learning Workflow")
with st.container(horizontal_alignment="center"):
    st.image("assets/ML_Steps.png")

with st.expander("Exploratory Data Analysis"):
    st.write("Exploratory Data Analysis (EDA) is typically the first step of " \
    "the ML workflow. EDA is performed to gain insight into the structure of the data through visualization and data summaries." \
    " After gaining familiarity with the data, it is easier to select preprocessing techniques and viable ML models.")

with st.expander("Data Preprocessing"):
    st.write("Data preprocessing encompasses multiple key steps, like data cleaning and feature engineering. Real world data" \
    " is messy and typically contains erroneous entries. Data cleaning targets this by handling duplicate entries, missing values, and other errors. " \
    "Feature engineering employs techniques like encoding and scaling to improve compatibility with ML models. Encoding " \
    "converts categorical features to numerical features while scaling aims to align the scale of numerical variables.")

with st.expander("Model Selection"):
    st.write("Selecting the model that best matches the structure of your data is integral to producing a model that generalizes to unseen data. " \
    "Different models pose different use cases and selecting the correct one has a direct effect on the utility of your model. When selecting a model, " \
    "it is important to understand your data, your use case, and your computational power. It is common for multiple models to be trained, tuned, and compared before " \
    "selecting a final model. Our Defintions and Use Cases page offers a brief overview of the strengths and weaknesses of each model. ")

with st.expander("Model Evaluation"):
    st.write("Model evaluation is the process of assessing the performance of a trained model. This is typically done by comparing a model's predictions to the true labels of a test set." \
    "From these predictions and labels, various metrics can be calculated to evaluate the performance of the model. Through hyperparameter tuning, the performance of a model can be improved. " \
    "Specific metrics can be viewed in the Definitions and Use Cases page.")