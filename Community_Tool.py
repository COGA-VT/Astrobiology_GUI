## Load Necessary Libraries
# House Keeping Libraries
import streamlit as st
import numpy as np
from numpy import mean, std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Machine Learning Libraries and Modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, \
    ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.multiclass import OneVsRestClassifier


# Set Layout of the Application
st.set_page_config(layout="wide")

# Main Page Set Up
st.title("Teachable Artificial Intelligence for Astrobiology Investigations (AI$^{2}$)")
tab1, tab2, tab3, tab4 = st.tabs(["About",
                                  "Data and Preprocessing",
                                  "Unsupervised Learning",
                                  "Supervised Learning"]
                                 )

with tab1:
    st.markdown(
        "This application is designed to allow astrobiologists to experiment with machine learning approaches including unsupervised and "
        "supervised methods using their own data."
        " This application will be especially useful for those who may be interested in applying machine learning but either don't know "
        "where to start or do not have the time to learn programming languages.")

    st.markdown('Developed by Floyd Nichols and Grant Major')
    st.markdown('email: floydnichols@vt.edu\n\n'
                'email: grantmajor@vt.edu')

with tab2:
    st.subheader("Upload a Data File")
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.error('Need to upload a file')
    else:
        def load_data(data_file, nrows):
            data = pd.read_csv(data_file, nrows=nrows)
            return data


        data_load_state = st.text('Loading data...')
        data = load_data(uploaded_file, 10000)
        data_load_state.text("Done!")

        if st.checkbox('Show Data'):
            st.subheader('Data')
            st.dataframe(data=data)

    st.divider()
    # Access and Download Example Data
    st.subheader(
        'If you do not have data of your own, use the following link to access available training sets from NASA AI Astrobiology')
    st.link_button('Go to NASA AI Astrobiology', 'https://ahed.nasa.gov/')

with tab3:
    col1, col2 = st.columns([1, 1])

    # Test code to make sure that a data file is uploaded before continuing
    try:
        data = data
    except:
        st.error("Please make sure that a data file is uploaded")
        st.stop()

    # Construct Dimensionality Reduction and Clustering
    with col1:
        st.subheader('**Here, the user can employ dimensionality reduction and clustering methods.**')
        st.divider()

        # Remove Columns that are Strings
        X = data.select_dtypes(include=['int64', 'float64'])

        st.subheader('Data and Hyperparameter Selection')

        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X = X.dropna()

        elements = st.multiselect("Select Explanatory Variables (default is all numerical columns):",
                                  X.columns,
                                  default=X.columns
                                  )

        y = data
        y = y.dropna()

        target = st.selectbox('Choose Target',
                              options=y.columns,
                              )

        options = st.selectbox(label='Select Dimensionality Reduction Method',
                               options=['Standard PCA',
                                        't-SNE'])

        # t-SNE Construction
        st.divider()
        # Set random state of the subsequent scripts
        np.random.seed(42)

        if options == 't-SNE':
            st.subheader('Define t-SNE Parameters')

            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )
            perplexity = st.number_input('Insert Perplexity',
                                         min_value=2
                                         )
            tsne = TSNE(n_components,
                        random_state=42,
                        perplexity=perplexity,
                        n_jobs=-1,
                        method='exact',
                        max_iter=5000
                        )
            tsne_result = tsne.fit_transform(X)
            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0],
                                           'tsne_2': tsne_result[:, 1]}
                                          )

            # DBSCAN
            st.divider()
            clusters = st.selectbox(label='Select Cluster Method',
                                    options=['Kmeans',
                                             'DBSCAN',
                                             'Target'])

            if clusters == 'Kmeans':
                st.subheader('Define K-means Parameters')
                n_clusters = st.number_input('Enter Number of Clusters',
                                             min_value=2
                                             )

                X_Kmeans = KMeans(n_clusters=n_clusters).fit(tsne_result)
                labels = X_Kmeans.labels_

            elif clusters == 'DBSCAN':
                st.subheader('Define DBSCAN Parameters')
                eps = st.number_input('Enter Eps',
                                      min_value=0.5
                                      )
                min_samples = st.number_input('Enter Minimum Samples',
                                              min_value=1
                                              )

                X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(tsne_result)
                DBSCAN_labels = X_DBSCAN.labels_
                DBSCAN_labels = DBSCAN_labels.astype(str)
                labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

            else:
                labels = y

            with col2:
                # Plot t-SNE Results
                st.subheader('t-Distributed Stochastic Neighbor Embedding')
                fig, ax = plt.subplots()
                fig = px.scatter(tsne_result_df,
                                 x='tsne_1',
                                 y='tsne_2',
                                 color=labels,
                                 title=options
                                 )
                fig.update_traces(
                    marker=dict(size=8,
                                line=dict(width=2,
                                          color='Black')
                                )
                )
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_aspect('auto')
                ax.legend('Cluster',
                          bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.subheader('Define Standard PCA Parameters')
            X = X[elements]  # Make prediction based on selected elements
            y = y[target]

            n_components = st.number_input('Insert Number of Components',
                                           min_value=2
                                           )

            pca = PCA(n_components=n_components)
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('pca', pca)])
            Xt = pipe.fit_transform(X)

            with col1:
                # DBSCAN
                st.divider()
                clusters = st.selectbox(label='Select Cluster Method',
                                        options=['Kmeans',
                                                 'DBSCAN',
                                                 'Target'])
                if clusters == 'Kmeans':
                    st.subheader('Define K-means Parameters')
                    n_clusters = st.number_input('Enter Number of Clusters',
                                                 min_value=2
                                                 )

                    X_Kmeans = KMeans(n_clusters=n_clusters).fit(Xt)
                    labels = X_Kmeans.labels_

                elif clusters == 'DBSCAN':
                    st.subheader('Define DBSCAN Parameters')
                    eps = st.number_input('Enter Eps',
                                          min_value=0.5
                                          )
                    min_samples = st.number_input('Enter Minimum Samples',
                                                  min_value=1
                                                  )

                    X_DBSCAN = DBSCAN(eps=eps, min_samples=min_samples).fit(Xt)
                    DBSCAN_labels = X_DBSCAN.labels_
                    DBSCAN_labels = DBSCAN_labels.astype(str)
                    labels = [outlier.replace('-1', 'Outlier') for outlier in DBSCAN_labels]

                else:
                    labels = y

            with col2:
                # Plot PCA Results
                st.subheader('Principal Component Analysis')
                fig, ax = plt.subplots()
                PCA_df = pd.DataFrame({'PCA_1': Xt[:, 0],
                                       'PCA_2': Xt[:, 1],
                                       'labels': labels},
                                      )
                fig = px.scatter(PCA_df,
                                 x='PCA_1',
                                 y='PCA_2',
                                 color=labels,
                                 title=options)
                fig.update_traces(
                    marker=dict(size=12,
                                line=dict(width=2,
                                          color='Black')
                                )
                )
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                ax.set_aspect('auto')
                ax.legend(bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

                # Define and Plot Explained Variance Ratio
                fig, ax = plt.subplots()
                exp_var_pca = pca.explained_variance_ratio_
                fig = px.bar(exp_var_pca,
                             x=range(0, len(exp_var_pca)),
                             y=exp_var_pca,
                             title='PCA Explained Variance Ratio')

                fig.update_traces(
                    marker=dict(color='grey',
                                line=dict(width=3,
                                          color='Black')
                                )
                )

                fig.update_layout(
                    xaxis_title='Principal Component Index',
                    yaxis_title='Explained Variance Ratio'
                )
                ax.set_aspect('auto')
                ax.legend(bbox_to_anchor=(0.8, 0.95),
                          loc=2,
                          borderaxespad=0.0)
                st.plotly_chart(fig, use_container_width=True)

with tab4:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader('**Here, the user can employ regression and classification methods.**')
        st.divider()

        # Remove Columns that are Strings
        X_sup = data.select_dtypes(include=['int64', 'float64'])

        st.subheader('Data and Hyperparameter Selection')

        X_sup = X_sup.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        X_sup = X.dropna()

        elements_sup = st.multiselect("Select Explanatory Variables (default is all numerical columns):",
                                      X_sup.columns,
                                      placeholder='Choose Option',
                                      default = X_sup.columns,
                                      )

        y_sup = data
        y_sup = y_sup.dropna()

        #TODO: Make it so that the last column is the target variable automatically
        target_sup = st.selectbox('Choose Target',
                                  options=y_sup.columns,
                                  placeholder='Choose Option'
                                  )

        ## Grant will work on this section ##
        #TODO: Make it so that an error message appears when no data file is uploaded

        #CLASSIFICATION CODE: ------------------------------------------------------------------------------------------

        #setting variable to None to prevent errors from undeclared variables:
        class_algorithim = None
        X_predictions = None
        predicting_data = None

        options_sup = st.selectbox(label='Select Prediction Type',
                                   options=['Classification',
                                            'Regression'])


        #BEGIN TRAIN TEST SPLIT SECTION
        st.subheader('Train Test Split: ')
        train_proportion = st.number_input('Enter the proportion of data to be allocated to training.',
                                           min_value=0.0,
                                           value = 0.75,
                                           step = 0.01,
                                           format = "%.2f")
        X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=train_proportion)

        st.divider()
        #END TRAIN TEST SPLIT SECTION

        #BEGIN MODEL SELECTION SECTION
        if options_sup == 'Classification':
            class_algorithim = st.selectbox(label='Choose Classification Algorithm',
                                            options=['Support Vector Machine (SVM)',
                                                     'k-Nearest Neighbors (k-NN)',
                                                     'algorithm 3']
                                            )
            if class_algorithim == 'Support Vector Machine (SVM)':
                X = data[elements_sup]
                y = data[target_sup]
                #Sets degree to a default value in case kernel_type isn't polynomial and thus degree isn't declared
                degree = 3

                #Selecting C value hyperparameter
                c_value = st.number_input('Input C Value', min_value=0.0,
                                          value=1.0,
                                          step=0.01,
                                          format="%.2f")

                #Selecting Kernel hyperparameter
                kernel_type = st.selectbox('Select a kernel',
                                           ('Linear', 'Polynomial', 'Radial Basis Function'),
                                           index=0)
                st.write('You selected:', kernel_type)

                #Selecting Degree hyperparameter
                if kernel_type == 'Polynomial':
                    degree = st.number_input('Enter a degree', min_value=0)


                #TODO Can add additional advanced settings
                #advanced_settings = st.checkbox('Enable advanced settings', value = False,)
                #if advanced_settings:


                #changes the kernel_type var to a valid value for the svm function
                if kernel_type == 'Linear':
                    kernel_type = 'linear'
                elif kernel_type == 'Polynomial':
                    kernel_type = 'poly'
                else:
                    kernel_type = 'rbf'

                # creates svm model using inputted values
                support_vector_machine = svm.SVC(C=c_value,
                                                 kernel=kernel_type,
                                                 degree=degree)

                selected_model = support_vector_machine


            #TODO Finish implementation of k-NN
            elif class_algorithim == 'k-Nearest Neighbors (k-NN)':

                k_value = st.number_input('Input k value.', min_value=1, value=1, )

                knn = KNeighborsClassifier(n_neighbors=k_value)
                selected_model = knn


            #fits the selected model to the training data obtained from train_test_split
            selected_model.fit(X_train, y_train)


    #TODO: Check for common variable in Unsupervised section
    #TODO: Move prediction data below metrics
    #BEGIN MODEL METRICS SECTION
    with col2:

        #Reads in data file that user wants predictions on
        predicting_data_file = st.file_uploader('Upload a file with values to be predicted.')

        if predicting_data_file is None:
            st.error('Need to upload a prediction file.')
        else:
            data_load_state2 = st.text('Loading data....')
            predicting_data = load_data(predicting_data_file, 10000)
            data_load_state2.text('Done!')


        st.divider()
        st.header("Model Performance Metrics")

        #BEGIN CLASSIFICATION REPORT CODE
        st.subheader("Classification Report:")

        y_predictions = selected_model.predict(X_test)
    #model.classes for labels
        class_report = classification_report(y_test, y_predictions, output_dict=True)
        st.text(class_report)
        class_report = pd.DataFrame(class_report)
        class_report_fig = sns.heatmap(class_report)


        st.pyplot(class_report_fig.get_figure())
        st.divider()
        #END CLASSIFICATION REPORT CODE

        #BEGIN CONFUSION MATRIX CODE
        st.subheader("Confusion Matrix:")
        conf_mat = confusion_matrix(y_test, y_predictions)
        st.text(conf_mat)
        conf_mat_fig = sns.heatmap(conf_mat)
        st.pyplot(conf_mat_fig.get_figure())
        #END CONFUSION MATRIX CODE




