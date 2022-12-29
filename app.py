import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
from PIL import Image
from utils import load_and_prep, get_classes, returnModel
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
# @st.cache(suppress_st_warning=True)
def predicting(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "Scores": values
                       })
    df = df.sort_values('Scores')
    return pred_class, pred_conf, df


class_names = get_classes()


# @st.cache(suppress_st_warning=True)
def prepareModel():
    # global model
    model = tf.keras.models.load_model("./models/FinalModel.hdf5")
    st.session_state['model'] = model

### PLot function  ###


def plotFoodInfo(namefood):
    foodinfo = nutritiondf[nutritiondf['name'] == namefood].loc[:, [
        'protein', 'calcium', 'fat', 'carbohydrates', 'vitamins']]
    labels = foodinfo.columns.to_numpy()
    values = foodinfo.values[0]
    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[
                        [{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=labels, values=values))

    fig.update_layout(
        title_text="Nutrition Facts of "+namefood + " (100g)",
        margin=dict(l=3, r=3),
        font=dict(
            # family="Courier New, monospace",
            size=20
            # color="RebeccaPurple"
        ),
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="Fruits<br><sup>Fruit sales in the month of January</sup>"
            )
        )
    )
    st.write(fig)
    urlname = namefood.replace(" ", "-")
    st.write(
        f"You can check out this [link](https://www.nutritionix.com/food/{urlname})")


def plotPrediction(labels, values):
    y_saving = values
    x = labels

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        marker=dict(
            color='rgba(50, 171, 96, 0.6)',
            line=dict(
                color='rgba(50, 171, 96, 1.0)',
                width=1),
        ),
        orientation='h',
    ))
    fig.update_layout(
        title='<b>Top 5 Predictions</b>',
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            # domain=[0, 0.85],
            tickfont=dict(size=15),
        ),
        xaxis_title="Percentage of prediction",
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        margin_pad=10,
    )

    annotations = []

    y_s = np.round(y_saving, decimals=2)

    # Adding labels
    for yd, xd in zip(y_s, x):
        annotations.append(dict(xref='x1', yref='y1',
                                y=xd, x=yd + 6,
                                text=str(yd) + '%',
                                font=dict(family='Arial', size=16,
                                          color='rgb(50, 171, 96)'),
                                showarrow=False))

    fig.update_layout(annotations=annotations)
    st.write(fig)


def plotHistory(history_load):
    loss = history_load['loss']
    val_loss = history_load['val_loss']

    accuracy = history_load['accuracy']
    val_accuracy = history_load['val_accuracy']

    epochs = np.arange(0, len(history_load['loss']))

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Accuracy', 'Loss'))
    fig.add_trace(
        go.Scatter(x=epochs, y=accuracy, name='accuracy'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=val_accuracy, name='val_accuracy'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=loss, name='loss'),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='val_loss'),
        row=1, col=2
    )

    fig.update_layout(height=600, width=800,
                      # title_text="Evaluate the model",
                      hoverlabel=dict(font=dict(color='white')),
                      yaxis=dict(
                          domain=[0, 0.75],
                      ),
                      yaxis2=dict(
                          domain=[0, 0.75],
                      ),
                      legend=dict(x=0.029, y=1.3, font_size=16),
                      font=dict(size=20)
                      )
    fig.update_annotations(yshift=-70, font=dict(size=20))

    # Update xaxis properties
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)
    return fig


def plotF1(class_f1_scores):

    # class_f1_scores=np.load('./history/score_101.npy',allow_pickle='TRUE').item()
    report_df = pd.DataFrame(class_f1_scores, index=['f1-scores']).T
    report_df = report_df.sort_values("f1-scores", ascending=False)
    class_names = report_df.index.values.tolist()
    fig, ax = plt.subplots(figsize=(12, 25))
    scores = ax.barh(range(len(report_df)), report_df["f1-scores"].values)
    ax.set_yticks(range(len(report_df)))
    plt.axvline(x=0.80, linestyle='--', color='r')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("f1-score")
    ax.set_title("F1-Scores for 101 Different Classes")
    ax.invert_yaxis()  # reverse the order
    return fig

# ------------------------------


st.set_page_config(page_title="Food Vision",
                   page_icon="🍔")


#### SideBar ####

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Projects"],  # required
        icons=["house", "book"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
    )

st.sidebar.title("What's Food Vision ?")
st.sidebar.write("""
FoodVision is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 

It can identify over 100 different food classes


**Accuracy :** **`85%`**

**Model :** **`EfficientNetB1`**

**Dataset :** **`Food101`**
""")


#### Main Body ####
@st.cache(allow_output_mutation=True)
def cached_model():
    # URL = "https://github.com/hasibzunair/cxr-predictor/releases/latest/download/CheXpert_DenseNet121_res224.h5"
    # URL = "https://github.com/lgbao123/food-101/raw/main/models/FinalModel.h5"
    # weights_path = tf.keras.utils.get_file(
    #     "FinalModel.h5",
    #     URL)
    model = tf.keras.models.load_model("./models/FinalModel.hdf5", compile=False)
    return model
model = cached_model()
if selected == "Home":


    nutritiondf = pd.read_csv("nutrition101.csv")
    st.title("Food Vision 🍔📷")
    st.header("Identify what's in your food photos!")

    file = st.file_uploader(label="Upload an image of food.",
                            type=["jpg", "jpeg", "png"])

    if not file:
        st.warning("Please upload an image")
        st.stop()

    else:
        image = file.read()
        st.image(image, use_column_width=True)
        if model:
            pred_button = st.button("Predict")

            if pred_button:
                start_time = time.time()
                pred_class, pred_conf, df = predicting(image, model)
                
                score_list =df['Scores'].to_list()
                score_list = sorted(score_list, reverse=True)
                check = all([score_list[0]>=5,score_list[1]>=5,score_list[2]>= 3])
                if check:
                    st.warning("It's not food or not in database ")
                else:    
                    st.success(f'Prediction : {pred_class}')
                    st.success(f'Confidence : {pred_conf*100:.2f}%')
                    st.write("Took {} seconds to run.".format(
                        round(time.time() - start_time, 3)))
                    pred_class = pred_class.replace("_", " ")
                    st.write(df[['Top 5 Predictions', 'Scores']].sort_values(
                        'Scores', ascending=False))
                    
                    plotPrediction(df['Top 5 Predictions'].to_numpy(),
                                df['Scores'].to_numpy())
                    plotFoodInfo(pred_class)
if selected == "Projects":
    st.title(f"Multiclass Classification using TensorFlow 2.0 on Food-101 Dataset")

    st.markdown(
        """
    The Food-101 Data Set :
    - This dataset consists of 101 food categories, with 101'000 images
    - Each type of food has 750 training samples and 250 test samples
    - Size : 5 gb 
    """
    )
    st.write("TensorFlow Workflow : ")
    image = Image.open('./extras/workflow.png')
    # image1 = Image.open('./extras/before_pre.png')
    # image2 = Image.open('./extras/after_pre.png')
    pre_image = Image.open('./extras/pre.png')
    fit_2_image = Image.open('./extras/2_fit.png')
    fit_10_image = Image.open('./extras/10_fit.png')
    fit_101_image = Image.open('./extras/101_fit.png')
    st.image([image], use_column_width=True)
    # st.image([image1,image2])
    st.markdown(
        """
    Preprocessing data :
    - Load the image and convert into numpy array
    - Resize the image (to the same size our model was trained on)
    - Rescale the image (get all values between 0 and 1)
    """
    )
    # cols= st.columns(2)
    # cols[0].image('./extras/before_pre.png')
    # cols[1].image('./extras/after_pre.png')
    # col1, col2  = st.beta_columns([1,2])

    # with col1:
    #     st.image([image1],use_column_width=True)

    # with col2:
    #     st.image([image3],use_column_width=True)
    st.image([pre_image])
    model_selected = st.selectbox("Select Your Model: ",
                                  ['2 Classes (Simple Convolutional)', '10 Classes (EfficientNet-B0)', '101 Classes (EfficientNet-B1)'])

    if model_selected == '2 Classes (Simple Convolutional)':

        model_1 = returnModel()
        st.header("Summary model : ")
        model_1.summary(print_fn=lambda x: st.text(x))
        st.image([fit_2_image])
        st.success("Total running time for training process: ~41 s (97% accuracy train - 88% accuracy in test) ")
        st.header("Evaluate the model : ")
        history_load = np.load('./history/history_2.npy',
                               allow_pickle='TRUE').item()
        fig = plotHistory(history_load)
        st.write(fig)
    if model_selected == '10 Classes (EfficientNet-B0)':
        model_1 = returnModel(select=2)
        st.header("Summary model : ")
        model_1.summary(print_fn=lambda x: st.text(x))
        st.image([fit_10_image])
        st.success("Total running time for training process: ~20 m (88% accuracy train - 92% accuracy in test) ")
        st.header("Evaluate the model : ")
        history_load = np.load('./history/history_10.npy',
                               allow_pickle='TRUE').item()
        fig = plotHistory(history_load)
        st.write(fig)
    if model_selected == '101 Classes (EfficientNet-B1)':
        model_1 = returnModel(select=3)
        st.header("Summary model : ")
        model_1.summary(print_fn=lambda x: st.text(x))
        st.image([fit_101_image])
        st.success("Total running time for training process: ~140 m (97% accuracy train - 85% accuracy in test)")
        st.header("Evaluate the model : ")
        history_load = np.load('./history/history_101.npy',
                               allow_pickle='TRUE').item()
        fig = plotHistory(history_load)
        st.write(fig)
        st.write("F1 score of each class :")
        class_f1_scores = np.load(
            './history/score_101.npy', allow_pickle='TRUE').item()
        fig1 = plotF1(class_f1_scores)
        st.pyplot(fig1)
