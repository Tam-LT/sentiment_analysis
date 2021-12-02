from flask import Flask, render_template, request
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
app = Flask(__name__)
@app.route('/')
def my_form():  # put application's code here
    return render_template('home.html')

@app.route('/', methods=['POST'])
def my_form_post():
    def standardize_data(row):
        # Xóa dấu chấm, phẩy, hỏi ở cuối câu
        row = re.sub(r"[\.,\?]+$-", "", row)
        # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
        row = row.replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("“", " ") \
            .replace(":", " ").replace("”", " ") \
            .replace('"', " ").replace("'", " ") \
            .replace("!", " ").replace("?", " ") \
            .replace("-", " ").replace("?", " ")
        row = row.strip()
        return row

    text1 = request.form['text1'].lower()
    df = pd.read_csv("data - data.csv")
    tweet_df = df[['comment', 'label']]
    tweet_df = tweet_df[tweet_df['label'] != 'NEU']
    dt = tweet_df.comment.apply(standardize_data)
    tweet = dt.values
    sentiment_label = ['Positive', 'Negative']
    model = tf.keras.models.load_model('model.h5')
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweet)
    encoded_docs = tokenizer.texts_to_sequences(tweet)
    padded_sequence = pad_sequences(encoded_docs, maxlen=200)
    tw = tokenizer.texts_to_sequences([text1])
    tw = pad_sequences(tw, maxlen=200)
    prediction = int(model.predict(tw).round().item())
    final = sentiment_label[prediction]
    if final == 'Negative':
        t = 'Negative ☹️☹️☹️'
        return render_template('home.html', final=t, text1=text1)
    elif final == 'Positive':
        t = 'Positive 🙂🙂🙂'
        return render_template('home.html', final=t, text1=text1)
    return render_template('home.html', final=final, text1=text1)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)

