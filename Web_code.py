from flask import Flask, request, render_template
from load_data import deal_input_data, recognize

app = Flask(__name__)

'''@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["input_text"]
    print(text)
    info = get_prediction(text)
    return jsonify(info)'''


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        text = request.form.get("input_text")
        print(text)
        test_data_path = './CLUE2020/testpre.char.bmes'
        f = open(test_data_path, "w", encoding='utf-8')
        f.write(text)
        f.close()

        deal_input_data(test_data_path)

        info = ner_predict()
        # print(info)
        return render_template("index.html", input_text=text, Result=info)
    return render_template("index.html")


def ner_predict():
    from pre_test import predictor, datasets
    test_label_list = predictor.predict(datasets['test'][:])['pred']  # 预测结果
    test_raw_char = datasets['test'][:]['raw_chars']  # 原始文字

    entity_list = []
    for i, j in zip(test_label_list, test_raw_char):
        entity = recognize(i, j)
        if len(entity) != 0:
            entity_list.append(entity)
    for i in entity_list:
        print(i)

    return entity_list


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
