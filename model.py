import pickle

def price_prediction(data):
    model_path = "model.pkl"
    loaded_model = pickle.load(open(model_path, "rb"))

    # "data" is a row of input values:
    y_pred = loaded_model.predict(data)
    return y_pred[0]
