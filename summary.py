from keras.models import load_model

model = load_model("corrosion_detection_model.h5", compile=False)
model.summary()
