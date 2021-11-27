import plotly.express as px


def show_spectrogram(morse_data):

    sample, y = morse_data.__getitem__(1)
    print(y)
    fig = px.imshow(sample.transpose_(1, 0))
    fig.show()
