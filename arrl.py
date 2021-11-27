from morse_model import MorseModel
from morse_data.morse_data import MorseData
import plotly.express as px

from prepare_file import prepare_file

file_string = "210112_20WPM.mp3"
batch_size = 64
sample_length = 25000

input, input_lengths = prepare_file(file_string, 64, 25000)

# fig = px.imshow(input[0, :, :].transpose_(1, 0))
# fig.show()


model = MorseModel()
model.save_model()


print("debug")
