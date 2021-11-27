import torch


def predict(model, input, input_lengths):

    # model.eval()

    prediction = model(input.cuda())

    _, max_index = torch.max(prediction, dim=2)
    max_index = max_index.to("cpu")

    for i, cur_seq in enumerate(max_index.transpose_(1, 0)):

        pred = cur_seq[0:input_lengths[i]]
        pred = torch.unique_consecutive(pred)
        pred = pred[pred.nonzero(as_tuple=True)]

        pred = pred.tolist()

        print("".join([chr(x+96) for x in pred]))
