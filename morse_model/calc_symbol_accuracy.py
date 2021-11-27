import torch


def calc_symbol_accuracy(model, input, target, input_lengths, target_lengths):

    # model.eval()
    counter = 0
    accuracy = 0

    prediction = model(input.cuda())
    target_cpu = target.to("cpu")

    _, max_index = torch.max(prediction, dim=2)
    max_index = max_index.to("cpu")

    for i, cur_seq in enumerate(max_index.t()):
        pred = cur_seq[0:input_lengths[i]]
        pred = torch.unique_consecutive(pred)
        pred = pred[pred.nonzero(as_tuple=True)]

        pred = pred.tolist()
        t = target_cpu[i].tolist()

        accuracy += sum(cur_p == cur_t for cur_p, cur_t in zip(pred, t))
        counter += target_lengths[i]

    return accuracy / counter
