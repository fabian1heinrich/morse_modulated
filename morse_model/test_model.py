def test_model(model, data_loader):

    accuracy = 0

    for batch_x, input_lengths, batch_y, target_lengths in data_loader:

        accuracy += model.calc_symbol_accuracy(
            batch_x,
            batch_y,
            input_lengths,
            target_lengths)

    return accuracy / len(data_loader)
