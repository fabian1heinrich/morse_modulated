from torch.optim.lr_scheduler import StepLR


def train_model(model,
                data_loader,
                n_epochs,
                optimizer,
                criterion):

    print("start training")

    lr_scheduler = StepLR(optimizer, gamma=0.85, step_size=1)
    for epoch in range(n_epochs):

        loss = 0
        accuracy = 0
        train_loss = 0
        for input, input_lengths, targets, target_lengths in data_loader:

            optimizer.zero_grad()

            y_pred = model(input.cuda())

            train_loss = criterion(
                y_pred, targets, input_lengths, target_lengths)

            train_loss.backward()
            optimizer.step()

            loss = train_loss.item()

        lr_scheduler.step()
        loss = loss
        accuracy = model.calc_symbol_accuracy(
            input, targets, input_lengths, target_lengths)

        print("epoch #{}/{} | loss = {:.6f} | accuracy = {:.2f}".format(
            epoch + 1, n_epochs,
            loss,
            accuracy))
