from torch.optim import SGD # Stochastic Gradient Descent to fit the neural network to the data
from Lightning.models.basic_nn import BasicNN_train


def train(model: BasicNN_train, inputs, labels):
    optimizer = SGD(model.parameters(), lr=0.1)
    print("Final bias, before optimization:" + str(model.final_bias.data) + "\n")

    for epoch in range(100):
        total_loss = 0

        for iteration in range(len(inputs)):
            input_i = inputs[iteration]
            label_i = labels[iteration]

            output_i = model(input_i)

            loss = (output_i - label_i) ** 2

            # We use loss.backward() to calculate the derivative of the loss function
            # with respect to the parameter or parameters we want to optimize
            # loss.backward() adds that to the previous derivative
            # loss.backward() accumulates the derivatives each time we go through the
            # nested loop and we need to keep this in mind
            loss.backward()

            total_loss += float(loss)

        if total_loss < 0.0001:
            print("Num steps: " + str(epoch))
            break

        # we take a small step towards a better value for parameters
        # optimizer.step() also has access to the derivatives stored in model and can use
        # them to step in the correct direction
        optimizer.step()
        # Now we need to zero out the derivatives that we're storing in model, and we do that with
        optimizer.zero_grad()

        print("Step: " + str(epoch) + ", Final Bias: " + str(model.final_bias) + "\n")
