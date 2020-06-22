# -*- coding: utf-8 -*-

import torch

def test_model(model, testloader, device):

    correct = 0
    total = 0

    for idx, (labels, inputs) in enumerate(testloader):
        # iter_batch = math.ceil(testloader/testloader.batch_size)
        # print(f'[phase: test] batch: {idx+1}/{iter_batch}', end='\r')

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.exp(outputs)
            _, predicted = torch.max(probabilities, 1)

            total = idx+1
            correct += torch.sum(predicted == labels.data)
            # print(f'{torch.max(torch.exp(outputs), dim=1)} - {torch.exp(outputs)}')

    print(f'[phase: test] acc: {total} {correct} {100*(correct.item()/total):.3f}')

def test_image(model, image, transform, device, labelencoder):

    input_tensor = transform(image).unsqueeze(0)
    inputs = input_tensor.to(device)
    outputs = model(inputs)
    probabilities = torch.exp(outputs)
    _, predicted = torch.max(outputs, 1)
    prob = (probabilities.cpu()).detach().numpy().flatten()

    print(f'class : {predicted.item()} {labelencoder.inverse_transform([predicted.item()])}')
    print(f'probabilities : {prob}')
