def accuracy(dataset, model, batch_size=32):
    # Testing the model and returning the accuracy on the given dataset
    total = 0
    correct = 0
    for batch in data.BucketIterator(dataset=dataset, batch_size=batch_size):
        output = model(batch.features)
        total += len(batch.label)
        prediction = (output > 0).long()
        correct += (prediction == batch.label).sum()

    return float(correct) / total  
    
def check_that_importing_is_working():
    print("Yes, it's alright')