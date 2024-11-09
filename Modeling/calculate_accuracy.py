def calculate_accuracy_from_file(input_file):
    correct = 0
    total = 0
    skipped = 0

    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if 'OUTPUT:' in line and 'ANSWER:' in line:
            # Extract the OUTPUT and ANSWER strings
            output_part = line.split('OUTPUT: ')[1].split(' | ANSWER: ')[0].strip()
            answer_part = line.split('ANSWER: ')[1].split(' | FNAME: ')[0].strip()
            # Increase the total sample count
            total += 1
            # Check if OUTPUT matches ANSWER
            if output_part == answer_part:
                correct += 1
            else:
                skipped += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    # Print and write the results to the file
    print("#####################")
    print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, accuracy))
    print("Skipped = {}".format(skipped))


# Example usage
input_file = "C:\\Users\\angel\\PycharmProjects\\pythonProject\\Modeling\\Outputs\\GPT2\\test"
calculate_accuracy_from_file(input_file)
