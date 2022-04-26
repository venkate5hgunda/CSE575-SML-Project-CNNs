import csv

epoch_index = []
training_loss = []
training_accuracy = []
training_top5_accuracy = []

validation_loss = []
validation_accuracy = []
validation_top5_accuracy = []

learning_rate = []
training_stamp = []
runtime = []

log_filename = 'vision-transformer_tf_training_log.txt'
output_filename = 'vision-transformer_tf_training_log.csv'

with open(log_filename, 'r') as log:
    for line in log:
        if line.startswith('Epoch ') and '/' in line:
            epoch_index.append(int(line.split(' ')[1].split('/')[0]))
        elif line.startswith('313/313 '):
            line_split = line.split()
            training_loss.append(float(line_split[7]))
            training_accuracy.append(float(line_split[10]))
            training_top5_accuracy.append(float(line_split[14]))
            if len(line_split)>15:
                validation_loss.append(float(line_split[17]))
                validation_accuracy.append(float(line_split[20]))
                validation_top5_accuracy.append(float(line_split[24]))
                learning_rate.append(float(line_split[27]))
                training_stamp.append(float(line_split[30]))
                runtime.append(float(line_split[33]))
            else:
                validation_loss.append(validation_loss[-1])
                validation_accuracy.append(validation_accuracy[-1])
                validation_top5_accuracy.append(validation_top5_accuracy[-1])
                learning_rate.append(learning_rate[-1])
                training_stamp.append(training_stamp[-1])
                runtime.append(runtime[-1])

with open(output_filename, 'w') as out_log:
    writer = csv.writer(out_log, delimiter=',')
    writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Training Top5 Accuracy', 'Validation Loss', 'Validation Accuracy', 'Validation Top5 Accuracy', 'Learning Rate', 'Training Stamp', 'Runtime'])
    for i in range(len(epoch_index)):
        writer.writerow([epoch_index[i], training_loss[i], training_accuracy[i], training_top5_accuracy[i], validation_loss[i], validation_accuracy[i], validation_top5_accuracy[i], learning_rate[i], training_stamp[i], runtime[i]])