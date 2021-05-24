import shutil, random, os
import zipfile

for foldername in os.listdir("./Data"):
    data_source = f'./Data/{foldername}/'
    testing_dir = f'./Testing/{foldername}/'

    no_of_images = len(os.listdir(data_source))
    testing_sample_size = int(no_of_images * 0.3)

    images = random.sample(os.listdir(data_source), testing_sample_size)
    for image in images:
        data_source_path = os.path.join(data_source, image)
        os.makedirs(os.path.dirname(testing_dir), exist_ok=True)
        shutil.move(data_source_path, testing_dir + image)

# shutil.make_archive('Training', 'zip', 'Data')
# shutil.make_archive('Testing', 'zip', 'Testing')