### BIOS 740 Computing HW 3 Code
### Andrew Walther
### November 17, 2021
##############################################################################
#set randomization seed
set.seed(2021)
#load relevant packages
library(tidyverse)
library(keras)
library(tensorflow)
##############################################################################
#Extract subfilepaths from source & confirm image counts
PetImages_Dirs <- list.dirs(path = "~/OneDrive - University of North Carolina at Chapel Hill/UNC - Fall 2021/BIOS 740 - Kosorok/Programming Assignments/Computing HW3/catdogdata/PetImages", 
                            full.names = TRUE, recursive = TRUE)
directory_object_counts <- sapply(PetImages_Dirs, function(dir){length(list.files(dir))})
##############################################################################
#set up source, training, & testing directories via filepaths
#source (contains all cat & dog images)
root_directory <- "~/OneDrive - University of North Carolina at Chapel Hill/UNC - Fall 2021/BIOS 740 - Kosorok/Programming Assignments/Computing HW3/catdogdata/PetImages"
source_dogs_directory <- "~/OneDrive - University of North Carolina at Chapel Hill/UNC - Fall 2021/BIOS 740 - Kosorok/Programming Assignments/Computing HW3/catdogdata/PetImages/Dog"
source_cats_directory <- "~/OneDrive - University of North Carolina at Chapel Hill/UNC - Fall 2021/BIOS 740 - Kosorok/Programming Assignments/Computing HW3/catdogdata/PetImages/Cat"
#set up training directories (cats & dogs)
training_directory <- file.path(root_directory, "training")
dir.create(training_directory)
training_dogs_directory <- file.path(training_directory, "dogs")
dir.create(training_dogs_directory)
training_cats_directory <- file.path(training_directory, "cats")
dir.create(training_cats_directory)
#set up testing directories (cats & dogs)
testing_directory <- file.path(root_directory, "testing")
dir.create(testing_directory)
testing_dogs_directory <- file.path(testing_directory, "dogs")
dir.create(testing_dogs_directory)
testing_cats_directory <- file.path(testing_directory, "cats")
dir.create(testing_cats_directory)
##############################################################################
#Data splitting & randomization, assignment to train/test
split_data <- function(source_dir, training_dest, testing_dest, split_size){
        #file paths for cat/dog images
        files <- list.files(path = source_dir, full.names = T)
        size <- file.size(files)
        #removing images with size zero & randomize remaining images
        shuffled_set <- cbind (files, size) %>% subset(size > 0, select = c(files)) %>% 
                as.character() %>% sample(replace = F)
        #select X% of images into training & remaining to testing
        training_length <- length(shuffled_set) * split_size
        testing_length <- length(shuffled_set) * (1 - split_size)
        training_set <- shuffled_set[1:training_length]
        testing_set <- shuffled_set[(training_length+1):length(shuffled_set)]
        #move train/test sets to appropriate directory (hide w/invisible)
        invisible(file.copy(from = training_set, to = training_dest))
        invisible(file.copy(from = testing_set, to = testing_dest))}
#Cat image splitting (test/train)
split_data(source_dir = source_cats_directory,
           training_dest = training_cats_directory,
           testing_dest = testing_cats_directory,
           split_size = 0.9)
#Dog image splitting (test/train)
split_data(source_dir = source_dogs_directory,
           training_dest = training_dogs_directory,
           testing_dest = testing_dogs_directory,
           split_size = 0.9)
#Check counts for dogs & cats (training = 11249 (90%) & testing = 1249 (10%))
cat("Cat training images:", length(list.files(training_cats_directory)), '\n')
cat("Cat testing images:", length(list.files(testing_cats_directory)), '\n')
cat("Dog training images:", length(list.files(training_dogs_directory)), '\n')
cat("Dog testing images:", length(list.files(testing_dogs_directory)), '\n')
##############################################################################
#set up CNN w/ convolution, pooling, flatten, & dense layers
model <- keras_model_sequential() %>%
        #convolution & pooling layers 1 (150x150 images with RGB color (3))
        layer_conv_2d(input_shape = c(150, 150, 3), filters = 16, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        #convolution & pooling layers 2
        layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        #convolution & pooling layers 3
        layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
        layer_max_pooling_2d(pool_size = c(2, 2)) %>%
        #flatten layer 1
        layer_flatten() %>%
        #dense layers 1 & 2
        layer_dense(units = 512, activation = "relu") %>%
        #binary classification needs sigmoid activation function
        layer_dense(units = 1, activation = "sigmoid") %>%
        #compile model
        compile(loss = 'binary_crossentropy', optimizer = optimizer_rmsprop(lr = 0.001), metrics = 'accuracy')
summary(model)
##############################################################################
#Training: Rescale RGB by 1/255
train_datagen <- image_data_generator(rescale = 1/255)
train_generator <- flow_images_from_directory(
        #target directory
        directory = training_directory,
        #training data generator
        generator = train_datagen,
        #resize images to 150x150 pixels
        target_size = c(150, 150),
        #Input batches of 250 images
        batch_size = 250,
        #binary class for (binary_crossentropy)
        class_mode = 'binary')
#Validation(testing): Rescale RGB by 1/255
validation_datagen <- image_data_generator(rescale = 1/255)
validation_generator <- flow_images_from_directory(
        #target directory
        directory = testing_directory,
        #testing data generator
        generator = validation_datagen,
        #resize images to 150x150 pixels
        target_size = c(150, 150),
        #Input batches of 250 images
        batch_size = 250,
        #binary class for (binary_crossentropy)
        class_mode = 'binary')
#check contents of first training batch (set of 250, 150x150 etc.)
batch_train <- generator_next(train_generator)
str(batch_train)
#check contents of first testing batch (set of 250, 150x150 etc.)
batch_test <- generator_next(validation_generator)
str(batch_test)
##############################################################################
start.time <- Sys.time()
#model fitting object
model_history <- model %>% fit_generator(
        generator = train_generator,
        #number of size 250 batches in each epoch
        steps_per_epoch = 50,
        #iterations over all of data
        epochs = 5,
        validation_data = validation_generator,
        validation_steps = 5)
#print out best loss and its corresponding accuracy
epoch <- which.min(model_history$metrics$val_loss)
loss <- round(model_history$metrics$val_loss[epoch],3) 
accuracy <- round(model_history$metrics$val_accuracy[epoch],3)
print(loss)
print(accuracy)
#plot training results
plot(model_history)
summary(model_history)
#function runtime
end.time <- Sys.time()
time.elapsed <- end.time-start.time
print(time.elapsed)