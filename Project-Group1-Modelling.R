#setwd("C:\\CIS8398\\Group1_Project")
getwd()

install.packages("imager")
library(keras)
library(httr)
library(imager)
library(jsonlite)


# Example code to extract images from a zip file
zip_file_path <- "flickrdata.zip"
extract_path <- "flickrdata"

unzip(zip_file_path, exdir = extract_path)

# Create the dataset directory 
original_dataset_dir <- "flickrdata/flickrdata" # we will only use the labelled data
base_dir <- "flickrdata_sub" # to store a subset of data that we are going to use
dir.create(base_dir)
train_dir <- file.path(base_dir, "train")
dir.create(train_dir)
validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)
test_dir <- file.path(base_dir, "test")
dir.create(test_dir)


train_mountain_dir <- file.path(train_dir, "mountain")
dir.create(train_mountain_dir)
train_river_dir <- file.path(train_dir, "river")
dir.create(train_river_dir)
train_city_dir <- file.path(train_dir, "city")
dir.create(train_city_dir)


validation_mountain_dir <- file.path(validation_dir, "mountain")
dir.create(validation_mountain_dir)
validation_river_dir <- file.path(validation_dir, "river")
dir.create(validation_river_dir)
validation_city_dir <- file.path(validation_dir, "city")
dir.create(validation_city_dir)


test_mountain_dir <- file.path(test_dir, "mountain")
dir.create(test_mountain_dir)
test_river_dir <- file.path(test_dir, "river")
dir.create(test_river_dir)
test_city_dir <- file.path(test_dir, "city")
dir.create(test_city_dir)


fnames <- paste0("mountain_", 1:600, ".jpg")
# use invisible to supress output message from file.copy()
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(train_mountain_dir)))
fnames <- paste0("mountain_", 601:800, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(validation_mountain_dir)))
fnames <- paste0("mountain_", 801:900, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                   file.path(test_mountain_dir)))


fnames <- paste0("river_", 1:600, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(train_river_dir)))
fnames <- paste0("river_", 601:800, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(validation_river_dir)))
fnames <- paste0("river_", 801:900, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                   file.path(test_river_dir)))


fnames <- paste0("city_", 1:600, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(train_city_dir)))
fnames <- paste0("city_", 601:800, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(validation_city_dir)))
fnames <- paste0("city_", 801:900, ".jpg")
invisible(file.copy(file.path(original_dataset_dir, fnames),
                    file.path(test_city_dir)))

#Folder - flickrdata_sub
class_labels <- sort(dir(train_dir))


train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)
test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir, # Target directory
  train_datagen, # Training data generator
  target_size = c(150, 150), # Resizes all images to 150 × 150
  batch_size = 32, # 32 samples in one batch
  class_mode = "categorical" 
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "categorical"
)

# Create a convolutional neural network (CNN) model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Add more convolutional layers
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") |>
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # Flatten the output before dense layers
  layer_flatten() %>%
  
  # Add more dense layers
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  
  # Output layer
  layer_dense(units = 3, activation = 'softmax')


# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  metrics = c('categorical_accuracy')
)

# Train the model using the generator
history_v1 <- model %>% fit(
    train_generator, # Adjust as needed
    epochs = 10,
    validation_data = validation_generator, # Adjust as needed
  )

save_model_hdf5(model,'convNN.h5')
saveRDS(history_v1, "history_convNN.rds")

# Create model_v2 using VGG16
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)


datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20

extract_features <- function(directory, sample_count, num_classes) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count, num_classes))
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "categorical"  # Use categorical class mode for multi-class classification
  )
  i <- 0
  while (TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size) + 1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range,] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count) break
  }
  return(list(features = features, labels = labels))
}

train <- extract_features(train_dir, 1800,3) # will take a while since we are running
validation <- extract_features(validation_dir, 600,3) # our images through conv_base
test <- extract_features(test_dir,300,3)

# Reshape the features
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# Create model 
model_v2 <- keras_model_sequential() |>
  layer_dense(units = 256, activation = "relu", input_shape = 4 * 4 * 512) |>
  layer_dropout(rate = 0.5) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dropout(rate = 0.3) |>
  layer_dense(units = 64, activation = "relu") |>
  layer_dropout(rate = 0.2) |>
  layer_dense(units = 3, activation = "softmax")


model_v2 |> compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("categorical_accuracy")
)


history_v2 <- model_v2 |> fit(
  train$features, train$labels,
  epochs = 10,
  batch_size = 32,
  validation_data = list(validation$features, validation$labels)
)

save_model_hdf5(model_v2,'VGG_16.h5')
saveRDS(history_v2, "history_vgg_16.rds")

#plot(history_v1)
#plot(history_v2)

#fnames <- list.files(test_dir, full.names = TRUE)
#img_path <- fnames[[3]] # Chooses one image

#img_path <- "flickrdata\\train"

#images <- list.files(img_path,full.names = TRUE)
#image_to_identify <- images[[6000]]
#img <- image_load(image_to_identify, target_size = c(150, 150))
#img_data <- image_to_array(img)
#image_data <- array_reshape(img_data, c(1, dim(img_data)))

#Image display

saveRDS(test,'test.rds')

prediction = model_v2 |> predict(test$features)
predicted_class <- which.max(prediction[20,])
predicted_label= class_labels[predicted_class]
print(predicted_label)


# Make predictions
#predictions <- predict(model, image_data)
# Print the predictions
#print(predictions)
# Get the predicted class index
#predicted_class <- which.max(predictions)
# Print the predicted class
#print(class_labels[predicted_class])

#Below is the code to retrieve images data from Flickr API based on the classified image. 

call_flickr_api <- function(search_term) {
  
  api_key <- "5d2eb50179ff9b98af4d203e5b79393b"
  base_url <- "https://www.flickr.com/services/rest/"
  # Parameters for Flickr API request
  params <- list(
    method = "flickr.photos.search",
    api_key = api_key,
    text = search_term,
    format = "json",
    nojsoncallback = 1
  )
  # Make the API request
  response <- GET(url = base_url, query = params)
  # Check for successful response
  if (http_status(response)$category == "Success") {
    return(content(response, "parsed"))
  } else {
    stop("Error calling Flickr API")
  }
}
# Call Flickr API with the predicted class label
flickr_response <- call_flickr_api(predicted_label)

# Print the Flickr API response
#print(flickr_response)

photos <- flickr_response$photos$photo

# Extract image URLs
image_urls <- lapply(photos[1:10], function(photo) {
  farm <- photo$farm
  server <- photo$server
  id <- photo$id
  secret <- photo$secret
  paste0("https://farm", farm, ".staticflickr.com/", server, "/", id, "_", secret, ".jpg")
})

urls <- lapply(image_urls, as.character)

# Download images
download_images <- function(urls, destination_folder = "downloaded_images") {
  dir.create(destination_folder, showWarnings = FALSE, recursive = TRUE)
  for (i in seq_along(urls)) {
    response <- GET(as.character(urls[i]), write_disk(paste0(getwd(), "\\downloaded_images", "/", sprintf("image_%03d.jpg", i))))
    if (http_status(response)$category != "Success") {
      warning(paste("Failed to download image", i))
    }
  }
}

# Download 10 images
download_images(urls)
