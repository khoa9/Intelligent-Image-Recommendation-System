library(FlickrAPI)

library(dplyr)

library(readr)

library(httr)

setFlickrAPIKey(api_key = "5d2eb50179ff9b98af4d203e5b79393b", install = TRUE, overwrite = TRUE)
readRenviron("~/.Renviron")


# Function to get photos with pagination and avoiding duplicates

get_photos_with_pagination <- function(tags, per_page = 500, total_photos = 4000) {
  all_photos <- list()
  # Set initial page
  page <- 1
  while (length(all_photos) < total_photos) {
    # Make API call with pagination
    api_result <- getPhotoSearch(
      sort = "date-posted-asc",
      tags = tags,
      extras = c("url_z", "geo", "tags"),
      tag_mode = 'all',
      safe_search = 1,
      content_types = 0,
      has_geo = 1,
      per_page = per_page,
      in_gallery = TRUE,
      page = page
    )
    # Checking if API call was successful
    if (!is.null(api_result) && length(api_result) > 0) {
      # Filter out duplicates
      new_photos <- api_result[!duplicated(api_result$id), ]
      
      # Filter photos with more than 7 words in tags
      # This is to avoid photos that are spam, which include all the possible tags
      new_photos <- new_photos[sapply(strsplit(new_photos$tags, " "), function(x) length(x) <= 7), ]
      
      # Combining results into a list
      all_photos <- c(all_photos, new_photos)
      
      # Incrementing page for the next API call
      page <- page + 1
    } else {
      # If API call fails or no more photos, breaking out of the loop
      break
    }
  }
  return(all_photos)
}


tags <- c("skyscrapper","Skyscrapper","skyscrappers")
per_page <- 500
total_photos <- 1500

# Get photos with pagination and avoiding duplicates
photos <- get_photos_with_pagination(tags, per_page, total_photos)


combined_photos <- lapply(split(photos, names(photos)), unlist)
photos_df <- data.frame(combined_photos)

sample = photos_df |> 
  select(id, secret, tags, url_z, owner, latitude, longitude, title)

a = distinct(sample)


# Define the words to filter directly in the code
words_to_filter <- c("human", "woman", "man","party") 

# Create a single regex pattern from the words
pattern <- paste(words_to_filter, collapse = "|")

# Create the new dataframe by filtering out rows with tags containing the defined words
photo_search <- a %>% 
  filter(!grepl(pattern, tags, ignore.case = TRUE))

# Extract the URLs
urls <- photo_search$url_z

# to remove rows with no urls
urls <- na.omit(urls)

# Define the folder where images will be saved - RENAME IF NECESSARY
download_folder <- "skyscrapper"

# Function to download images from URLs and rename them
download_images <- function(urls, download_folder, keyword = "city") {
  # Create download folder if it doesn't exist
  if (!dir.exists(download_folder)) {
    dir.create(download_folder, recursive = TRUE)
  }
  for (i in seq_along(urls)) {
    # Get the URL
    url <- urls[i]
    # Generate the new file name using the keyword and index
    file_name <- sprintf("%s_%d.jpg", keyword, i)
    # Define the destination file path
    destfile <- file.path(download_folder, file_name)
    # Download the file
    download.file(url, destfile, mode = "wb")
    # Print a message to indicate progress
    cat(sprintf("Downloaded and renamed: %s\n", file_name))
  }
}

# Call the function to download the images
# Replace 'download_folder' with your desired folder name
download_images(urls, download_folder)
