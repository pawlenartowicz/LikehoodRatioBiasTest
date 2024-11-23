library(reticulate)
library(zcurve)

py_run_string("import pickle")

# Load the pickle file
py_run_string("with open('sample_to_tests2.pkl', 'rb') as file:
                  data = pickle.load(file)")

# Convert Python object to an R object
data_r <- py$data


results <- list()  # List to store results for each dataset

for (i in seq_along(data_r)) {
  print(i)
  # Extract the z-values or p-values (adjust to your column name)
  z_values <- as.numeric(data_r[[i]]$data)
  
  # Use tryCatch to handle errors during the Z-curve analysis
  result <- tryCatch({
    # Try running Z-curve analysis
    zcurve(z_values, bootstrap=FALSE)
  }, error = function(e) {
    # If an error occurs, return the error message
    message(paste("Error in dataset", i, ":", e$message))
    return(NA)  # Store NA or some other placeholder in case of an error
  })
  
  # Save the result or NA in case of an error
  results[[i]] <- result
  
}

missings <- list()  # List to store missing values

# Loop over the results
for (i in 1:2000) {
  r <- results[[i]]  # Access the result for dataset i
  
  # Check if the result is not NULL and is a valid zcurve object
  if (!is.null(r) && inherits(r, "zcurve")) {
    # Extract ODR and EDR estimates from the result
    odr <- ODR(r)$Estimate  # Observed Discovery Rate
    edr <- EDR(r)$Estimate  # Expected Discovery Rate
    
    # Calculate the missing value and store it in the list
    missings[[i]] <- 1 - edr / odr
  } else {
    # If result is NULL or not a valid zcurve object, store NA in the missings list
    missings[[i]] <- NA
  }
}

missings_df <- data.frame(missings = unlist(missings))

# Save as a CSV file
write.csv(missings_df, "results_r2.csv", row.names = FALSE)

