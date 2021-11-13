library(behaviorchange)

dat <- get(data("iris"))

dat$Species_int <- as.numeric(dat$Species)

# dat[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")] <-
#   lapply(
#     dat[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")],
#     function(x) {
#       lowestObservation <- min(x, na.rm = TRUE);
#       highestObservation <- max(x, na.rm = TRUE);
#       return(
#         (x - lowestObservation) / highestObservation
#       );
#     }
#   );

output<-behaviorchange::determinant_selection_table(
data=dat,
determinants = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"),
target = 'Species_int',
sortBy = 6
)

getwd()

write.csv(output, "pci_iris_example.csv")
