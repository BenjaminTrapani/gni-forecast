install.packages('GGally',dependencies = TRUE)
require(GGally)

setwd("/Users/benjamintrapani/Dropbox/NortheasternUniversity/SP17/machine-learning/ml-nba-final-project")

playerData <- read.csv("data/nba_basketball_data.csv")
playerData

seasonData <- read.csv("data/season_logs.csv")
seasonData

econData <- read.csv("data/cleaned-econ-data.csv")
econData['CountryName']
econData['Year']
econData['Population..total']
ubPresent <- subset(econData, econData[,'Population..total_present'] == 1)
dim(ubPresent)
preds = subset(ubPresent, select = -c(Population..total))
options(max.print=5.5E5)
head(ubPresent)
initialFit = glm(Population..total~., data=ubPresent)
initialFit
