install.packages('GGally',dependencies = TRUE)
require(GGally)

setwd("/Users/benjamintrapani/Dropbox/NortheasternUniversity/SP17/machine-learning/ml-nba-final-project")

playerData <- read.csv("data/nba_basketball_data.csv")
playerData

seasonData <- read.csv("data/season_logs.csv")
seasonData

mergedData <- merge(playerData, seasonData, by = "game_id")
mergedData

